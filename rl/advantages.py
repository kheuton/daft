"""
advantages.py — Per-sample advantage computation for DAFT RL training.

Implements GRPO (A1), pass@k closed-form (A2), and vote_k exact DP (A3)
advantage estimators as specified in DESIGN.md sections 3, A1-A3.

All functions return float64 numpy arrays. NO normalization is applied here;
the caller applies static per-arm scale constants (see DESIGN.md §3).
"""

from __future__ import annotations

import itertools
from math import comb
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_advantages(
    correct: np.ndarray,
    class_ids: np.ndarray,
    mode: Literal[
        "grpo", "pass_at_k", "vote_k", "vote_passk_hybrid",
        "prm_margin", "prm_weighted", "prm_weighted_pm",
    ],
    k: int,
    hybrid_lambda: float = 0.25,
    verifier_scores: np.ndarray | None = None,
    prm_exponent: float = 2.0,
    empty_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-sample advantages for a batch of groups.

    Parameters
    ----------
    correct : (B, G) array of {0, 1} int/float — binary correctness per sample.
    class_ids : (B, G) int array — equivalence-class ids of canonical answers
        within each group (e.g. from rewards.class_ids_from_canonicals).
        Unparseable answers form their own normal class.
    mode : one of 'grpo', 'pass_at_k', 'vote_k', 'vote_passk_hybrid' (Round-1
        arms) or the Phase-2 verifier-aware arms 'prm_margin', 'prm_weighted',
        'prm_weighted_pm' (require verifier_scores).
    k : subset size for pass@k / vote_k modes.  Ignored for 'grpo' and prm modes.
    hybrid_lambda : weight on pass@k term in vote_passk_hybrid mode.
    verifier_scores : (B, G) float array in [0, 1] — the per-completion AGGREGATED
        PRM score (the deployed weighted-vote keys on weight w = clip(score,0)^p).
        Required for the prm_* modes; ignored otherwise.
    prm_exponent : p in the deployed PRM-weighted-vote weight w_i = clip(score,0)^p.
        Used only by the prm_* modes (DESIGN: match the deployed rule's exponent).

    Returns
    -------
    advantages : (B, G) float64 array.  Not normalized.

    Phase-2 verifier-aware arms (decision-aware credit for the deployed
    PRM-weighted-vote(p, drop-empty) rule; utility tied to GROUND-TRUTH
    correctness so the PRM cannot manufacture reward for wrong answers):
      * 'prm_margin' (D3, primary): per group let w_i = clip(score_i,0)^p and the
        deployed class mass mass_j = sum of w over class j. Credit +w_i to members
        of the TOP-MASS CORRECT class and -w_i to members of the TOP-MASS WRONG
        class (0 elsewhere), then subtract the group mean. This is the dense,
        vote-aware credit for the argmax-of-class-mass decision: it pushes the
        top correct class above the confidently-wrong plurality. Zeroed on
        all-correct / all-wrong groups (parity with grpo, which has no signal
        there).
      * 'prm_weighted' (D2): reward_i = w_i * correct_i; A_i = reward - mean.
        Per-sample correct-side reweighting only (no wrong-side pressure).
      * 'prm_weighted_pm' (D2pm): reward_i = w_i * (2*correct_i - 1); A = r - mean.
        Symmetric: rewards correct by +w, penalizes wrong by -w.
    """
    correct = np.asarray(correct, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    B, G = correct.shape

    if mode == "grpo":
        return _grpo_advantages(correct)

    if mode == "pass_at_k":
        assert 1 <= k <= G - 1, f"pass_at_k requires 1 <= k <= G-1, got k={k}, G={G}"
        return _passk_advantages(correct, k)

    if mode == "vote_k":
        assert 1 <= k <= G - 1, f"vote_k requires 1 <= k <= G-1, got k={k}, G={G}"
        return _votek_advantages(correct, class_ids, k)

    if mode == "vote_passk_hybrid":
        assert 1 <= k <= G - 1, f"vote_passk_hybrid requires 1 <= k <= G-1, got k={k}, G={G}"
        A_vote = _votek_advantages(correct, class_ids, k)
        A_passk = _passk_advantages(correct, k)
        return A_vote + hybrid_lambda * A_passk

    if mode in ("prm_margin", "prm_weighted", "prm_weighted_pm"):
        assert verifier_scores is not None, f"mode {mode!r} requires verifier_scores"
        vs = np.asarray(verifier_scores, dtype=np.float64)
        assert vs.shape == correct.shape, (
            f"verifier_scores shape {vs.shape} != correct shape {correct.shape}"
        )
        w = np.clip(vs, 0.0, None) ** float(prm_exponent)  # (B,G) deployed weights
        if mode == "prm_weighted":
            return _prm_weighted_advantages(correct, w)
        if mode == "prm_weighted_pm":
            return _prm_weighted_pm_advantages(correct, w)
        em = None if empty_mask is None else np.asarray(empty_mask, dtype=bool)
        return _prm_margin_advantages(correct, class_ids, w, empty_mask=em)

    raise ValueError(f"Unknown mode: {mode!r}")


def pass_at_n_unbiased(num_correct: int, n_total: int, k: int) -> float:
    """Unbiased pass@k estimator (Codex/HumanEval formula).

    R̂(C, G) = 1 - C(G-C, k) / C(G, k)

    Parameters
    ----------
    num_correct : number of correct completions (C).
    n_total : total completions in group (G).
    k : target k.

    Returns
    -------
    float — unbiased pass@k estimate in [0, 1].
    """
    C, G = int(num_correct), int(n_total)
    denom = comb(G, k)
    if denom == 0:
        return 0.0
    numer = _safe_comb(G - C, k)
    return float(1.0 - numer / denom)


def maj_at_k_exact(correct: np.ndarray, class_ids: np.ndarray, k: int) -> float:
    """Exact E[u(maj(S))] averaged over all C(G, k) subsets.

    Uses hypergeometric DP over class-count vectors.  Tie-break utility:
    u = 1[correct class in argmax set] / |argmax set|.

    Parameters
    ----------
    correct : (G,) array of {0, 1}.
    class_ids : (G,) int array of equivalence-class ids.
    k : subset size.

    Returns
    -------
    float — exact expected majority utility.
    """
    correct = np.asarray(correct, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    G = len(correct)
    assert 1 <= k <= G - 1, f"maj_at_k_exact requires 1 <= k <= G-1, got k={k}, G={G}"

    class_info = _extract_class_info(correct, class_ids)
    return _maj_at_k_exact_from_info(class_info, G, k)


def maj_at_k_estimate(
    correct: np.ndarray,
    class_ids: np.ndarray,
    k: int,
    max_exact_classes: int = 10,
    n_subsets: int = 500,
    seed: int = 0,
) -> float:
    """E[u(maj(S))] — exact DP when the class count is small, else seeded
    Monte-Carlo over random k-subsets.

    The exact DP enumerates multivariate-hypergeometric count vectors, which is
    intractable on eval groups (G=64 samples, tens of distinct classes — the
    test1500 metrics phase burned hours before this dispatcher existed).
    MC with 500 subsets gives ~0.02 std on a {0,1} utility, well under the
    between-problem noise, and matches DESIGN sec 5 (>=200 subsets).
    """
    correct = np.asarray(correct, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    G = len(correct)
    assert 1 <= k <= G - 1, f"maj_at_k_estimate requires 1 <= k <= G-1, got k={k}, G={G}"

    n_classes = len(np.unique(class_ids))
    if n_classes <= max_exact_classes:
        return maj_at_k_exact(correct, class_ids, k)

    rng = np.random.default_rng(seed)
    correct_classes = set(class_ids[correct > 0.5].tolist())
    total = 0.0
    for _ in range(n_subsets):
        idx = rng.choice(G, size=k, replace=False)
        sub = class_ids[idx]
        vals, counts = np.unique(sub, return_counts=True)
        top = counts.max()
        argmax = vals[counts == top]
        n_corr_in_argmax = sum(1 for v in argmax if v in correct_classes)
        total += n_corr_in_argmax / len(argmax)
    return total / n_subsets


# ---------------------------------------------------------------------------
# Brute-force reference implementations (used only in tests)
# ---------------------------------------------------------------------------

def brute_force_passk_advantages(correct: np.ndarray, k: int) -> np.ndarray:
    """Brute-force LOO pass@k advantages via leave-one-out of the group estimator.

    For each sample i, computes the LOO difference of the pass@k group estimator:
        A_i = R̂(C, G) - R̂(C - c_i, G - 1)
    where R̂(C, G) = 1 - C(G-C, k) / C(G, k) is the unbiased pass@k estimator,
    C = sum of correct in group, and c_i = 1 if sample i is correct else 0.

    This is the quantity the closed-form formula computes exactly.  The
    'brute force' here means computing it directly from group statistics
    (not from the closed-form algebra), which independently verifies the
    closed-form formulas in _passk_advantages.

    Note: an alternative 'subset conditional' brute force computing
    E[1_pass | i in S] - E[1_pass | i not in S] gives a DIFFERENT (larger
    by G/k) quantity, which is not the DESIGN-specified LOO advantage.
    The DESIGN specifies the LOO of R̂(C,G), which is what the gradient test
    confirms gives the correct policy gradient direction for pass@k.
    """
    correct = np.asarray(correct, dtype=np.float64)
    G = len(correct)
    C = int(correct.sum())
    denom_G = _safe_comb(G, k)
    advantages = np.zeros(G, dtype=np.float64)

    if denom_G == 0:
        return advantages

    R_full = 1.0 - _safe_comb(G - C, k) / denom_G

    for i in range(G):
        ci = int(correct[i] > 0.5)
        C_minus_i = C - ci
        G_minus_i = G - 1
        denom_G1 = _safe_comb(G_minus_i, k)
        if denom_G1 == 0:
            R_loo = 0.0
        else:
            R_loo = 1.0 - _safe_comb(G_minus_i - C_minus_i, k) / denom_G1
        advantages[i] = R_full - R_loo

    return advantages


def brute_force_votek_advantages(
    correct: np.ndarray, class_ids: np.ndarray, k: int
) -> np.ndarray:
    """Brute-force LOO vote@k advantages by enumerating all subsets.

    A_i = E_{S∋i,|S|=k}[u(maj(S))] - E_{S∌i,|S|=k}[u(maj(S))]
    """
    correct = np.asarray(correct, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    G = len(correct)
    indices = list(range(G))
    advantages = np.zeros(G, dtype=np.float64)

    for i in range(G):
        rest = [j for j in indices if j != i]

        with_i_vals = []
        for sub in itertools.combinations(rest, k - 1):
            subset = [i] + list(sub)
            u = _subset_majority_utility(subset, correct, class_ids)
            with_i_vals.append(u)

        without_i_vals = []
        for sub in itertools.combinations(rest, k):
            u = _subset_majority_utility(list(sub), correct, class_ids)
            without_i_vals.append(u)

        mean_with = np.mean(with_i_vals) if with_i_vals else 0.0
        mean_without = np.mean(without_i_vals) if without_i_vals else 0.0
        advantages[i] = mean_with - mean_without

    return advantages


def brute_force_maj_at_k_exact(
    correct: np.ndarray, class_ids: np.ndarray, k: int
) -> float:
    """Brute-force exact E[u(maj(S))] by enumerating all C(G,k) subsets."""
    correct = np.asarray(correct, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    G = len(correct)
    indices = list(range(G))
    utils = []
    for sub in itertools.combinations(indices, k):
        u = _subset_majority_utility(list(sub), correct, class_ids)
        utils.append(u)
    return float(np.mean(utils)) if utils else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_comb(a: int, b: int) -> int:
    """comb(a, b) with convention comb(a, b) = 0 when a < b."""
    if a < b or b < 0:
        return 0
    return comb(a, b)


def _grpo_advantages(correct: np.ndarray) -> np.ndarray:
    """A_i = c_i - mean(c) per group.  No std division."""
    # correct is (B, G) float64
    group_means = correct.mean(axis=1, keepdims=True)
    return correct - group_means


# ---------------------------------------------------------------------------
# Phase-2 verifier-aware advantages (PRM-weighted-vote decision-aware credit)
# ---------------------------------------------------------------------------

def _zero_degenerate(adv: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Zero advantages for all-correct / all-wrong groups (grpo parity: no
    correct-vs-wrong contest there -> no learning signal)."""
    nc = (correct > 0.5).sum(axis=1)
    G = correct.shape[1]
    degen = (nc == 0) | (nc == G)
    adv[degen] = 0.0
    return adv


def _prm_weighted_advantages(correct: np.ndarray, w: np.ndarray) -> np.ndarray:
    """D2: reward_i = w_i * correct_i ; A_i = reward - group_mean(reward).

    Per-sample correct-side reweighting by the deployed PRM weight. NOTE: under
    last-agg this is ~cosine 0.95 to grpo control (PRM|correct piles at 1.0);
    kept for ablation. INVARIANT: reward_i == 0 whenever correct_i == 0.
    """
    reward = w * correct
    adv = reward - reward.mean(axis=1, keepdims=True)
    return _zero_degenerate(adv, correct)


def _prm_weighted_pm_advantages(correct: np.ndarray, w: np.ndarray) -> np.ndarray:
    """D2pm: reward_i = w_i * (2*correct_i - 1) ; A_i = reward - group_mean.

    Symmetric: rewards correct by +w, penalizes wrong by -w (adds wrong-side
    pressure that D2 lacks). Dense.
    """
    reward = w * (2.0 * correct - 1.0)
    adv = reward - reward.mean(axis=1, keepdims=True)
    return _zero_degenerate(adv, correct)


def _prm_margin_advantages(
    correct: np.ndarray, class_ids: np.ndarray, w: np.ndarray,
    empty_mask: np.ndarray | None = None,
) -> np.ndarray:
    """D3 (primary): dense, vote-aware credit for the PRM-weighted-vote decision.

    The deployed rule selects argmax over class mass mass_j = sum_{i in j} w_i,
    DROPPING empty/unparseable-canonical completions (drop_empty=True). For each
    group let C* = the CORRECT class with the largest mass and W* = the WRONG class
    with the largest mass (the two contenders whose mass ordering decides
    correctness), considering only non-empty classes. Credit:
        +w_i  to the genuinely-CORRECT members of C*  (raise the top correct mass)
        -w_i  to members of W*   (lower the confidently-wrong plurality's mass)
         0    to all other samples
    then subtract the group mean (variance reduction; ~zero-mean like the other
    arms -> untouched samples carry the small -mean baseline shift, not exactly 0).
    Zeroed on all-correct / all-wrong groups (no contest).

    Two faithfulness rules (code review):
      * credit +w only to CORRECT members of C*: a class is flagged correct if ANY
        member is correct, but a truncated/graded-wrong completion can share a
        correct canonical (train_grpo keeps its canonical while forcing is_corr=0);
        crediting it would reward truncation. Mass that SELECTS C*/W* still sums all
        members (matches the deployed vote); only the per-sample CREDIT is gated.
      * drop the empty-canonical class from C*/W* selection (empty_mask), mirroring
        the deployed drop_empty=True so the surrogate targets the same contest.

    Unlike D2 it pressures the WRONG plurality directly (the diagnosed bottleneck),
    and unlike D1 (exact LOO decision-marginal) it is dense (credits whole classes).
    """
    B, G = correct.shape
    adv = np.zeros((B, G), dtype=np.float64)
    if empty_mask is None:
        empty_mask = np.zeros((B, G), dtype=bool)
    for b in range(B):
        c = correct[b]
        nc = int((c > 0.5).sum())
        if nc == 0 or nc == G:
            continue  # no correct-vs-wrong contest -> no signal (grpo parity)
        cls = class_ids[b]
        wb = w[b]
        emp = empty_mask[b]
        mass: dict[int, float] = {}
        cls_is_correct: dict[int, bool] = {}
        cls_is_empty: dict[int, bool] = {}
        for i in range(G):
            j = int(cls[i])
            mass[j] = mass.get(j, 0.0) + float(wb[i])  # deployed mass = all members
            if c[i] > 0.5:
                cls_is_correct[j] = True
            if emp[i]:
                cls_is_empty[j] = True
        # candidate classes exclude the dropped (empty-canonical) class.
        correct_classes = [j for j in mass if cls_is_correct.get(j, False) and not cls_is_empty.get(j, False)]
        wrong_classes = [j for j in mass if not cls_is_correct.get(j, False) and not cls_is_empty.get(j, False)]
        # top-mass class on each side; deterministic tie-break: larger mass, then
        # smaller class id.
        Cstar = max(correct_classes, key=lambda j: (mass[j], -j)) if correct_classes else None
        Wstar = max(wrong_classes, key=lambda j: (mass[j], -j)) if wrong_classes else None
        base = np.zeros(G, dtype=np.float64)
        for i in range(G):
            j = int(cls[i])
            if j == Cstar and c[i] > 0.5:    # credit ONLY genuinely-correct members of C*
                base[i] = wb[i]
            elif j == Wstar:
                base[i] = -wb[i]
        adv[b] = base - base.mean()
    return adv


def _passk_advantages(correct: np.ndarray, k: int) -> np.ndarray:
    """Closed-form LOO pass@k advantages.

    For a group with C correct out of G:
        A_corr(C)  = (k / (G-k)) * C(G-C, k) / C(G, k)
        A_wrong(C) = C(G-1-C, k) / C(G-1, k) - C(G-C, k) / C(G, k)
    """
    B, G = correct.shape
    advantages = np.zeros((B, G), dtype=np.float64)

    for b in range(B):
        C = int(correct[b].sum())
        denom_G = _safe_comb(G, k)
        if denom_G == 0:
            continue

        # Precompute shared term
        term_GC = _safe_comb(G - C, k) / denom_G

        denom_G1 = _safe_comb(G - 1, k)
        if denom_G1 == 0:
            term_G1C = 0.0
        else:
            term_G1C = _safe_comb(G - 1 - C, k) / denom_G1

        A_corr = (k / (G - k)) * term_GC
        A_wrong = term_G1C - term_GC

        for g in range(G):
            if correct[b, g] > 0.5:
                advantages[b, g] = A_corr
            else:
                advantages[b, g] = A_wrong

    return advantages


# ---------------------------------------------------------------------------
# vote_k exact DP
# ---------------------------------------------------------------------------

class _ClassInfo:
    """Parsed per-group class information for vote_k DP."""
    __slots__ = ("unique_ids", "counts", "is_correct", "n_classes")

    def __init__(
        self,
        unique_ids: list[int],
        counts: list[int],
        is_correct: list[bool],
    ):
        self.unique_ids = unique_ids
        self.counts = counts
        self.is_correct = is_correct  # per-class boolean: True iff that class is graded correct
        self.n_classes = len(unique_ids)


def _extract_class_info(correct: np.ndarray, class_ids: np.ndarray) -> _ClassInfo:
    """Extract class structure from a single group.

    Each distinct canonical class_id forms its own voting bloc.  A class is
    marked correct iff at least one of its members is graded correct.
    Multiple distinct correct classes are kept separate so that the DP
    matches the deployed eval rule (each canonical form votes separately;
    utility = probability a uniformly tie-broken plurality winner is graded
    correct).
    """
    G = len(correct)
    correct_set = set(int(class_ids[g]) for g in range(G) if correct[g] > 0.5)

    # Count per class (no merging)
    unique_ids_arr, counts_arr = np.unique(class_ids, return_counts=True)
    unique_ids = list(unique_ids_arr.astype(int))
    counts = list(counts_arr.astype(int))

    # Per-class correctness flag: True iff the class id is in correct_set
    is_correct = [uid in correct_set for uid in unique_ids]

    return _ClassInfo(unique_ids, counts, is_correct)


def _majority_utility(count_vec: tuple[int, ...], is_correct: list[bool]) -> float:
    """Compute utility u(maj) for a subset with given class count vector.

    Tie-break: u = |argmax_set ∩ correct_classes| / |argmax_set|.

    This matches the deployed eval rule: each canonical form votes separately,
    and utility is the probability that a uniformly tie-broken plurality winner
    is graded correct.  Reduces to u = 1[correct ∈ argmax] / |argmax| in the
    single-correct-class case.
    """
    if not any(is_correct):
        return 0.0
    max_count = max(count_vec)
    if max_count == 0:
        return 0.0
    argmax_set = [i for i, c in enumerate(count_vec) if c == max_count]
    correct_in_argmax = sum(1 for i in argmax_set if is_correct[i])
    if correct_in_argmax == 0:
        return 0.0
    return correct_in_argmax / len(argmax_set)


def _maj_at_k_exact_from_info(class_info: _ClassInfo, G: int, k: int) -> float:
    """Exact E[u(maj(S))] using multivariate hypergeometric DP.

    Enumerate all count vectors (n_0, n_1, ..., n_{C-1}) where sum = k,
    0 <= n_c <= group_count_c.  Weight by product of binomials / C(G, k).
    """
    n_classes = class_info.n_classes
    counts = class_info.counts
    is_correct = class_info.is_correct
    denom = _safe_comb(G, k)
    if denom == 0:
        return 0.0

    total_util = 0.0
    # Enumerate count vectors via recursive generator
    for count_vec in _enumerate_count_vectors(counts, k):
        # weight = product of C(group_count_c, n_c) for all c
        weight = 1
        for c_idx, n_c in enumerate(count_vec):
            weight *= _safe_comb(counts[c_idx], n_c)
        u = _majority_utility(count_vec, is_correct)
        total_util += weight * u

    return total_util / denom


def _enumerate_count_vectors(counts: list[int], k: int):
    """Generate all count vectors (n_0, ..., n_{C-1}) with sum=k and n_c <= counts[c].

    Uses recursion: fix n_0, then recurse on remaining classes.
    """
    n_classes = len(counts)
    if n_classes == 0:
        if k == 0:
            yield ()
        return
    if n_classes == 1:
        if 0 <= k <= counts[0]:
            yield (k,)
        return

    for n0 in range(min(counts[0], k) + 1):
        for rest in _enumerate_count_vectors(counts[1:], k - n0):
            yield (n0,) + rest


def _votek_advantages(
    correct: np.ndarray, class_ids: np.ndarray, k: int
) -> np.ndarray:
    """Exact vote_k advantages using hypergeometric DP.

    A_i = E_{S∋i,|S|=k}[u(maj(S))] - E_{S∌i,|S|=k}[u(maj(S))]

    For S ∋ i (size k): choose k-1 from the remaining G-1 samples.
    For S ∌ i (size k): choose k from the remaining G-1 samples.

    We enumerate count vectors over the G-1 remaining samples' class counts,
    then add sample i's class contribution to get the full subset count vector.
    """
    B, G = correct.shape
    advantages = np.zeros((B, G), dtype=np.float64)

    for b in range(B):
        class_info = _extract_class_info(correct[b], class_ids[b])
        n_classes = class_info.n_classes
        counts = class_info.counts
        is_correct = class_info.is_correct

        # Map each sample to its class index using raw (unmerged) class_ids
        id_to_class_idx = {uid: idx for idx, uid in enumerate(class_info.unique_ids)}
        sample_class_idx = np.array(
            [id_to_class_idx[int(class_ids[b, g])] for g in range(G)], dtype=np.int64
        )

        for g in range(G):
            ci = int(sample_class_idx[g])  # class index of sample g

            # Counts for the G-1 remaining samples
            remaining_counts = list(counts)
            remaining_counts[ci] -= 1
            assert remaining_counts[ci] >= 0

            # E[u | S ∋ i, |S|=k]: choose k-1 from remaining G-1
            # Full count vector = (choose from remaining) + (1 at position ci)
            denom_with = _safe_comb(G - 1, k - 1)
            with_util = 0.0
            if denom_with > 0:
                for cvec in _enumerate_count_vectors(remaining_counts, k - 1):
                    weight = 1
                    for c_idx, n_c in enumerate(cvec):
                        weight *= _safe_comb(remaining_counts[c_idx], n_c)
                    # Add sample g's class contribution
                    full_cvec = list(cvec)
                    full_cvec[ci] += 1
                    u = _majority_utility(tuple(full_cvec), is_correct)
                    with_util += weight * u
                with_util /= denom_with

            # E[u | S ∌ i, |S|=k]: choose k from remaining G-1
            denom_without = _safe_comb(G - 1, k)
            without_util = 0.0
            if denom_without > 0:
                for cvec in _enumerate_count_vectors(remaining_counts, k):
                    weight = 1
                    for c_idx, n_c in enumerate(cvec):
                        weight *= _safe_comb(remaining_counts[c_idx], n_c)
                    u = _majority_utility(cvec, is_correct)
                    without_util += weight * u
                without_util /= denom_without

            advantages[b, g] = with_util - without_util

    return advantages


def _subset_majority_utility(
    subset: list[int], correct: np.ndarray, class_ids: np.ndarray
) -> float:
    """Compute u(maj(S)) for a given subset of indices.

    Used by brute-force reference implementations.
    """
    if not subset:
        return 0.0

    # Check if any correct in subset
    correct_set = set(
        int(class_ids[g]) for g in range(len(correct)) if correct[g] > 0.5
    )

    # Count votes per class id
    votes: dict[int, int] = {}
    for g in subset:
        cid = int(class_ids[g])
        votes[cid] = votes.get(cid, 0) + 1

    if not votes:
        return 0.0

    max_votes = max(votes.values())
    argmax_set = [cid for cid, v in votes.items() if v == max_votes]

    # Correct if any correct canonical in argmax set
    correct_in_argmax = [cid for cid in argmax_set if cid in correct_set]
    if correct_in_argmax:
        return len(correct_in_argmax) / len(argmax_set)
    return 0.0
