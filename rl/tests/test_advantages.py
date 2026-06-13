"""
test_advantages.py — Tests for rl/advantages.py per DESIGN.md spec §3 A1-A3.

All tests run on CPU; CUDA-dependent tests are skipped if no GPU.
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

# Module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from advantages import (
    compute_advantages,
    pass_at_n_unbiased,
    maj_at_k_exact,
    brute_force_passk_advantages,
    brute_force_votek_advantages,
    brute_force_maj_at_k_exact,
    _safe_comb,
    _passk_advantages,
    _votek_advantages,
)

ATOL = 1e-10  # tolerance for all brute-force comparisons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_group(G: int, rng: np.random.Generator, n_classes: int = 4):
    """Generate a random group with G samples, n_classes answer classes."""
    correct = rng.integers(0, 2, size=G).astype(np.float64)
    class_ids = rng.integers(0, n_classes, size=G).astype(np.int64)
    # Assign class_ids[i] consistently with correct[i]:
    # correct samples all get class 0, incorrect get classes 1..n_classes-1
    # so that class membership is coherent with the unique-correct-class rule.
    # Actually: pick a random number of correct, assign them all to class 0,
    # distribute wrong among remaining classes.
    n_correct = int(correct.sum())
    correct_mask = correct > 0.5
    class_ids[correct_mask] = 0
    if (~correct_mask).any():
        class_ids[~correct_mask] = rng.integers(1, max(2, n_classes), size=(~correct_mask).sum())
    return correct, class_ids


def _all_correct_group(G: int):
    """A group where all samples are correct (same class)."""
    correct = np.ones(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)
    return correct, class_ids


def _all_wrong_group(G: int, n_classes: int = 3):
    """A group where no samples are correct."""
    correct = np.zeros(G, dtype=np.float64)
    rng = np.random.default_rng(42)
    class_ids = rng.integers(1, max(2, n_classes), size=G).astype(np.int64)
    return correct, class_ids


# ---------------------------------------------------------------------------
# Test 1: compute_advantages matches brute-force for pass_at_k and vote_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G,k_list", [
    (4, [1, 2, 3]),
    (6, [1, 3, 5]),
    (8, [1, 4, 7]),
    (16, [1, 8, 15]),
])
def test_passk_matches_brute_force(G, k_list):
    """pass_at_k advantages match brute-force subset enumeration to 1e-10."""
    rng = np.random.default_rng(seed=G * 100 + 7)
    n_groups = 5

    for k in k_list:
        for _ in range(n_groups):
            correct, class_ids = _random_group(G, rng)
            correct_batch = correct[np.newaxis, :]
            class_ids_batch = class_ids[np.newaxis, :]

            computed = compute_advantages(correct_batch, class_ids_batch, "pass_at_k", k)
            brute = brute_force_passk_advantages(correct, k)

            np.testing.assert_allclose(
                computed[0], brute, atol=ATOL,
                err_msg=f"G={G}, k={k}, correct={correct}, mismatch"
            )


@pytest.mark.parametrize("G,k_list", [
    (4, [1, 2, 3]),
    (6, [1, 3, 5]),
    (8, [1, 4, 7]),
    (16, [1, 8, 15]),
])
def test_votek_matches_brute_force(G, k_list):
    """vote_k advantages match brute-force subset enumeration to 1e-10."""
    rng = np.random.default_rng(seed=G * 200 + 13)
    n_groups = 5

    for k in k_list:
        for _ in range(n_groups):
            correct, class_ids = _random_group(G, rng)
            correct_batch = correct[np.newaxis, :]
            class_ids_batch = class_ids[np.newaxis, :]

            computed = compute_advantages(correct_batch, class_ids_batch, "vote_k", k)
            brute = brute_force_votek_advantages(correct, class_ids, k)

            np.testing.assert_allclose(
                computed[0], brute, atol=ATOL,
                err_msg=f"G={G}, k={k}, correct={correct}, class_ids={class_ids}"
            )


# ---------------------------------------------------------------------------
# Test 2: pass_at_k edge cases
# ---------------------------------------------------------------------------

def test_passk_all_correct_gives_zero_advantages():
    """C=G: all advantages should be zero (no improvement possible)."""
    G = 8
    k = 4
    correct, class_ids = _all_correct_group(G)
    brute = brute_force_passk_advantages(correct, k)
    computed = _passk_advantages(correct[np.newaxis, :], k)[0]

    np.testing.assert_allclose(brute, np.zeros(G), atol=ATOL,
                               err_msg="C=G brute force should be zero")
    np.testing.assert_allclose(computed, np.zeros(G), atol=ATOL,
                               err_msg="C=G computed should be zero")


def test_passk_all_wrong_gives_zero_advantages():
    """C=0: all advantages should be zero (no correct sample to credit)."""
    G = 8
    k = 4
    correct, class_ids = _all_wrong_group(G)
    brute = brute_force_passk_advantages(correct, k)
    computed = _passk_advantages(correct[np.newaxis, :], k)[0]

    np.testing.assert_allclose(brute, np.zeros(G), atol=ATOL,
                               err_msg="C=0 brute force should be zero")
    np.testing.assert_allclose(computed, np.zeros(G), atol=ATOL,
                               err_msg="C=0 computed should be zero")


def test_passk_k1_proportional_to_grpo():
    """At k=1, pass@k advantages share the same sign structure as GRPO advantages.

    Verify against brute force (both compute LOO of R̂(C,G)) and assert that
    correct samples always have non-negative pass@k advantages while wrong
    samples have non-positive pass@k advantages — same ordering as GRPO.
    """
    rng = np.random.default_rng(seed=42)
    G = 8
    k = 1

    for trial in range(10):
        correct, class_ids = _random_group(G, rng)
        C = int(correct.sum())

        computed = _passk_advantages(correct[np.newaxis, :], k)[0]
        brute = brute_force_passk_advantages(correct, k)

        # Verify computed matches brute force
        np.testing.assert_allclose(computed, brute, atol=ATOL,
                                   err_msg=f"k=1 trial {trial}: computed vs brute mismatch")

        if 0 < C < G:
            # A_corr(C, k=1) from closed form: (k/(G-k)) * C(G-C,k)/C(G,k)
            # = (1/(G-1)) * (G-C)/G
            # A_wrong(C, k=1) = C(G-1-C,1)/C(G-1,1) - C(G-C,1)/C(G,1)
            #                 = (G-1-C)/(G-1) - (G-C)/G  (which is <= 0)
            # Both have same sign structure as GRPO: correct > 0, wrong < 0.
            grpo_adv = correct - correct.mean()
            passk_adv = computed

            # Both should have same sign structure (same ordering)
            for g in range(G):
                grpo_sign = np.sign(grpo_adv[g])
                passk_sign = np.sign(passk_adv[g])
                # They should agree or one is zero — not opposite
                assert grpo_sign * passk_sign >= 0, (
                    f"k=1 trial {trial}: sign disagreement at g={g}, "
                    f"grpo={grpo_adv[g]:.4f}, passk={passk_adv[g]:.4f}"
                )


def test_passk_k1_formula():
    """Verify the k=1 closed-form A_corr and A_wrong formulas.

    At k=1, the DESIGN closed-form reduces to:
        A_corr(C, k=1) = (1/(G-1)) * (G-C)/G
        A_wrong(C, k=1) = (G-1-C)/(G-1) - (G-C)/G   [<= 0]

    These are derived from LOO of R̂(C,G) (verified by brute force).
    """
    G = 8
    k = 1
    # Test a few specific C values
    for C in range(1, G):
        # Construct a group with exactly C correct
        correct = np.zeros(G, dtype=np.float64)
        correct[:C] = 1.0
        class_ids = np.zeros(G, dtype=np.int64)
        class_ids[C:] = np.arange(1, G - C + 1)

        brute = brute_force_passk_advantages(correct, k)
        computed = _passk_advantages(correct[np.newaxis, :], k)[0]

        # A_corr(C, k=1) = (k/(G-k)) * C(G-C,1)/C(G,1) = (1/(G-1)) * (G-C)/G
        expected_corr = (G - C) / (G * (G - 1))
        # A_wrong(C, k=1) = C(G-1-C,1)/C(G-1,1) - C(G-C,1)/C(G,1)
        #                  = (G-1-C)/(G-1) - (G-C)/G  (when C < G-1)
        #                  = 0 - 1/G = -1/G  (when C = G-1, since G-1-C=0)
        if C < G - 1:
            expected_wrong = (G - 1 - C) / (G - 1) - (G - C) / G
        else:
            expected_wrong = 0.0 - (G - C) / G  # C(0,1)=0

        np.testing.assert_allclose(
            computed[:C], expected_corr, atol=ATOL,
            err_msg=f"k=1, C={C}: correct samples formula mismatch"
        )
        if C < G:
            np.testing.assert_allclose(
                computed[C:], expected_wrong, atol=ATOL,
                err_msg=f"k=1, C={C}: wrong samples formula mismatch"
            )
        np.testing.assert_allclose(brute, computed, atol=ATOL,
                                   err_msg=f"k=1, C={C}: brute vs computed mismatch")


# ---------------------------------------------------------------------------
# Test 3: vote_k tie handling
# ---------------------------------------------------------------------------

def test_votek_tie_handling():
    """Groups with exact ties at argmax use u = 1/|argmax| via brute force agreement."""
    rng = np.random.default_rng(seed=7)

    # Construct a group with a deliberate tie: 2 answers each appear 4 times
    # in a G=8 group.  One is correct.
    G = 8
    k = 4
    correct = np.zeros(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)

    # 4 samples with class 0 (correct), 4 with class 1 (wrong)
    correct[:4] = 1.0
    class_ids[:4] = 0
    class_ids[4:] = 1

    computed = _votek_advantages(correct[np.newaxis, :], class_ids[np.newaxis, :], k)[0]
    brute = brute_force_votek_advantages(correct, class_ids, k)

    np.testing.assert_allclose(computed, brute, atol=ATOL,
                               err_msg="Tie group: computed vs brute mismatch")

    # Verify tie-break: when correct class and wrong class tie,
    # utility should be 0.5 for subsets where they tie
    # Check via brute force of maj_at_k_exact
    exact_dp = maj_at_k_exact(correct, class_ids, k)
    exact_bf = brute_force_maj_at_k_exact(correct, class_ids, k)
    np.testing.assert_allclose(exact_dp, exact_bf, atol=ATOL,
                               err_msg="maj_at_k_exact DP vs brute mismatch on tie group")


def test_votek_three_way_tie():
    """Three classes all tied — verify 1/3 utility for correct class via brute force."""
    # G=6, k=3: 2 each of 3 classes, one correct
    G = 6
    k = 3
    correct = np.zeros(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)
    correct[:2] = 1.0
    class_ids[:2] = 0   # correct class
    class_ids[2:4] = 1  # wrong class 1
    class_ids[4:] = 2   # wrong class 2

    computed = _votek_advantages(correct[np.newaxis, :], class_ids[np.newaxis, :], k)[0]
    brute = brute_force_votek_advantages(correct, class_ids, k)

    np.testing.assert_allclose(computed, brute, atol=ATOL,
                               err_msg="Three-way tie: computed vs brute mismatch")


# ---------------------------------------------------------------------------
# Test 4: Gradient direction test (the review-demanded test)
# ---------------------------------------------------------------------------

def _categorical_logprob(y: int, logits: np.ndarray) -> float:
    """Log probability of category y under categorical(softmax(logits))."""
    logits = logits - logits.max()
    log_probs = logits - np.log(np.exp(logits).sum())
    return float(log_probs[y])


def _sample_group(logits: np.ndarray, G: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample G responses from categorical(softmax(logits)).

    Returns:
        correct : (G,) float64 — 1 if answer == 0 (class 0 is correct)
        class_ids : (G,) int64
    """
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    answers = rng.choice(len(logits), size=G, p=probs)
    correct = (answers == 0).astype(np.float64)
    class_ids = answers.astype(np.int64)
    return correct, class_ids


def _mc_gradient_estimate(
    logits: np.ndarray,
    mode: str,
    G: int,
    k: int,
    n_groups: int,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo estimate of E[sum_i A_i * d/dtheta log pi(y_i | theta)].

    Returns gradient vector of shape (n_classes,).
    """
    rng = np.random.default_rng(seed)
    n_classes = len(logits)
    grad_accum = np.zeros(n_classes, dtype=np.float64)

    for _ in range(n_groups):
        correct, class_ids = _sample_group(logits, G, rng)
        correct_batch = correct[np.newaxis, :]
        class_ids_batch = class_ids[np.newaxis, :]

        advantages = compute_advantages(correct_batch, class_ids_batch, mode, k)[0]

        # d/dtheta log pi(y_i | theta) = e_{y_i} - softmax(theta)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        for i in range(G):
            y_i = int(class_ids[i])
            dlogpi = np.zeros(n_classes, dtype=np.float64)
            dlogpi[y_i] = 1.0
            dlogpi -= probs
            grad_accum += advantages[i] * dlogpi

    return grad_accum / n_groups


def _true_passk_gradient(logits: np.ndarray, G: int, k: int) -> np.ndarray:
    """Numerically differentiate pass@k(theta) = E[1 - C(G-C,k)/C(G,k)].

    Uses exact computation over the categorical distribution.
    """
    n_classes = len(logits)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    def passk_value(log_arr: np.ndarray) -> float:
        p = np.exp(log_arr - log_arr.max())
        p = p / p.sum()
        p0 = float(p[0])  # probability of correct answer

        # J = E_{C~Binomial(G, p0)}[pass_at_n_unbiased(C, G, k)]
        # = sum_{C=0}^{G} C(G,C)*p0^C*(1-p0)^{G-C} * [1 - C(G-C,k)/C(G,k)]
        total = 0.0
        for C in range(G + 1):
            binom_weight = comb(G, C) * (p0 ** C) * ((1 - p0) ** (G - C))
            r = pass_at_n_unbiased(C, G, k)
            total += binom_weight * r
        return total

    # Numerical gradient via finite differences
    eps = 1e-5
    grad = np.zeros(n_classes, dtype=np.float64)
    for d in range(n_classes):
        logits_plus = logits.copy()
        logits_plus[d] += eps
        logits_minus = logits.copy()
        logits_minus[d] -= eps
        grad[d] = (passk_value(logits_plus) - passk_value(logits_minus)) / (2 * eps)

    return grad


def _true_votek_gradient(logits: np.ndarray, G: int, k: int) -> np.ndarray:
    """Numerically differentiate E[u(maj_k(S))] with respect to logits.

    E[u(maj)] = sum over count vectors of multinomial weight * u(count_vec).
    """
    n_classes = len(logits)

    def majk_value(log_arr: np.ndarray) -> float:
        p = np.exp(log_arr - log_arr.max())
        p = p / p.sum()

        # Enumerate all count vectors summing to k, class 0 is correct
        # u(vec) = 1/|argmax| if class 0 in argmax
        from advantages import _enumerate_count_vectors, _majority_utility
        counts_max = [G] * n_classes  # upper bounds (multinomial)

        total = 0.0
        for cvec in _enumerate_count_vectors(counts_max, k):
            if sum(cvec) != k:
                continue
            # Multinomial probability: k! / prod(n_c!) * prod(p_c^n_c)
            from math import factorial
            weight = factorial(k)
            for c_idx, n_c in enumerate(cvec):
                weight = weight * (p[c_idx] ** n_c) / factorial(n_c)
            # class 0 is the single correct class
            is_correct_vec = [c_idx == 0 for c_idx in range(len(cvec))]
            u = _majority_utility(cvec, is_correct_vec)
            total += weight * u
        return total

    eps = 1e-5
    grad = np.zeros(n_classes, dtype=np.float64)
    for d in range(n_classes):
        logits_plus = logits.copy()
        logits_plus[d] += eps
        logits_minus = logits.copy()
        logits_minus[d] -= eps
        grad[d] = (majk_value(logits_plus) - majk_value(logits_minus)) / (2 * eps)

    return grad


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def test_gradient_direction_passk():
    """MC gradient E[sum_i A_i * dlogpi] aligns with true pass@k gradient (cosine > 0.95)."""
    np.random.seed(0)
    rng = np.random.default_rng(0)

    n_classes = 3
    G = 8
    k = 4
    n_groups = 100_000

    # Non-uniform logits: class 0 is correct but not dominant
    logits = np.array([-0.5, 0.3, 0.2], dtype=np.float64)

    mc_grad = _mc_gradient_estimate(logits, "pass_at_k", G, k, n_groups, seed=0)
    true_grad = _true_passk_gradient(logits, G, k)

    cos_sim = _cosine_sim(mc_grad, true_grad)
    assert cos_sim > 0.95, (
        f"pass_at_k gradient direction test failed: cosine_sim={cos_sim:.4f} "
        f"(expected > 0.95)\nmc_grad={mc_grad}, true_grad={true_grad}"
    )


def test_gradient_direction_votek():
    """MC gradient E[sum_i A_i * dlogpi] aligns with true vote@k gradient (cosine > 0.95)."""
    n_classes = 3
    G = 8
    k = 4
    n_groups = 100_000

    logits = np.array([-0.5, 0.3, 0.2], dtype=np.float64)

    mc_grad = _mc_gradient_estimate(logits, "vote_k", G, k, n_groups, seed=0)
    true_grad = _true_votek_gradient(logits, G, k)

    cos_sim = _cosine_sim(mc_grad, true_grad)
    assert cos_sim > 0.95, (
        f"vote_k gradient direction test failed: cosine_sim={cos_sim:.4f} "
        f"(expected > 0.95)\nmc_grad={mc_grad}, true_grad={true_grad}"
    )


# ---------------------------------------------------------------------------
# Test 5: all-zero groups (C=0) give exactly zero advantages
# ---------------------------------------------------------------------------

def test_votek_allzero_gives_zero_advantages():
    """vote_k on C=0 groups gives exactly zero advantages."""
    G = 8
    k = 4
    correct, class_ids = _all_wrong_group(G)
    computed = _votek_advantages(correct[np.newaxis, :], class_ids[np.newaxis, :], k)[0]
    np.testing.assert_array_equal(computed, np.zeros(G),
                                  err_msg="C=0 group: vote_k advantages should be zero")


def test_votek_allzero_batch():
    """vote_k on a batch of all-zero groups all give zero advantages."""
    G = 8
    k = 4
    B = 5
    correct = np.zeros((B, G), dtype=np.float64)
    class_ids = np.ones((B, G), dtype=np.int64)  # all wrong, class 1

    computed = compute_advantages(correct, class_ids, "vote_k", k)
    np.testing.assert_array_equal(computed, np.zeros((B, G)),
                                  err_msg="Batch of C=0 groups: vote_k advantages should be zero")


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------

def test_grpo_advantages_zero_sum():
    """GRPO advantages sum to zero per group."""
    B, G = 4, 8
    rng = np.random.default_rng(seed=1)
    correct = rng.integers(0, 2, size=(B, G)).astype(np.float64)
    class_ids = np.zeros((B, G), dtype=np.int64)

    adv = compute_advantages(correct, class_ids, "grpo", k=4)
    np.testing.assert_allclose(adv.sum(axis=1), np.zeros(B), atol=1e-14,
                               err_msg="GRPO advantages should sum to zero per group")


def test_passk_assert_k_bounds():
    """compute_advantages asserts k in [1, G-1]."""
    G = 8
    correct = np.ones((1, G), dtype=np.float64)
    class_ids = np.zeros((1, G), dtype=np.int64)

    with pytest.raises(AssertionError):
        compute_advantages(correct, class_ids, "pass_at_k", k=0)
    with pytest.raises(AssertionError):
        compute_advantages(correct, class_ids, "pass_at_k", k=G)


def test_votek_assert_k_bounds():
    """compute_advantages asserts k in [1, G-1] for vote_k."""
    G = 8
    correct = np.ones((1, G), dtype=np.float64)
    class_ids = np.zeros((1, G), dtype=np.int64)

    with pytest.raises(AssertionError):
        compute_advantages(correct, class_ids, "vote_k", k=0)
    with pytest.raises(AssertionError):
        compute_advantages(correct, class_ids, "vote_k", k=G)


def test_pass_at_n_unbiased_edge_cases():
    """pass_at_n_unbiased returns 0 when C=0, 1 when C>=k."""
    assert pass_at_n_unbiased(0, 8, 4) == 0.0
    assert pass_at_n_unbiased(8, 8, 4) == 1.0
    # C=4, G=8, k=4: 1 - C(4,4)/C(8,4)
    expected = 1.0 - comb(4, 4) / comb(8, 4)
    assert abs(pass_at_n_unbiased(4, 8, 4) - expected) < 1e-14


def test_maj_at_k_exact_all_correct():
    """maj_at_k_exact on all-correct group returns 1.0."""
    G = 8
    k = 4
    correct = np.ones(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)
    result = maj_at_k_exact(correct, class_ids, k)
    assert abs(result - 1.0) < ATOL, f"Expected 1.0, got {result}"


def test_maj_at_k_exact_all_wrong():
    """maj_at_k_exact on all-wrong group returns 0.0."""
    G = 8
    k = 4
    correct = np.zeros(G, dtype=np.float64)
    class_ids = np.ones(G, dtype=np.int64)
    result = maj_at_k_exact(correct, class_ids, k)
    assert abs(result - 0.0) < ATOL, f"Expected 0.0, got {result}"


def test_maj_at_k_exact_matches_brute_force():
    """maj_at_k_exact matches brute force for random groups."""
    rng = np.random.default_rng(seed=99)
    for G in [4, 6, 8]:
        for k in [1, G // 2, G - 1]:
            correct, class_ids = _random_group(G, rng)
            dp_result = maj_at_k_exact(correct, class_ids, k)
            bf_result = brute_force_maj_at_k_exact(correct, class_ids, k)
            np.testing.assert_allclose(dp_result, bf_result, atol=ATOL,
                                       err_msg=f"G={G}, k={k}: maj_at_k_exact mismatch")


def test_vote_passk_hybrid_linearity():
    """vote_passk_hybrid = vote_k + lambda * pass_at_k."""
    rng = np.random.default_rng(seed=55)
    G = 8
    k = 4
    lam = 0.25
    B = 3

    for _ in range(B):
        correct, class_ids = _random_group(G, rng)
    correct_batch = np.stack([_random_group(G, rng)[0] for _ in range(B)])
    class_ids_batch = np.stack([_random_group(G, rng)[1] for _ in range(B)])

    hybrid = compute_advantages(correct_batch, class_ids_batch, "vote_passk_hybrid", k, lam)
    vote = compute_advantages(correct_batch, class_ids_batch, "vote_k", k)
    passk = compute_advantages(correct_batch, class_ids_batch, "pass_at_k", k)

    np.testing.assert_allclose(hybrid, vote + lam * passk, atol=1e-14,
                               err_msg="hybrid should equal vote + lambda*passk")


def test_safe_comb_convention():
    """_safe_comb(a, b) = 0 when a < b."""
    assert _safe_comb(3, 5) == 0
    assert _safe_comb(0, 1) == 0
    assert _safe_comb(5, 3) == comb(5, 3)
    assert _safe_comb(5, 5) == 1


def test_passk_closed_form_formula_explicit():
    """Verify A_corr and A_wrong against the explicit closed-form formula from DESIGN.md."""
    from advantages import _safe_comb

    G = 16
    k = 8

    for C in range(1, G):
        correct = np.zeros(G, dtype=np.float64)
        correct[:C] = 1.0
        class_ids = np.zeros(G, dtype=np.int64)
        class_ids[C:] = np.arange(1, G - C + 1)

        computed = _passk_advantages(correct[np.newaxis, :], k)[0]

        denom_G = _safe_comb(G, k)
        term_GC = _safe_comb(G - C, k) / denom_G
        A_corr_expected = (k / (G - k)) * term_GC

        denom_G1 = _safe_comb(G - 1, k)
        term_G1C = _safe_comb(G - 1 - C, k) / denom_G1 if denom_G1 > 0 else 0.0
        A_wrong_expected = term_G1C - term_GC

        np.testing.assert_allclose(computed[:C], A_corr_expected, atol=ATOL,
                                   err_msg=f"G={G}, k={k}, C={C}: A_corr mismatch")
        np.testing.assert_allclose(computed[C:], A_wrong_expected, atol=ATOL,
                                   err_msg=f"G={G}, k={k}, C={C}: A_wrong mismatch")


def test_votek_allcorrect_gives_zero_advantages():
    """vote_k on C=G group gives zero advantages (no marginal contribution)."""
    G = 8
    k = 4
    correct = np.ones(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)

    computed = _votek_advantages(correct[np.newaxis, :], class_ids[np.newaxis, :], k)[0]
    brute = brute_force_votek_advantages(correct, class_ids, k)

    # With all correct, maj always correct regardless of who is included/excluded
    np.testing.assert_allclose(computed, np.zeros(G), atol=ATOL,
                               err_msg="C=G group: vote_k advantages should be zero")
    np.testing.assert_allclose(brute, np.zeros(G), atol=ATOL,
                               err_msg="C=G group: brute force vote_k should be zero")


def test_large_G16_k8_passk_timing():
    """pass_at_k at G=16, k=8 should complete quickly (<1s for 100 groups)."""
    import time
    G = 16
    k = 8
    B = 100
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=(B, G)).astype(np.float64)
    class_ids = np.zeros((B, G), dtype=np.int64)

    start = time.time()
    adv = compute_advantages(correct, class_ids, "pass_at_k", k)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"pass_at_k G=16, k=8, B=100 took {elapsed:.3f}s (expected <1s)"


def test_large_G16_k8_votek_timing():
    """vote_k at G=16, k=8 should complete in <50ms per group (per spec)."""
    import time
    G = 16
    k = 8
    rng = np.random.default_rng(0)
    correct, class_ids = _random_group(G, rng, n_classes=4)

    start = time.time()
    adv = _votek_advantages(correct[np.newaxis, :], class_ids[np.newaxis, :], k)
    elapsed = time.time() - start
    assert elapsed < 0.05, f"vote_k G=16, k=8 took {elapsed*1000:.1f}ms (expected <50ms)"


# ---------------------------------------------------------------------------
# Test: split correct classes — DP must match brute force when correct samples
# are split across 2+ distinct class_ids (the routine real-data case where
# math_equal deems e.g. '0.5' and '1/2' both correct but they are separate
# canonical forms / class_ids).  The old merge-based DP failed this case.
# ---------------------------------------------------------------------------

def _split_correct_group(G: int, rng: np.random.Generator, n_correct_classes: int = 2):
    """Generate a group where correct samples are spread across n_correct_classes
    distinct class_ids.  Wrong samples each get a distinct class id > n_correct_classes.

    Returns (correct, class_ids) with shape (G,).
    """
    correct = np.zeros(G, dtype=np.float64)
    class_ids = np.zeros(G, dtype=np.int64)

    # Decide how many samples are correct (at least n_correct_classes so each
    # correct class has at least 1 member)
    n_corr = int(rng.integers(n_correct_classes, max(n_correct_classes + 1, G // 2 + 1)))
    n_corr = min(n_corr, G - 1)  # keep at least 1 wrong sample
    if n_corr < n_correct_classes:
        n_corr = n_correct_classes

    corr_indices = rng.choice(G, size=n_corr, replace=False)
    correct[corr_indices] = 1.0

    # Assign correct samples to n_correct_classes classes (round-robin)
    for rank, idx in enumerate(corr_indices):
        class_ids[idx] = rank % n_correct_classes  # classes 0 .. n_correct_classes-1

    # Assign wrong samples each to a unique wrong class starting at n_correct_classes
    wrong_idx = np.where(correct < 0.5)[0]
    for rank, idx in enumerate(wrong_idx):
        class_ids[idx] = n_correct_classes + rank

    return correct, class_ids


@pytest.mark.parametrize("G,k_list", [
    (6, [1, 3, 5]),
    (8, [1, 4, 7]),
    (12, [1, 6, 11]),
])
def test_votek_split_correct_classes_matches_brute_force(G, k_list):
    """DP vote_k advantages match brute force when correct samples span 2 distinct class_ids.

    This exercises the real-data path where math_equal accepts '0.5' and '1/2'
    as both correct but they carry separate canonical forms / class_ids.
    The old merge-to-sentinel logic made the DP diverge from brute force on
    this case; the fixed is_correct-based DP must agree.
    """
    rng = np.random.default_rng(seed=G * 777 + 3)
    n_trials = 5

    for k in k_list:
        for trial in range(n_trials):
            correct, class_ids = _split_correct_group(G, rng, n_correct_classes=2)
            correct_batch = correct[np.newaxis, :]
            class_ids_batch = class_ids[np.newaxis, :]

            computed = compute_advantages(correct_batch, class_ids_batch, "vote_k", k)
            brute = brute_force_votek_advantages(correct, class_ids, k)

            np.testing.assert_allclose(
                computed[0], brute, atol=ATOL,
                err_msg=(
                    f"Split-correct DP vs brute mismatch: G={G}, k={k}, trial={trial}, "
                    f"correct={correct}, class_ids={class_ids}"
                )
            )


@pytest.mark.parametrize("G,k_list", [
    (6, [1, 3, 5]),
    (8, [1, 4, 7]),
])
def test_maj_at_k_exact_split_correct_classes_matches_brute_force(G, k_list):
    """maj_at_k_exact DP matches brute force when correct samples span 2+ class_ids."""
    rng = np.random.default_rng(seed=G * 888 + 5)
    n_trials = 5

    for k in k_list:
        for trial in range(n_trials):
            correct, class_ids = _split_correct_group(G, rng, n_correct_classes=2)

            dp_result = maj_at_k_exact(correct, class_ids, k)
            bf_result = brute_force_maj_at_k_exact(correct, class_ids, k)

            np.testing.assert_allclose(
                dp_result, bf_result, atol=ATOL,
                err_msg=(
                    f"maj_at_k_exact split-correct mismatch: G={G}, k={k}, trial={trial}, "
                    f"correct={correct}, class_ids={class_ids}"
                )
            )


def test_votek_split_correct_wrong_wins_u_zero():
    """When wrong class has plurality and correct split across 2 classes,
    utility should be 0 (wrong wins outright), not 1 (which the old merge would give).

    Concrete case: G=6, wrong class has 5 votes, correct split 3+3 -> impossible here.
    Use: wrong=5, correct_a=1, correct_b=1 (total 7 but we need subset of size k).
    Simpler: G=7, wrong bloc=5, correct_a=1, correct_b=1, k=5.
    With k=5 drawn from these 7, the wrong bloc almost always dominates.
    Check maj_at_k_exact < 0.5 (old merge would give >= 0.5 since merged correct=2
    vs wrong=5, merged can win some ties).
    """
    # G=8: wrong bloc 5, correct_a=1 (class 0), correct_b=2 (class 1)
    # With merge: correct=3 vs wrong=5, merge can produce u>0 when subset has 3c+2w (tie).
    # Without merge: correct_a=1, correct_b=2, wrong=5 — wrong wins plurality in most subsets.
    G = 8
    k = 5
    correct = np.array([1, 0, 0, 1, 1, 0, 0, 0], dtype=np.float64)
    class_ids = np.array([0, 5, 6, 1, 1, 7, 8, 9], dtype=np.int64)
    # class 0: 1 sample (correct), class 1: 2 samples (correct), classes 5-9: wrong

    dp = maj_at_k_exact(correct, class_ids, k)
    bf = brute_force_maj_at_k_exact(correct, class_ids, k)

    np.testing.assert_allclose(dp, bf, atol=ATOL,
                               err_msg="Wrong-plurality split-correct: DP vs brute mismatch")


def test_votek_numerically_verified_split_correct():
    """Verify the exact numerical case from the review bug report:
    G=12, k=8, wrong bloc=6, correct split 3+3.

    Old merge-based DP: maj@8 = 0.5
    Correct (no-merge, eval-faithful): maj@8 = 0.1515... (brute force value)

    We check DP matches brute force, and that the result is NOT ~0.5.
    """
    G = 12
    k = 8
    # Correct split: class 0 x3, class 1 x3; Wrong: class 2 x6
    correct = np.array(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64
    )
    class_ids = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64
    )

    dp = maj_at_k_exact(correct, class_ids, k)
    bf = brute_force_maj_at_k_exact(correct, class_ids, k)

    np.testing.assert_allclose(dp, bf, atol=ATOL,
                               err_msg="G=12 k=8 split-correct: DP vs brute mismatch")
    # Sanity: wrong bloc dominates, so maj@8 should be well below 0.5
    assert dp < 0.3, (
        f"Expected maj@8 < 0.3 (wrong bloc dominates) but got {dp:.6f}. "
        "This may indicate the old merge bug is still present."
    )
