"""
phase0_lib.py — Selection-gap analysis on cached DAFT eval rollouts.

Phase 0 of the Round-2 reframe (bottleneck = SELECTION, not coverage).
Pure offline analysis over rl/eval_outputs/*/T1.0_n64/*/completions.jsonl.

Source of truth = completions.jsonl (per problem: 64 completions + 64 canonical
strings + 64 correctness flags + gt answer). Everything is recomputed from raw
so we do not depend on the inconsistent per-shard metrics.json.

Key fact that makes the gap autopsy label-cheap: correctness is per-sample
math_equal(gt, canon), so ALL correct canonical classes are mutually equivalent
(each == gt). Merging the correct classes is therefore the exact effect a perfect
canonicalizer would have on the correct side — computable from correct_mask alone.

Deps: numpy + rl.advantages (numpy-only). No sal / no GPU.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rl.advantages import maj_at_k_estimate, pass_at_n_unbiased

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
EVAL = os.path.join(REPO, "rl", "eval_outputs")

RUNS = {
    "pi0":   "qwen_1.5b_s1_vanilla_1g_20260611_024222",
    "grpo":  "arm_a_grpo_final",
    "passk": "arm_b_passk_final",
    "votek": "arm_c_votek_final",
}
DATASETS = ["math500", "math_test_1500"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Problem:
    problem_id: str
    dataset: str
    answer: str
    canonicals: list[str]          # len n (==64)
    correct: np.ndarray            # (n,) int {0,1}
    lengths: np.ndarray            # (n,) int char-lengths of completions
    n: int


def _load_run(subdir: str) -> dict[str, Problem]:
    out: dict[str, Problem] = {}
    for ds in DATASETS:
        pat = os.path.join(EVAL, subdir, ds, "T1.0_n64", "shard*", "completions.jsonl")
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    pid = r["problem_id"]
                    canon = list(r["canonicals"])
                    corr = np.asarray(r["correct_mask"], dtype=np.int64)
                    lens = np.asarray([len(c) for c in r["completions"]], dtype=np.int64)
                    out[pid] = Problem(pid, ds, r["answer"], canon, corr, lens, len(corr))
    return out


def load_paired() -> dict[str, dict[str, Problem]]:
    """Return {arm: {problem_id: Problem}} restricted to problem_ids present in
    ALL runs (paired pool)."""
    runs = {name: _load_run(sub) for name, sub in RUNS.items()}
    common = set.intersection(*(set(d.keys()) for d in runs.values()))
    paired = {name: {pid: d[pid] for pid in common} for name, d in runs.items()}
    return paired


# --------------------------------------------------------------------------
# Class structure
# --------------------------------------------------------------------------
def class_ids(canonicals: list[str]) -> np.ndarray:
    """Integer class ids by exact canonical-string equality (== eval rule,
    rl.rewards.class_ids_from_canonicals). '' (unparseable) is one class."""
    idmap: dict[str, int] = {}
    ids = np.empty(len(canonicals), dtype=np.int64)
    for i, c in enumerate(canonicals):
        if c not in idmap:
            idmap[c] = len(idmap)
        ids[i] = idmap[c]
    return ids


def class_counts(p: Problem):
    """Return (correct_counts, wrong_counts, empty_count) over exact-canonical
    classes. correct_counts/wrong_counts are lists of per-class sample counts."""
    correct_classes = {p.canonicals[i] for i in range(p.n) if p.correct[i] > 0.5}
    by_canon: dict[str, int] = {}
    for c in p.canonicals:
        by_canon[c] = by_canon.get(c, 0) + 1
    correct_counts, wrong_counts = [], []
    empty_count = 0
    for canon, cnt in by_canon.items():
        if canon == "":
            empty_count = cnt
        if canon in correct_classes:
            correct_counts.append(cnt)
        else:
            wrong_counts.append(cnt)
    return correct_counts, wrong_counts, empty_count


# --------------------------------------------------------------------------
# Coverage / vote metrics (per problem)
# --------------------------------------------------------------------------
def pass_at_k(p: Problem, k: int) -> float:
    return pass_at_n_unbiased(int(p.correct.sum()), p.n, k)


def maj_at_k(p: Problem, k: int, seed: int = 0) -> float:
    return maj_at_k_estimate(p.correct, class_ids(p.canonicals), k, seed=seed)


def _plurality_utility(counts_by_class: dict, correct_classes: set) -> float:
    """utility = (# correct classes in argmax) / |argmax|."""
    if not counts_by_class:
        return 0.0
    top = max(counts_by_class.values())
    argmax = [c for c, n in counts_by_class.items() if n == top]
    cin = sum(1 for c in argmax if c in correct_classes)
    return cin / len(argmax)


# --------------------------------------------------------------------------
# Selectors (single decision over all n=64 samples -> utility in [0,1])
# --------------------------------------------------------------------------
def sel_oracle(p: Problem) -> float:
    """Coverage ceiling: any correct sample exists."""
    return 1.0 if p.correct.sum() >= 1 else 0.0


def sel_plain_maj(p: Problem) -> float:
    """Deployed rule: plurality over exact-canonical classes (full budget)."""
    correct_classes = {p.canonicals[i] for i in range(p.n) if p.correct[i] > 0.5}
    counts: dict[str, int] = {}
    for c in p.canonicals:
        counts[c] = counts.get(c, 0) + 1
    return _plurality_utility(counts, correct_classes)


def sel_drop_empty(p: Problem) -> float:
    """PRM-free: same plurality but unparseable ('') answers do not vote."""
    correct_classes = {p.canonicals[i] for i in range(p.n) if p.correct[i] > 0.5}
    counts: dict[str, int] = {}
    for c in p.canonicals:
        if c == "":
            continue
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 0.0
    return _plurality_utility(counts, correct_classes)


def sel_tiebreak_short(p: Problem) -> float:
    """PRM-free: plurality; break ties toward the class with the shortest mean
    completion length (a confidence-free heuristic). Resolves category-(a) ties
    deterministically instead of crediting 1/|argmax|."""
    correct_classes = {p.canonicals[i] for i in range(p.n) if p.correct[i] > 0.5}
    counts: dict[str, int] = {}
    lensum: dict[str, int] = {}
    for c, L in zip(p.canonicals, p.lengths):
        counts[c] = counts.get(c, 0) + 1
        lensum[c] = lensum.get(c, 0) + int(L)
    if not counts:
        return 0.0
    top = max(counts.values())
    argmax = [c for c, n in counts.items() if n == top]
    # pick min mean length
    winner = min(argmax, key=lambda c: lensum[c] / counts[c])
    return 1.0 if winner in correct_classes else 0.0


def sel_merged_correct_ceiling(p: Problem) -> float:
    """ORACLE/CEILING (uses labels): merge all correct canonical classes (they
    are all math_equal to gt) into one bloc, vote vs unmerged wrong classes.
    Upper bound on what a perfect *correct-side* canonicalizer can recover.
    Wrong-side forms are NOT merged, so this is an upper bound."""
    cc, wc, _ = class_counts(p)
    if not cc:
        return 0.0
    merged = sum(cc)
    best_wrong = max(wc) if wc else 0
    if merged > best_wrong:
        return 1.0
    if merged == best_wrong:
        return 0.5  # tie between merged-correct bloc and a wrong bloc
    return 0.0


SELECTORS = {
    "oracle@64 (coverage ceiling)": sel_oracle,
    "plain maj (deployed)": sel_plain_maj,
    "maj, drop-unparseable": sel_drop_empty,
    "maj, tiebreak-shortest": sel_tiebreak_short,
    "merged-correct ceiling*": sel_merged_correct_ceiling,
}


# --------------------------------------------------------------------------
# Gap autopsy
# --------------------------------------------------------------------------
@dataclass
class Autopsy:
    n_total: int
    n_dead: int            # pass@64 == 0 (correct answer never generated)
    n_solvable: int        # >=1 correct
    n_won: int             # best_correct > best_wrong (deployed wins outright)
    n_misselected: int     # solvable but not won outright
    n_frag: int            # (c) merged-correct beats best wrong  -> canonicalization
    n_tie: int             # (a) merged-correct ties best wrong    -> tie-break
    n_minority: int        # (b) wrong genuinely more popular      -> needs selector/policy


def autopsy(probs: Iterable[Problem]) -> Autopsy:
    n_total = n_dead = n_solvable = n_won = n_mis = n_frag = n_tie = n_min = 0
    for p in probs:
        n_total += 1
        C = int(p.correct.sum())
        if C == 0:
            n_dead += 1
            continue
        n_solvable += 1
        cc, wc, _ = class_counts(p)
        best_correct = max(cc) if cc else 0
        best_wrong = max(wc) if wc else 0
        merged = sum(cc)  # == C
        if best_correct > best_wrong:
            n_won += 1
            continue
        # mis-selected (deployed does not win outright: tie or loss)
        n_mis += 1
        if merged > best_wrong:
            n_frag += 1
        elif merged == best_wrong:
            n_tie += 1
        else:
            n_min += 1
    return Autopsy(n_total, n_dead, n_solvable, n_won, n_mis, n_frag, n_tie, n_min)


# --------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------
def selector_accuracy(probs: list[Problem]) -> dict[str, float]:
    out = {}
    for name, fn in SELECTORS.items():
        out[name] = float(np.mean([fn(p) for p in probs]))
    return out


def passk_curve(probs: list[Problem], ks=(1, 2, 4, 8, 16, 32, 64)) -> dict[int, float]:
    return {k: float(np.mean([pass_at_k(p, k) for p in probs])) for k in ks}


def majk_curve(probs: list[Problem], ks=(1, 2, 4, 8, 16, 32)) -> dict[int, float]:
    return {k: float(np.mean([maj_at_k(p, k) for p in probs])) for k in ks}
