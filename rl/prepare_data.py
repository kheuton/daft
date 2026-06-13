"""
prepare_data.py — MATH train decontamination, held-out eval slice, and
                  rl_train.jsonl construction from filter sweep output.

Usage (two-phase):

Phase 1 (CPU, runs first):
    python prepare_data.py

    Produces:
        rl/data/math_train_decontam.jsonl   -- decontaminated MATH train
        rl/data/math_test_1500.jsonl        -- level-stratified 1500-problem eval slice

Phase 2 (after filter_sweep.py, GPU):
    python prepare_data.py --from-sweep [--sweep-glob rl/data/sweep_shard*.jsonl]

    Produces:
        rl/data/rl_train.jsonl   -- learnability-filtered train set
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RL_DIR = Path(__file__).parent
DATA_DIR = RL_DIR / "data"
EVAL_DIR = RL_DIR.parent / "eval" / "datasets"
MATH500_PATH = EVAL_DIR / "math500.jsonl"

# ---------------------------------------------------------------------------
# Text normalization (used for decontamination matching)
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Whitespace/latex-light normalization: lowercase, collapse whitespace, strip $ signs."""
    s = s.lower()
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_8grams(normalized: str) -> set[tuple[str, ...]]:
    """Return the set of 8-token (word-level) grams from a normalized string."""
    tokens = normalized.split()
    if len(tokens) < 8:
        return set()
    return {tuple(tokens[i : i + 8]) for i in range(len(tokens) - 7)}


def texts_share_8gram(norm_a: str, norm_b: str) -> bool:
    """True if two normalized texts share at least one 8-gram."""
    grams_a = get_8grams(norm_a)
    if not grams_a:
        return False
    grams_b = get_8grams(norm_b)
    return bool(grams_a & grams_b)


# ---------------------------------------------------------------------------
# Boxed-answer extraction (Hendrycks-style)
# ---------------------------------------------------------------------------

def _extract_last_boxed(solution: str) -> str | None:
    """Extract content of the last \\boxed{...} in solution, handling nested braces."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        # Try \boxed with space or other variants
        idx = solution.rfind(r"\boxed {")
        if idx == -1:
            return None
        start = idx + len(r"\boxed {")
    else:
        start = idx + len(r"\boxed{")

    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return solution[start : i - 1]
    return None


def extract_answer_from_solution(solution: str) -> str | None:
    """Extract the ground-truth answer from a MATH-style solution string.

    Tries:
    1. Last \\boxed{...} content (handles nested braces).
    2. Returns None if nothing found.
    """
    return _extract_last_boxed(solution)


# ---------------------------------------------------------------------------
# Phase 1: decontamination + train JSONL
# ---------------------------------------------------------------------------

def load_math500_problems() -> list[str]:
    """Load problem texts from math500.jsonl."""
    problems = []
    with open(MATH500_PATH) as f:
        for line in f:
            row = json.loads(line)
            problems.append(row["problem"])
    return problems


def load_s1k_questions() -> list[str]:
    """Load question texts from simplescaling/s1K HuggingFace dataset."""
    from datasets import load_dataset  # local import so CPU tests can import module

    ds = load_dataset("simplescaling/s1K", split="train")
    return [row["question"] for row in ds]


def build_decontam_index(
    ref_texts: list[str],
) -> tuple[set[str], dict[tuple[str, ...], bool]]:
    """Build exact-normalized set and 8-gram index from reference texts."""
    exact: set[str] = set()
    gram_index: dict[tuple[str, ...], bool] = {}
    for t in ref_texts:
        n = normalize_text(t)
        exact.add(n)
        for gram in get_8grams(n):
            gram_index[gram] = True
    return exact, gram_index


def is_contaminated(
    problem_norm: str,
    exact_set: set[str],
    gram_index: dict[tuple[str, ...], bool],
) -> bool:
    """True if problem_norm exactly matches or shares an 8-gram with reference."""
    if problem_norm in exact_set:
        return True
    for gram in get_8grams(problem_norm):
        if gram in gram_index:
            return True
    return False


def run_phase1() -> None:
    """Load MATH train, decontaminate, write math_train_decontam.jsonl and math_test_1500.jsonl."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DigitalLearningGmbH/MATH-lighteval train split (7500 items)...")
    math_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    print(f"  Loaded {len(math_train)} train items.")

    # -- Decontamination references --
    print("\nLoading MATH-500 problems for decontamination...")
    math500_problems = load_math500_problems()
    print(f"  Loaded {len(math500_problems)} MATH-500 problems.")

    print("Loading simplescaling/s1K questions for decontamination...")
    s1k_questions = load_s1k_questions()
    print(f"  Loaded {len(s1k_questions)} s1K questions.")

    # Build indices
    exact500, gram500 = build_decontam_index(math500_problems)
    exact_s1k, gram_s1k = build_decontam_index(s1k_questions)

    # -- Filter train --
    kept = []
    dropped_math500 = 0
    dropped_s1k = 0
    dropped_both = 0
    dropped_no_answer = 0

    for item in math_train:
        problem = item["problem"]
        solution = item["solution"]
        level = item["level"]
        ptype = item["type"]

        norm = normalize_text(problem)
        hit500 = is_contaminated(norm, exact500, gram500)
        hit_s1k = is_contaminated(norm, exact_s1k, gram_s1k)

        if hit500 and hit_s1k:
            dropped_both += 1
            continue
        elif hit500:
            dropped_math500 += 1
            continue
        elif hit_s1k:
            dropped_s1k += 1
            continue

        # Extract answer
        answer = extract_answer_from_solution(solution)
        if answer is None:
            dropped_no_answer += 1
            continue

        kept.append({
            "problem": problem,
            "answer": answer,
            "level": level,
            "type": ptype,
        })

    total_dropped = dropped_math500 + dropped_s1k + dropped_both
    print(f"\nDecontamination results:")
    print(f"  Original train size:              {len(math_train)}")
    print(f"  Dropped (MATH-500 only):          {dropped_math500}")
    print(f"  Dropped (s1K only):               {dropped_s1k}")
    print(f"  Dropped (both MATH-500 and s1K):  {dropped_both}")
    print(f"  Total decontamination drops:      {total_dropped}")
    print(f"  Dropped (no extractable answer):  {dropped_no_answer}")
    print(f"  Kept for RL training:             {len(kept)}")

    out_train = DATA_DIR / "math_train_decontam.jsonl"
    with open(out_train, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(kept)} rows to {out_train}")

    # -- Build held-out eval slice from MATH-lighteval TEST --
    print("\nLoading DigitalLearningGmbH/MATH-lighteval test split (5000 items)...")
    math_test = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    print(f"  Loaded {len(math_test)} test items.")

    # Remove math500 problems (exact-normalized match)
    n_matched_500 = 0
    test_candidates = []
    for item in math_test:
        norm = normalize_text(item["problem"])
        if norm in exact500:
            n_matched_500 += 1
        else:
            answer = extract_answer_from_solution(item["solution"])
            if answer is not None:
                test_candidates.append({
                    "problem": item["problem"],
                    "answer": answer,
                    "level": item["level"],
                    "type": item["type"],
                })

    print(f"  MATH-500 problems matched and removed from test: {n_matched_500}")
    if n_matched_500 < 400 or n_matched_500 > 600:
        print(f"  WARNING: Expected ~500 matches but got {n_matched_500}; investigate.")
    else:
        print(f"  Match count {n_matched_500} is in expected range [400, 600]. OK.")
    print(f"  Remaining test candidates (with answer): {len(test_candidates)}")

    # Level-stratified sample 1500 with seed 0
    import random
    rng = random.Random(0)

    by_level: dict[str, list[dict]] = defaultdict(list)
    for item in test_candidates:
        by_level[item["level"]].append(item)

    # Compute stratum sizes proportional to counts, total = 1500
    total_candidates = len(test_candidates)
    target = 1500
    assert total_candidates >= target, (
        f"Not enough test candidates ({total_candidates}) to sample {target}."
    )

    # Proportional allocation (Hamilton/largest-remainder method)
    levels = sorted(by_level.keys())
    raw_counts = {lvl: len(by_level[lvl]) * target / total_candidates for lvl in levels}
    floors = {lvl: int(raw_counts[lvl]) for lvl in levels}
    remainder = target - sum(floors.values())
    # Sort by fractional part descending, break ties by level
    fracs = sorted(levels, key=lambda lvl: -(raw_counts[lvl] - floors[lvl]))
    for lvl in fracs[:remainder]:
        floors[lvl] += 1

    sampled = []
    for lvl in levels:
        pool = list(by_level[lvl])
        rng.shuffle(pool)
        n_sample = floors[lvl]
        sampled.extend(pool[:n_sample])
        print(f"  Level {lvl}: {len(pool)} candidates, sampled {n_sample}")

    print(f"  Total sampled: {len(sampled)} (target {target})")
    assert len(sampled) == target, f"Stratified sample size mismatch: {len(sampled)} != {target}"

    out_test = DATA_DIR / "math_test_1500.jsonl"
    with open(out_test, "w") as f:
        for row in sampled:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(sampled)} rows to {out_test}")


# ---------------------------------------------------------------------------
# Phase 2: merge sweep shards, apply learnability filter -> rl_train.jsonl
# ---------------------------------------------------------------------------

def run_phase2(sweep_glob: str) -> None:
    """Merge sweep shards, filter 1 <= C <= 15, write rl_train.jsonl."""
    shard_paths = sorted(glob.glob(sweep_glob))
    if not shard_paths:
        print(f"ERROR: No files found matching {sweep_glob!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_paths)} sweep shard(s):")
    for p in shard_paths:
        print(f"  {p}")

    rows: list[dict] = []
    for p in shard_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"Total sweep rows loaded: {len(rows)}")

    # Validate schema
    required_fields = {"problem_id", "problem", "answer", "level", "type",
                       "C", "G", "correct_mask", "canonicals",
                       "flippable_maj8", "mean_len", "trunc_frac"}
    for i, row in enumerate(rows[:5]):
        missing = required_fields - set(row.keys())
        if missing:
            print(f"WARNING: row {i} missing fields: {missing}")

    # Learnability filter: 1 <= C <= G-1 (G=16 -> 1..15)
    G = rows[0]["G"] if rows else 16
    kept = []
    c_dist: Counter = Counter()
    dropped_all_wrong = 0
    dropped_all_correct = 0

    for row in rows:
        C = int(row["C"])
        c_dist[C] += 1
        if C == 0:
            dropped_all_wrong += 1
        elif C >= G:
            dropped_all_correct += 1
        else:
            kept.append({
                "problem": row["problem"],
                "answer": row["answer"],
                "level": row["level"],
                "type": row["type"],
                "C16": C,
                "weight": 1.0,
            })

    print(f"\nLearnability filter (1 <= C <= {G-1}):")
    print(f"  Total sweep rows:     {len(rows)}")
    print(f"  Dropped (C=0):        {dropped_all_wrong}")
    print(f"  Dropped (C={G}):      {dropped_all_correct}")
    print(f"  Kept for RL training: {len(kept)}")
    print(f"\nC distribution (before filter):")
    for c in sorted(c_dist.keys()):
        print(f"  C={c:2d}: {c_dist[c]} problems")

    out_path = DATA_DIR / "rl_train.jsonl"
    with open(out_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(kept)} rows to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MATH train data for RL training (two-phase)."
    )
    parser.add_argument(
        "--from-sweep",
        action="store_true",
        help="Phase 2: merge sweep shards and build rl_train.jsonl.",
    )
    parser.add_argument(
        "--sweep-glob",
        default=str(DATA_DIR / "sweep*.jsonl"),
        help="Glob pattern for sweep shard files (phase 2).",
    )
    args = parser.parse_args()

    if args.from_sweep:
        run_phase2(args.sweep_glob)
    else:
        run_phase1()


if __name__ == "__main__":
    main()
