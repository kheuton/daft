"""
build_band_completions.py — assemble the completions.jsonl to PRM-score (Phase 1b).

Filters the CACHED grpo rollouts (which retain completion TEXT, and on which the
mis-selected band was defined) down to the Phase-1a target problems
(band + control), so rl.score_prm can score them. Writes one file in the exact
schema score_prm expects (needs `problem` + `completions`).

Run:  python -m rl.analysis.build_band_completions
"""

from __future__ import annotations

import glob
import json
import os

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
EVAL = os.path.join(REPO, "rl", "eval_outputs", "arm_a_grpo_final")
DATASETS = ["math500", "math_test_1500"]
OUT = os.path.join(REPO, "rl", "analysis", "outputs")
TARGETS = os.path.join(OUT, "redecode_targets.jsonl")


def main():
    band = {}
    with open(TARGETS) as f:
        for line in f:
            r = json.loads(line)
            band[r["problem_id"]] = r["band"]
    print(f"targets: {len(band)}")

    cached = {}
    for ds in DATASETS:
        pat = os.path.join(EVAL, ds, "T1.0_n64", "shard*", "completions.jsonl")
        for fp in sorted(glob.glob(pat)):
            with open(fp) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r["problem_id"] in band:
                        cached[r["problem_id"]] = r  # last-wins (matches analysis)

    out_path = os.path.join(OUT, "band_completions.jsonl")
    n = 0
    with open(out_path, "w") as fo:
        for pid, r in cached.items():
            row = {
                "problem_id": pid,
                "problem": r["problem"],
                "answer": r["answer"],
                "completions": r["completions"],
                "canonicals": r["canonicals"],
                "correct_mask": r["correct_mask"],
                "band": band[pid],
            }
            fo.write(json.dumps(row) + "\n")
            n += 1
    missing = set(band) - set(cached)
    print(f"wrote {n} rows to {out_path}  ({n*64} completions to score)")
    if missing:
        print(f"WARNING: {len(missing)} target ids not found in cached grpo completions")


if __name__ == "__main__":
    main()
