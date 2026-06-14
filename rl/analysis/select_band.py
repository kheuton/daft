"""
select_band.py — pick the problems to re-decode with logprobs (Phase 1a).

Targets = grpo's mis-selected band (solvable but the deployed vote does NOT win
outright: best_correct <= best_wrong) + a random control sample of won-outright
problems (to confirm a confidence selector does not BREAK the easy ones).

Writes rl/analysis/outputs/redecode_targets.jsonl with {problem_id, problem,
answer, band}. Source = grpo cached completions (the deployment arm); we only
need each problem's text + gt, then we generate fresh samples WITH logprobs.

Run:  python -m rl.analysis.select_band
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
EVAL = os.path.join(REPO, "rl", "eval_outputs")
GRPO = "arm_a_grpo_final"
DATASETS = ["math500", "math_test_1500"]
OUT = os.path.join(REPO, "rl", "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)

N_CONTROL = 300
SEED = 0


def load_grpo():
    recs = {}
    for ds in DATASETS:
        pat = os.path.join(EVAL, GRPO, ds, "T1.0_n64", "shard*", "completions.jsonl")
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    recs[r["problem_id"]] = r  # last-wins dedup (matches analysis)
    return recs


def band_of(r) -> str:
    canon = r["canonicals"]
    correct = np.asarray(r["correct_mask"])
    C = int(correct.sum())
    if C == 0:
        return "dead"
    correct_classes = {canon[i] for i in range(len(canon)) if correct[i] > 0.5}
    counts = {}
    for c in canon:
        counts[c] = counts.get(c, 0) + 1
    best_correct = max((n for c, n in counts.items() if c in correct_classes), default=0)
    best_wrong = max((n for c, n in counts.items() if c not in correct_classes), default=0)
    return "won" if best_correct > best_wrong else "misselected"


def main():
    recs = load_grpo()
    mis, won = [], []
    for pid, r in recs.items():
        b = band_of(r)
        row = {"problem_id": pid, "problem": r["problem"], "answer": r["answer"]}
        if b == "misselected":
            row["band"] = "misselected"
            mis.append(row)
        elif b == "won":
            won.append({**row, "band": "control"})

    rng = np.random.default_rng(SEED)
    control = [won[i] for i in rng.choice(len(won), size=min(N_CONTROL, len(won)), replace=False)]
    targets = mis + control

    path = os.path.join(OUT, "redecode_targets.jsonl")
    with open(path, "w") as fh:
        for row in targets:
            fh.write(json.dumps(row) + "\n")

    print(f"grpo problems: {len(recs)}")
    print(f"  mis-selected (band): {len(mis)}")
    print(f"  won (control pool):  {len(won)}  -> sampled {len(control)}")
    print(f"  TOTAL targets:       {len(targets)}  ({len(targets)*64} generations @ n=64)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
