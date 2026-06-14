"""
verify_distinguish.py — independently verify the Phase-2 design panel's LOAD-BEARING
claim before committing the overnight run: under per-step PRM aggregation, which agg
makes the dense PRM-weighted-correctness reward (D2: reward_i=(agg_i**p)*correct_i)
genuinely DISTINCT from plain GRPO control (D0: A_i=c_i-mean(c)), and what scale
constant matches D0's effective step size.

For each agg in {last,min,prod,mean} and p in {1,2,4}:
  (1) PRM-score distribution among CORRECT completions (mean/std/frac>0.9/frac==0)
      -> tests "last piles at 1.0 -> D2 collapses to control".
  (2) cosine(A_D2, A_D0) over simulated G=16 groups -> distinguishability.
  (3) raw D2 advantage RMS (incl zeros) and ratio to D0's RMS -> scale constant
      = 0.3066 * (rms_D2/rms_D0)  (D0 grpo scale is 0.3066).
  (4) deployed PRM-weighted-vote(p,drop-empty) band/control/pool recovery -> which
      agg is the best DEPLOYED rule (train must match deploy).

numpy-only. Reads rl/analysis/outputs/prm_scored_steps/scored_shard*.jsonl.
Run:  python -m rl.analysis.verify_distinguish
"""
from __future__ import annotations
import glob, json, math, os
from collections import defaultdict
import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
SC = os.path.join(REPO, "rl", "analysis", "outputs", "prm_scored_steps")
# pool projection constants (match analyze_agg.py): dead/won/band/pool, base maj
N_DEAD, N_WON, N_BAND, N_POOL, BASE = 349, 1064, 578, 1991, 0.540
GRPO_SCALE = 0.3066
G = 16
N_SUB = 4          # subsamples of G per problem
SEED = 0
AGGS = ["last", "min", "prod", "mean"]
PS = [1, 2, 4]


def agg_of(steps, strat):
    if not steps:
        return 0.0
    a = np.asarray(steps, dtype=np.float64)
    if strat == "last":
        return float(a[-1])
    if strat == "min":
        return float(a.min())
    if strat == "prod":
        return float(np.prod(a))
    if strat == "mean":
        return float(a.mean())
    raise ValueError(strat)


def correct_classes(canon, cmask):
    return {canon[i] for i in range(len(canon)) if cmask[i] > 0.5}


def wvote(canon, scores, cmask, p, drop_empty=True):
    cc = correct_classes(canon, cmask)
    acc = defaultdict(float)
    for c, s in zip(canon, scores):
        if drop_empty and c == "":
            continue
        acc[c] += max(s, 0.0) ** p
    if not acc:
        return 0.0
    top = max(acc.values())
    am = [c for c, v in acc.items() if abs(v - top) < 1e-300 or v == top]
    return sum(1 for c in am if c in cc) / len(am)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    by = defaultdict(list)
    for r in rows:
        by[r.get("band", "?")].append(r)
    print(f"# verify_distinguish — {len(rows)} problems "
          f"(band={len(by.get('misselected', []))} control={len(by.get('control', []))})\n")

    # Precompute per-problem agg-score arrays (64,) for each agg.
    for r in rows:
        ss = r["step_scores"]
        r["_agg"] = {a: np.array([agg_of(s, a) for s in ss], dtype=np.float64) for a in AGGS}
        r["_c"] = np.asarray(r["correct_mask"], dtype=np.float64)

    # ---- (1) PRM-score distribution among CORRECT completions ----
    print("## (1) agg-score distribution among CORRECT completions")
    print("| agg | n_correct | mean | std | frac>0.9 | frac==0 |")
    print("|" + "---|" * 6)
    for a in AGGS:
        vals = np.concatenate([r["_agg"][a][r["_c"] > 0.5] for r in rows if (r["_c"] > 0.5).any()])
        print(f"| {a} | {len(vals)} | {vals.mean():.3f} | {vals.std():.3f} | "
              f"{(vals > 0.9).mean():.3f} | {(vals == 0.0).mean():.3f} |")
    print()

    # ---- (2)+(3) cosine to control + advantage RMS, per (agg,p) ----
    rng = np.random.default_rng(SEED)
    # Build the simulated G=16 subsamples once (same indices across aggs/p).
    subs = []  # list of (problem_ref, idx array)
    for r in rows:
        n = len(r["_c"])
        if n < G:
            continue
        for _ in range(N_SUB):
            idx = rng.choice(n, size=G, replace=False)
            c = r["_c"][idx]
            if c.sum() == 0 or c.sum() == G:
                continue  # degenerate: D0 advantage is all-zero, no gradient
            subs.append((r, idx))
    print(f"## (2)+(3) cosine(A_D2,A_D0) and advantage RMS over {len(subs)} non-degenerate G={G} groups\n")
    print("| agg | p | mean cos | median cos | p10 cos | frac groups<0.8 | rmsD2/rmsD0 | scale_D2 |")
    print("|" + "---|" * 8)
    # D0 advantage RMS (incl zeros) for the same groups
    d0_sq = []
    for r, idx in subs:
        c = r["_c"][idx]
        a0 = c - c.mean()
        d0_sq.append((a0 ** 2))
    rms_d0 = math.sqrt(np.concatenate(d0_sq).mean())

    results = {}
    for a in AGGS:
        for p in PS:
            cosines = []
            d2_sq = []
            for r, idx in subs:
                c = r["_c"][idx]
                s = r["_agg"][a][idx]
                rwd = np.clip(s, 0.0, None) ** p * c
                a2 = rwd - rwd.mean()
                a0 = c - c.mean()
                d2_sq.append(a2 ** 2)
                n2, n0 = np.linalg.norm(a2), np.linalg.norm(a0)
                if n2 > 1e-12 and n0 > 1e-12:
                    cosines.append(float(np.dot(a2, a0) / (n2 * n0)))
            cosines = np.array(cosines)
            rms_d2 = math.sqrt(np.concatenate(d2_sq).mean())
            ratio = rms_d2 / rms_d0 if rms_d0 > 0 else float("nan")
            scale_d2 = GRPO_SCALE * ratio
            results[(a, p)] = dict(mean_cos=float(cosines.mean()), rms_d2=rms_d2, ratio=ratio, scale=scale_d2)
            print(f"| {a} | {p} | {cosines.mean():.3f} | {np.median(cosines):.3f} | "
                  f"{np.percentile(cosines,10):.3f} | {(cosines<0.8).mean():.3f} | "
                  f"{ratio:.3f} | {scale_d2:.4f} |")
    print(f"\n(D0 control advantage RMS over the same groups = {rms_d0:.4f}; "
          f"grpo scale_constants.json = {GRPO_SCALE})\n")

    # ---- (4) deployed PRM-weighted-vote recovery per (agg,p) ----
    mis = by.get("misselected", [])
    ctl = by.get("control", [])
    print("## (4) deployed PRM-weighted-vote(drop-empty) recovery (full 64-sample groups)\n")
    print("| agg | p | band | control | pool maj | pool Δ vs 0.540 |")
    print("|" + "---|" * 6)
    for a in AGGS:
        for p in PS:
            b = float(np.mean([wvote(r["canonicals"], r["_agg"][a], r["_c"], p) for r in mis])) if mis else float("nan")
            cc = float(np.mean([wvote(r["canonicals"], r["_agg"][a], r["_c"], p) for r in ctl])) if ctl else float("nan")
            pool = (N_WON * cc + N_BAND * b) / N_POOL
            print(f"| {a} | {p} | {b:.3f} | {cc:.3f} | {pool:.3f} | {pool-BASE:+.3f} |")
    print("\nNOTE: numbers on the CURRENT (possibly partial) cache; the waiter's analyze_agg "
          "on the COMPLETE job is authoritative for the deployed ceiling.")


if __name__ == "__main__":
    main()
