"""
verify_d3_headline.py — INDEPENDENT recompute of the D3-vs-arm_a headline from
the raw scored shards, using a deliberately different code path than
eval_d3_compare.py (explicit per-row loops, independent wvote/plain-maj impls,
explicit join-coverage audit). Goal: catch any bug in the headline before it is
reported. Also dumps per-band join coverage and any unjoined ('?') rows.

Run:
  python -m rl.analysis.verify_d3_headline \
     --d3_dir   rl/eval_outputs/arm_d_prmmargin_band/scored \
     --arma_dir rl/analysis/outputs/prm_scored_steps
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict
import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
BAND_DATA = os.path.join(REPO, "rl", "analysis", "outputs", "band_eval_data.jsonl")
N_DEAD, N_WON, N_BAND, N_POOL, BASE = 349, 1064, 578, 1991, 0.540


def load_bandmap():
    m = {}
    for line in open(BAND_DATA):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m[r["problem_id"]] = r["band"]
    return m


def pm_one(canon, cm):
    """plain majority, drop empty; credit = fraction of tied-top classes that are correct."""
    votes = Counter(c for c in canon if c != "")
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [c for c, v in votes.items() if v == top]
    correct = {canon[i] for i in range(len(canon)) if cm[i] > 0.5}
    return sum(1 for c in tied if c in correct) / len(tied)


def wv_one(canon, last, cm, p=4):
    """PRM-weighted vote, last-agg, exponent p, drop empty."""
    mass = defaultdict(float)
    for c, s in zip(canon, last):
        if c == "":
            continue
        mass[c] += max(float(s), 0.0) ** p
    if not mass:
        return 0.0
    top = max(mass.values())
    tied = [c for c, v in mass.items() if v == top]
    correct = {canon[i] for i in range(len(canon)) if cm[i] > 0.5}
    return sum(1 for c in tied if c in correct) / len(tied)


def load_rows(d, bmap):
    rows_by_band = defaultdict(list)
    files = sorted(set(glob.glob(os.path.join(d, "scored_shard*.jsonl")) +
                       glob.glob(os.path.join(d, "*.jsonl"))))
    unjoined = 0
    total = 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "step_scores" not in r:
                continue
            total += 1
            pid = r.get("problem_id") or r.get("unique_id")
            band = bmap.get(pid)
            if band is None:
                unjoined += 1
                band = "?"
            last = [float(s[-1]) if s else 0.0 for s in r["step_scores"]]
            rows_by_band[band].append((r["canonicals"], last, r["correct_mask"]))
    return rows_by_band, files, total, unjoined


def summ(rows_by_band, label):
    res = {}
    print(f"\n=== {label} ===")
    print(f"  files joined; bands present: { {k: len(v) for k, v in rows_by_band.items()} }")
    for band in ("misselected", "control"):
        rows = rows_by_band.get(band, [])
        if not rows:
            res[band] = None
            continue
        # also report n_completions per row to confirm 64
        ncs = Counter(len(c) for c, _, _ in rows)
        pm = float(np.mean([pm_one(c, cm) for c, _, cm in rows]))
        wv = float(np.mean([wv_one(c, l, cm) for c, l, cm in rows]))
        res[band] = (len(rows), pm, wv)
        print(f"  {band:12s} n={len(rows):4d}  plain_maj={pm:.4f}  wvote(last,p4)={wv:.4f}  ncomp={dict(ncs)}")
    if res.get("misselected") and res.get("control"):
        b, c = res["misselected"], res["control"]
        pool_pm = (N_WON * c[1] + N_BAND * b[1]) / N_POOL
        pool_wv = (N_WON * c[2] + N_BAND * b[2]) / N_POOL
        res["pool"] = (pool_pm, pool_wv)
        print(f"  {'pool proj':12s} n={N_POOL:4d}  plain_maj={pool_pm:.4f} ({pool_pm-BASE:+.4f})  "
              f"wvote={pool_wv:.4f} ({pool_wv-BASE:+.4f})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3_dir", default=os.path.join(REPO, "rl/eval_outputs/arm_d_prmmargin_band/scored"))
    ap.add_argument("--arma_dir", default=os.path.join(REPO, "rl/analysis/outputs/prm_scored_steps"))
    args = ap.parse_args()
    bmap = load_bandmap()
    print(f"bandmap: {len(bmap)} problems; band counts = {Counter(bmap.values())}")

    a_rows, a_files, a_tot, a_unj = load_rows(args.arma_dir, bmap)
    d_rows, d_files, d_tot, d_unj = load_rows(args.d3_dir, bmap)
    print(f"\narm_a dir: {len(a_files)} files, {a_tot} rows, {a_unj} unjoined")
    print(f"D3    dir: {len(d_files)} files, {d_tot} rows, {d_unj} unjoined")

    a = summ(a_rows, "arm_a (cached bar)")
    d = summ(d_rows, "D3 prm_margin (fresh)")

    print("\n=== HEADLINE (D3 - arm_a), independent recompute ===")
    if a.get("misselected") and d.get("misselected"):
        print(f"  band plain-maj : {d['misselected'][1]-a['misselected'][1]:+.4f}")
        print(f"  band wvote     : {d['misselected'][2]-a['misselected'][2]:+.4f}")
    if a.get("control") and d.get("control"):
        print(f"  control plain  : {d['control'][1]-a['control'][1]:+.4f}")
        print(f"  control wvote  : {d['control'][2]-a['control'][2]:+.4f}")
    if a.get("pool") and d.get("pool"):
        print(f"  pool plain-maj : {d['pool'][0]-a['pool'][0]:+.4f}")
        print(f"  pool wvote     : {d['pool'][1]-a['pool'][1]:+.4f}")


if __name__ == "__main__":
    main()
