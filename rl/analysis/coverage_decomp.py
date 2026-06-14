"""
coverage_decomp.py — decompose the D3-vs-arm_a band/control deltas into a
COVERAGE component (does the problem have ANY correct completion among the 64
rollouts = pass@64) vs a SELECTION component (given coverage, does the deployed
rule pick correct). The project thesis is "bottleneck is SELECTION not coverage";
this checks whether D3's band gain is on-thesis (selection) or off-thesis
(coverage).

For each band, for each arm, report:
  cover   = frac of problems with >=1 correct completion (pass@64)
  ncorr   = mean #correct completions / 64
  maj     = plain-majority credit (drop-empty)
  wvote   = PRM-wvote(last,p4,drop-empty)
  sel|cov = wvote conditioned on covered problems  (selection efficiency)
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict
import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
BAND_DATA = os.path.join(REPO, "rl", "analysis", "outputs", "band_eval_data.jsonl")


def bandmap():
    m = {}
    for line in open(BAND_DATA):
        line = line.strip()
        if line:
            r = json.loads(line)
            m[r["problem_id"]] = r["band"]
    return m


def pm_one(canon, cm):
    votes = Counter(c for c in canon if c != "")
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [c for c, v in votes.items() if v == top]
    correct = {canon[i] for i in range(len(canon)) if cm[i] > 0.5}
    return sum(1 for c in tied if c in correct) / len(tied)


def wv_one(canon, last, cm, p=4):
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


def load(d, bmap):
    by = defaultdict(list)
    for f in sorted(set(glob.glob(os.path.join(d, "scored_shard*.jsonl")) +
                        glob.glob(os.path.join(d, "*.jsonl")))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "step_scores" not in r:
                continue
            pid = r.get("problem_id") or r.get("unique_id")
            band = bmap.get(pid, "?")
            last = [float(s[-1]) if s else 0.0 for s in r["step_scores"]]
            by[band].append((pid, r["canonicals"], last, r["correct_mask"]))
    return by


def report(by, label):
    print(f"\n=== {label} ===")
    print(f"{'band':12s} {'n':>4s} {'cover':>7s} {'ncorr':>7s} {'maj':>7s} {'wvote':>7s} {'wv|cov':>7s}")
    out = {}
    for band in ("misselected", "control"):
        rows = by.get(band, [])
        if not rows:
            continue
        cover, ncorr, maj, wv, wv_cov = [], [], [], [], []
        for pid, canon, last, cm in rows:
            nc = sum(1 for x in cm if x > 0.5)
            covered = nc > 0
            cover.append(covered)
            ncorr.append(nc / len(cm))
            maj.append(pm_one(canon, cm))
            w = wv_one(canon, last, cm)
            wv.append(w)
            if covered:
                wv_cov.append(w)
        out[band] = {
            "n": len(rows), "cover": float(np.mean(cover)), "ncorr": float(np.mean(ncorr)),
            "maj": float(np.mean(maj)), "wv": float(np.mean(wv)),
            "wv_cov": float(np.mean(wv_cov)) if wv_cov else float("nan"),
            "pids": {pid for pid, *_ in rows},
            "covered_pids": {pid for pid, c, l, cm in rows if sum(1 for x in cm if x > 0.5) > 0},
        }
        o = out[band]
        print(f"{band:12s} {o['n']:4d} {o['cover']:7.3f} {o['ncorr']:7.3f} {o['maj']:7.3f} {o['wv']:7.3f} {o['wv_cov']:7.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3_dir", default=os.path.join(REPO, "rl/eval_outputs/arm_d_prmmargin_band/scored"))
    ap.add_argument("--arma_dir", default=os.path.join(REPO, "rl/analysis/outputs/prm_scored_steps"))
    args = ap.parse_args()
    bmap = bandmap()
    a = report(load(args.arma_dir, bmap), "arm_a (cached)")
    d = report(load(args.d3_dir, bmap), "D3 prm_margin (fresh)")

    print("\n=== COVERAGE vs SELECTION decomposition (band: misselected) ===")
    ab, db = a["misselected"], d["misselected"]
    print(f"  coverage (pass@64):  arm_a {ab['cover']:.3f} -> D3 {db['cover']:.3f}  (delta {db['cover']-ab['cover']:+.3f})")
    print(f"  selection|covered :  arm_a {ab['wv_cov']:.3f} -> D3 {db['wv_cov']:.3f}  (delta {db['wv_cov']-ab['wv_cov']:+.3f})")
    print(f"  -> band wvote delta {db['wv']-ab['wv']:+.3f} = coverage-driven if cover delta dominates, "
          f"selection-driven if wv|cov delta dominates")
    # per-problem coverage flips
    gained = db["covered_pids"] - ab["covered_pids"]
    lost = ab["covered_pids"] - db["covered_pids"]
    print(f"  band coverage flips: D3 GAINED coverage on {len(gained)} problems, LOST on {len(lost)} "
          f"(net {len(gained)-len(lost):+d} of {ab['n']})")
    print("\n=== control safety ===")
    ac, dc = a["control"], d["control"]
    print(f"  coverage:  arm_a {ac['cover']:.3f} -> D3 {dc['cover']:.3f}")
    clost = ac["covered_pids"] - dc["covered_pids"]
    print(f"  control coverage LOST on {len(clost)} of {ac['n']} problems (D3 no longer produces any correct answer)")


if __name__ == "__main__":
    main()
