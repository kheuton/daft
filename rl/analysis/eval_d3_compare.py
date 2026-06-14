"""
eval_d3_compare.py — compare the D3 (prm_margin) policy to arm_a on the 878
band+control problems under BOTH plain majority and the DEPLOYED rule
PRM-weighted-vote(last, p=4, drop-empty).

arm_a (cached, the bar): rl/analysis/outputs/prm_scored_steps/scored_shard*.jsonl
D3 (fresh): the dir produced by eval_d3_band.sbatch (quick_eval D3/final ->
score_prm --save_step_scores). Same problem_ids, same scorer.

Headline questions:
  (1) plain maj on the band: did the D3 POLICY itself select correct more often
      (without the verifier)?  arm_a band plain-maj ~0.02 by construction.
  (2) PRM-wvote(last,p4) band: did D3 raise the verifier-recoverable rate above
      arm_a's ~0.27?  (the on-thesis win)
  (3) control maj must NOT regress (safety).

Run:  python -m rl.analysis.eval_d3_compare --d3_dir rl/eval_outputs/arm_d_prmmargin_band
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
ARM_A_DIR = os.path.join(REPO, "rl", "analysis", "outputs", "prm_scored_steps")
N_DEAD, N_WON, N_BAND, N_POOL, BASE = 349, 1064, 578, 1991, 0.540
P = 4


def agg_last(steps):
    return float(steps[-1]) if steps else 0.0


def correct_classes(canon, cm):
    return {canon[i] for i in range(len(canon)) if cm[i] > 0.5}


def plain_maj(canon, cm, drop_empty=True):
    cc = correct_classes(canon, cm)
    acc = defaultdict(int)
    for c in canon:
        if drop_empty and c == "":
            continue
        acc[c] += 1
    if not acc:
        return 0.0
    top = max(acc.values()); am = [c for c, v in acc.items() if v == top]
    return sum(1 for c in am if c in cc) / len(am)


def wvote(canon, scores, cm, p=P, drop_empty=True):
    cc = correct_classes(canon, cm)
    acc = defaultdict(float)
    for c, s in zip(canon, scores):
        if drop_empty and c == "":
            continue
        acc[c] += max(s, 0.0) ** p
    if not acc:
        return 0.0
    top = max(acc.values()); am = [c for c, v in acc.items() if abs(v - top) < 1e-12 or v == top]
    return sum(1 for c in am if c in cc) / len(am)


BAND_DATA = os.path.join(REPO, "rl", "analysis", "outputs", "band_eval_data.jsonl")


def band_map():
    m = {}
    for line in open(BAND_DATA):
        if line.strip():
            r = json.loads(line)
            m[r["problem_id"]] = r.get("band", "?")
    return m


def load(scored_dir, bmap):
    by = defaultdict(list)
    for f in sorted(set(glob.glob(os.path.join(scored_dir, "scored_shard*.jsonl")) +
                        glob.glob(os.path.join(scored_dir, "*.jsonl")))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "step_scores" not in r:
                continue
            r["_last"] = [agg_last(s) for s in r["step_scores"]]
            r["_cm"] = r["correct_mask"]
            # band joined by problem_id (quick_eval does not preserve the field).
            pid = r.get("problem_id") or r.get("unique_id")
            band = bmap.get(pid, r.get("band", "?"))
            by[band].append(r)
    return by


def summarize(by, label):
    out = {}
    for band in ("misselected", "control"):
        rows = by.get(band, [])
        if not rows:
            out[band] = None
            continue
        pm = float(np.mean([plain_maj(r["canonicals"], r["_cm"]) for r in rows]))
        wv = float(np.mean([wvote(r["canonicals"], r["_last"], r["_cm"]) for r in rows]))
        out[band] = (len(rows), pm, wv)
    b = out.get("misselected"); c = out.get("control")
    if b and c:
        pool_pm = (N_WON * c[1] + N_BAND * b[1]) / N_POOL
        pool_wv = (N_WON * c[2] + N_BAND * b[2]) / N_POOL
        out["pool"] = (pool_pm, pool_wv)
    print(f"\n## {label}")
    print("| band | n | plain maj | PRM-wvote(last,p4) |")
    print("|---|---|---|---|")
    for band in ("misselected", "control"):
        if out.get(band):
            n, pm, wv = out[band]
            print(f"| {band} | {n} | {pm:.3f} | {wv:.3f} |")
    if "pool" in out:
        print(f"| **pool proj** | {N_POOL} | {out['pool'][0]:.3f} ({out['pool'][0]-BASE:+.3f}) | "
              f"{out['pool'][1]:.3f} ({out['pool'][1]-BASE:+.3f}) |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3_dir", required=True, help="dir with D3 scored_shard*.jsonl")
    args = ap.parse_args()
    bmap = band_map()
    arm_a = summarize(load(ARM_A_DIR, bmap), "arm_a (cached, the bar)")
    d3 = summarize(load(args.d3_dir, bmap), "arm_d prm_margin (D3, fresh)")
    print("\n## HEADLINE (D3 - arm_a)")
    if arm_a.get("misselected") and d3.get("misselected"):
        print(f"  band plain-maj:  {d3['misselected'][1]-arm_a['misselected'][1]:+.3f}  "
              f"(did the D3 POLICY itself select better on the band?)")
        print(f"  band PRM-wvote:  {d3['misselected'][2]-arm_a['misselected'][2]:+.3f}  "
              f"(on-thesis: did D3 raise the verifier-recoverable rate?)")
    if arm_a.get("control") and d3.get("control"):
        print(f"  control PRM-wvote: {d3['control'][2]-arm_a['control'][2]:+.3f}  (safety: must not regress)")
    if arm_a.get("pool") and d3.get("pool"):
        print(f"  pool PRM-wvote:  {d3['pool'][1]-arm_a['pool'][1]:+.3f}")
    print("\nNOTE: arm_a band completions are its OWN rollouts (mis-selected by construction); "
          "D3 are fresh D3-policy rollouts on the SAME problem_ids. A positive band delta = "
          "D3 training moved the deployed decision toward correct. Pool proj reuses dead349/won1064/band578.")


if __name__ == "__main__":
    main()
