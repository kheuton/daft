"""
analyze_agg.py — PRM verifier ceiling with PROPER per-step scoring.

Reads rl/analysis/outputs/prm_scored_steps/scored_shard*.jsonl (now real per-step
PRM scores after the step-split fix). For each aggregation (last/min/prod/mean)
reports the best decision rule's band/control/pool/AUC; then sweeps rules under
the best aggregation. Answers: does proper per-step scoring + min/prod raise the
verifier ceiling above the whole-sequence result (+8.2pt, AUC 0.69)?

Run:  python -m rl.analysis.analyze_agg
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
OUT = os.path.join(REPO, "rl", "analysis", "outputs")
SC = os.path.join(OUT, "prm_scored_steps")
N_DEAD, N_WON, N_BAND, N_POOL, BASE = 349, 1064, 578, 1991, 0.540
lines: list[str] = []


def say(s=""):
    print(s); lines.append(s)


def aggregate(steps, strat):
    if not steps:
        return 0.0
    if strat == "last":
        return float(steps[-1])
    if strat == "min":
        return float(min(steps))
    if strat == "prod":
        return float(math.prod(steps))
    if strat == "mean":
        return float(sum(steps) / len(steps))
    raise ValueError(strat)


def correct_classes(r):
    return {r["canonicals"][i] for i in range(len(r["canonicals"])) if r["correct_mask"][i] > 0.5}


def _vote(items, cc):
    a = defaultdict(float)
    for c, w in items:
        a[c] += w
    if not a:
        return 0.0
    top = max(a.values()); am = [c for c, v in a.items() if abs(v - top) < 1e-12]
    return sum(1 for c in am if c in cc) / len(am)


def wvote(r, scores, p=2.0, drop_empty=True):
    cc = correct_classes(r)
    return _vote([(c, max(s, 0.0) ** p) for c, s in zip(r["canonicals"], scores)
                  if not (drop_empty and c == "")], cc)


def bon(r, scores, drop_empty=True):
    s = np.asarray(scores, float).copy()
    if drop_empty:
        for i, c in enumerate(r["canonicals"]):
            if c == "":
                s[i] = -np.inf
    if not np.isfinite(s).any():
        return 0.0
    return float(r["correct_mask"][int(np.argmax(s))] > 0.5)


def auc_prob(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    by = defaultdict(list)
    for r in rows:
        by[r["band"]].append(r)
    mis, ctl = by["misselected"], by["control"]

    nsteps = [len(ss) for r in rows for ss in r["step_scores"]]
    say("# Phase 1b — PRM ceiling with proper per-step scoring\n")
    say(f"band={len(mis)} control={len(ctl)}. Steps/completion now: "
        f"mean={np.mean(nsteps):.1f}, median={int(np.median(nsteps))}, max={max(nsteps)} "
        f"(was 1 before the split fix).\n")

    def stats(scored, rule):
        b = float(np.mean([rule(r, scored[id(r)]) for r in mis]))
        c = float(np.mean([rule(r, scored[id(r)]) for r in ctl]))
        pool = (N_WON * c + N_BAND * b) / N_POOL
        aucs = [auc_prob(scored[id(r)], r["correct_mask"]) for r in mis
                if 1 <= sum(r["correct_mask"]) <= len(r["correct_mask"]) - 1]
        aucs = [a for a in aucs if not np.isnan(a)]
        return b, c, pool, float(np.mean(aucs))

    say("## Aggregation sweep (rule = PRM-weighted vote p=2, drop-empty)\n")
    say("| aggregation | band | control | pool maj | band %-gap | AUC | pool Δ |")
    say("|" + "---|" * 7)
    agg_pool = {}
    for strat in ["last", "min", "prod", "mean"]:
        scored = {id(r): [aggregate(ss, strat) for ss in r["step_scores"]] for r in rows}
        b, c, pool, a = stats(scored, lambda r, s: wvote(r, s, 2.0))
        agg_pool[strat] = pool
        say(f"| {strat} | {b:.3f} | {c:.3f} | {pool:.3f} | {(b-0.02)/0.98*100:.1f}% | {a:.3f} | {pool-BASE:+.3f} |")
    best = max(agg_pool, key=agg_pool.get)
    say(f"\n**best aggregation: {best}** (pool {agg_pool[best]:.3f}). Rule sweep under it:\n")

    scored = {id(r): [aggregate(ss, best) for ss in r["step_scores"]] for r in rows}
    say(f"| rule ({best} agg) | band | control | pool maj | band %-gap | pool Δ |")
    say("|" + "---|" * 6)
    rules = [("best-of-n (drop-empty)", lambda r, s: bon(r, s)),
             ("wvote p=1", lambda r, s: wvote(r, s, 1)),
             ("wvote p=2", lambda r, s: wvote(r, s, 2)),
             ("wvote p=4", lambda r, s: wvote(r, s, 4))]
    for name, rule in rules:
        b, c, pool, _ = stats(scored, rule)
        say(f"| {name} | {b:.3f} | {c:.3f} | {pool:.3f} | {(b-0.02)/0.98*100:.1f}% | {pool-BASE:+.3f} |")
    say("")
    say("Compare to whole-sequence PRM (1-step, prior): wvote p=2 -> band 0.356, "
        "pool 0.622 (+8.2pt), AUC 0.690.")

    with open(os.path.join(OUT, "PHASE1b_agg.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[wrote", os.path.join(OUT, "PHASE1b_agg.md"), "]")


if __name__ == "__main__":
    main()
