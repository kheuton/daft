"""
analyze_prm_ceiling.py — tighten the PRM verifier ceiling using ONLY cached
agg_scores (no new GPU). Phase 1b found best-of-n sometimes picks an unparseable
completion scored ~1.0; here we add a drop-unparseable filter and sweep selection
rules / powers / thresholds to find the best realizable PRM-free-of-retraining
decision rule, with the hard constraint that it must NOT tank easy problems.

Reads rl/analysis/outputs/prm_scored/scored_shard*.jsonl; writes PHASE1b_ceiling.md.
Run:  python -m rl.analysis.analyze_prm_ceiling
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
OUT = os.path.join(REPO, "rl", "analysis", "outputs")
SC = os.path.join(OUT, "prm_scored")
lines: list[str] = []

# pool composition (grpo, 1991 paired): dead never-correct + won-outright + band
N_DEAD, N_WON, N_BAND, N_POOL = 349, 1064, 578, 1991
BASE_POOL_MAJ = 0.540  # plain full-64 plurality, Phase 0


def say(s=""):
    print(s); lines.append(s)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def correct_classes(r):
    return {r["canonicals"][i] for i in range(len(r["canonicals"])) if r["correct_mask"][i] > 0.5}


def _vote(items, cc):
    """items: list of (canon, weight). utility = correct mass in argmax / |argmax|."""
    agg = defaultdict(float)
    for c, w in items:
        agg[c] += w
    if not agg:
        return 0.0
    top = max(agg.values())
    am = [c for c, v in agg.items() if abs(v - top) < 1e-12]
    return sum(1 for c in am if c in cc) / len(am)


# ---- selector family (all from cached agg_scores) ----
def maj(r, drop_empty=False):
    cc = correct_classes(r)
    items = [(c, 1.0) for c in r["canonicals"] if not (drop_empty and c == "")]
    return _vote(items, cc)


def bon(r, drop_empty=False):
    cc = correct_classes(r)
    s = np.asarray(r["agg_scores"], float).copy()
    if drop_empty:
        for i, c in enumerate(r["canonicals"]):
            if c == "":
                s[i] = -np.inf
    if not np.isfinite(s).any():
        return 0.0
    i = int(np.argmax(s))
    return float(r["correct_mask"][i] > 0.5)


def wvote(r, p=1.0, drop_empty=True):
    cc = correct_classes(r)
    items = [(c, max(sc, 0.0) ** p) for c, sc in zip(r["canonicals"], r["agg_scores"])
             if not (drop_empty and c == "")]
    return _vote(items, cc)


def thresh_maj(r, tau, drop_empty=True):
    cc = correct_classes(r)
    items = [(c, 1.0) for c, sc in zip(r["canonicals"], r["agg_scores"])
             if sc >= tau and not (drop_empty and c == "")]
    if not items:  # fall back to plain majority if nothing clears the bar
        return maj(r, drop_empty=drop_empty)
    return _vote(items, cc)


def maj_prm_tiebreak(r, drop_empty=True):
    """Plurality by COUNT; ties broken by higher class mean PRM score. Conservative:
    keeps majority on easy problems, only uses PRM to settle ties/near-ties."""
    cc = correct_classes(r)
    cnt = defaultdict(int); ssum = defaultdict(float)
    for c, sc in zip(r["canonicals"], r["agg_scores"]):
        if drop_empty and c == "":
            continue
        cnt[c] += 1; ssum[c] += sc
    if not cnt:
        return 0.0
    top = max(cnt.values())
    am = [c for c in cnt if cnt[c] == top]
    winner = max(am, key=lambda c: ssum[c] / cnt[c])
    return 1.0 if winner in cc else 0.0


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[r["band"]].append(r)
    mis, ctl = by["misselected"], by["control"]
    say("# Phase 1b ceiling — best realizable PRM decision rule (no retraining)\n")
    say(f"band={len(mis)} control={len(ctl)}; oracle=1.0 on band, plain majority=0.020.\n")

    def row_for(name, fn):
        b = float(np.mean([fn(r) for r in mis]))
        c = float(np.mean([fn(r) for r in ctl]))
        pool = (N_DEAD * 0 + N_WON * c + N_BAND * b) / N_POOL
        recov = (b - 0.020) / (1.0 - 0.020)
        return name, b, c, pool, recov

    rules = [
        ("plain majority", lambda r: maj(r)),
        ("PRM best-of-n (all)", lambda r: bon(r)),
        ("PRM best-of-n (drop-empty)", lambda r: bon(r, drop_empty=True)),
        ("PRM-wvote p=1", lambda r: wvote(r, 1)),
        ("PRM-wvote p=2", lambda r: wvote(r, 2)),
        ("PRM-wvote p=4", lambda r: wvote(r, 4)),
        ("PRM-wvote p=8", lambda r: wvote(r, 8)),
        ("thresh-maj tau=0.5", lambda r: thresh_maj(r, 0.5)),
        ("thresh-maj tau=0.7", lambda r: thresh_maj(r, 0.7)),
        ("thresh-maj tau=0.9", lambda r: thresh_maj(r, 0.9)),
        ("maj + PRM tiebreak", lambda r: maj_prm_tiebreak(r)),
    ]
    results = [row_for(n, f) for n, f in rules]

    say("| rule | band | control | pool maj | band %-gap | pool Δ vs 0.540 |")
    say("|" + "---|" * 6)
    for n, b, c, pool, rec in results:
        say(f"| {n} | {b:.3f} | {c:.3f} | {pool:.3f} | {rec*100:.1f}% | {pool-BASE_POOL_MAJ:+.3f} |")
    say("")

    # best rule that does NOT hurt control much (>=0.97) maximizing pool
    safe = [r for r in results if r[2] >= 0.97]
    safe.sort(key=lambda r: -r[3])
    best_overall = max(results, key=lambda r: r[3])
    say(f"- **best pool maj overall**: {best_overall[0]} -> {best_overall[3]:.3f} "
        f"({best_overall[3]-BASE_POOL_MAJ:+.3f} vs majority)")
    if safe:
        s = safe[0]
        say(f"- **best 'safe' rule (control>=0.97)**: {s[0]} -> pool {s[3]:.3f} "
            f"({s[3]-BASE_POOL_MAJ:+.3f}), band {s[1]:.3f} ({s[4]*100:.1f}% of gap)")
    say("")
    say("Note: this is the ceiling for the EXISTING PRM with 'last' aggregation. "
        "min/prod aggregation and a held-out judge are separate (need re-score / new model).")

    with open(os.path.join(OUT, "PHASE1b_ceiling.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n[wrote", os.path.join(OUT, "PHASE1b_ceiling.md"), "]")


if __name__ == "__main__":
    main()
