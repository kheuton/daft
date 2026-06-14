"""
analyze_prm.py — Phase 1b: how much of the selection gap can the PRM recover?

Reads rl/analysis/outputs/prm_scored/scored_shard*.jsonl (cached band+control
completions + per-completion agg_scores). On the mis-selected band (oracle=1.0 by
construction, since every band problem has >=1 correct sample), compares:
  majority vote  vs  PRM-best-of-n  vs  PRM-weighted-vote  vs  oracle
and reports the PRM's correct-vs-wrong discrimination AUC (compare to confidence's
within-problem AUC of 0.589 from Phase 1a). Stratifies the band into the rank-2
near-miss head vs the deep tail.

Run:  python -m rl.analysis.analyze_prm
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
OUT = os.path.join(REPO, "rl", "analysis", "outputs")
SC = os.path.join(OUT, "prm_scored")
lines: list[str] = []


def say(s: str = ""):
    print(s); lines.append(s)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def oracle(r):
    return 1.0 if sum(r["correct_mask"]) >= 1 else 0.0


def maj(r, weights=None):
    canon = r["canonicals"]; corr = r["correct_mask"]
    cc = {canon[i] for i in range(len(canon)) if corr[i] > 0.5}
    w = np.ones(len(canon)) if weights is None else np.asarray(weights, float)
    agg = defaultdict(float)
    for c, wi in zip(canon, w):
        agg[c] += wi
    top = max(agg.values())
    am = [c for c, v in agg.items() if abs(v - top) < 1e-12]
    return sum(1 for c in am if c in cc) / len(am)


def bon(r):
    s = np.asarray(r["agg_scores"], float)
    return float(r["correct_mask"][int(np.argmax(s))] > 0.5)


def wvote(r, power=1.0):
    s = np.asarray(r["agg_scores"], float) ** power
    return maj(r, weights=s)


def best_correct_rank(r):
    """Rank (1=most common) of the most-common correct canonical class."""
    canon = r["canonicals"]; corr = r["correct_mask"]
    cc = {canon[i] for i in range(len(canon)) if corr[i] > 0.5}
    cnt = defaultdict(int)
    for c in canon:
        cnt[c] += 1
    order = sorted(cnt.items(), key=lambda kv: -kv[1])
    for i, (c, _) in enumerate(order):
        if c in cc:
            return i + 1
    return None


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y > 0.5].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[r["band"]].append(r)
    mis = by["misselected"]; ctl = by["control"]
    say("# Phase 1b — PRM verifier ceiling on the selection gap\n")
    say(f"Scored {len(rows)} cached problems with Qwen2.5-Math-PRM-7B "
        f"(misselected={len(mis)}, control={len(ctl)}); these are the exact "
        f"samples the band was defined on, so oracle=1.0 on the band.\n")

    sels = {
        "oracle": oracle,
        "majority vote": lambda r: maj(r),
        "PRM best-of-n": bon,
        "PRM-weighted vote (p=1)": lambda r: wvote(r, 1.0),
        "PRM-weighted vote (p=4)": lambda r: wvote(r, 4.0),
    }
    say("## Selector accuracy\n")
    say("| selector | misselected band | control | all |")
    say("|" + "---|" * 4)
    acc = {}
    for name, fn in sels.items():
        acc[name] = {b: float(np.mean([fn(r) for r in rr])) for b, rr in
                     [("misselected", mis), ("control", ctl), ("all", rows)]}
        a = acc[name]
        say(f"| {name} | {a['misselected']:.3f} | {a['control']:.3f} | {a['all']:.3f} |")
    say("")
    base = acc["majority vote"]["misselected"]
    say(f"Band gap = oracle({1.0:.3f}) - majority({base:.3f}) = {1.0-base:.3f}. "
        f"%-of-gap recovered by the PRM:\n")
    for name in ["PRM best-of-n", "PRM-weighted vote (p=1)", "PRM-weighted vote (p=4)"]:
        rec = (acc[name]["misselected"] - base) / (1.0 - base)
        say(f"- {name}: band acc {acc[name]['misselected']:.3f}  ->  {rec*100:.1f}% of gap")
    say("")

    # discrimination AUC (compare to confidence 0.589)
    say("## PRM discrimination: correct vs wrong samples (band)\n")
    pooled_s, pooled_y, pp_auc = [], [], []
    for r in mis:
        y = r["correct_mask"]
        pooled_s.extend(r["agg_scores"]); pooled_y.extend(y)
        if 1 <= sum(y) <= len(y) - 1:
            a = auc(r["agg_scores"], y)
            if not np.isnan(a):
                pp_auc.append(a)
    say(f"- pooled sample-level AUC: {auc(pooled_s, pooled_y):.3f}")
    say(f"- within-problem AUC (avg over {len(pp_auc)} band problems): "
        f"**{np.mean(pp_auc):.3f}**  (Phase 1a confidence was 0.589; 0.5=useless)\n")

    # stratify band by correct-answer rank
    say("## Where does the PRM help? (band stratified by correct-answer rank)\n")
    say("| stratum | n | majority | PRM best-of-n | PRM-wvote p=4 |")
    say("|" + "---|" * 5)
    strata = [("rank-2 (near miss)", lambda r: best_correct_rank(r) == 2),
              ("rank 3-5", lambda r: best_correct_rank(r) in (3, 4, 5)),
              ("rank 6+ (deep tail)", lambda r: (best_correct_rank(r) or 99) >= 6)]
    for label, pred in strata:
        sub = [r for r in mis if pred(r)]
        if not sub:
            continue
        say(f"| {label} | {len(sub)} | {np.mean([maj(r) for r in sub]):.3f} "
            f"| {np.mean([bon(r) for r in sub]):.3f} "
            f"| {np.mean([wvote(r,4.0) for r in sub]):.3f} |")
    say("")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["majority vote", "PRM best-of-n", "PRM-weighted vote (p=1)", "PRM-weighted vote (p=4)", "oracle"]
    ax[0].bar(range(len(names)), [acc[n]["misselected"] for n in names],
              color=["#9aa0a6", "#4c9f70", "#3f7cac", "#2a4d69", "#b4436c"])
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("accuracy on mis-selected band"); ax[0].set_ylim(0, 1)
    ax[0].set_title("PRM verifier vs majority on the band")
    s = np.asarray(pooled_s, float); y = np.asarray(pooled_y)
    ax[1].hist(s[y > 0.5], bins=30, alpha=0.6, density=True, label="correct", color="#4c9f70")
    ax[1].hist(s[y <= 0.5], bins=30, alpha=0.6, density=True, label="wrong", color="#9aa0a6")
    ax[1].set_xlabel("PRM agg_score"); ax[1].legend(); ax[1].set_title("PRM score: correct vs wrong (band)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_phase1b.png"), dpi=130)
    plt.close(fig)
    say("Figure: fig_phase1b.png")

    with open(os.path.join(OUT, "PHASE1b.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n[wrote", os.path.join(OUT, "PHASE1b.md"), "]")


if __name__ == "__main__":
    main()
