"""
analyze_confidence.py — Phase 1a decisive test.

Question: on grpo's mis-selected band, can the model's OWN confidence (token
logprobs) pick the correct minority over its confidently-wrong plurality?

If maxconf/conf-weighted-vote >> majority vote on the band, confidence is a free
PRM-free selector. If maxconf ~ maj (and within-problem AUC ~ 0.5), the model is
genuinely confidently-wrong -> an EXTERNAL verifier is required (Phase 1b).

Reads rl/analysis/outputs/redecode/shard*.jsonl, writes PHASE1a.md + figure.
Run:  python -m rl.analysis.analyze_confidence
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
RD = os.path.join(OUT, "redecode")
lines: list[str] = []


def say(s: str = ""):
    print(s); lines.append(s)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(RD, "shard*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# ---- selectors over one problem's fresh samples -> utility in [0,1] ----
def plurality_util(canon, correct, weights=None):
    """weighted plurality; weights default to 1 (count). utility = correct mass
    in argmax / |argmax-classes|... reduce to standard: pick max-weight class,
    fractional credit on ties."""
    correct_classes = {canon[i] for i in range(len(canon)) if correct[i] > 0.5}
    w = np.ones(len(canon)) if weights is None else np.asarray(weights, dtype=float)
    agg = defaultdict(float)
    for c, wi in zip(canon, w):
        agg[c] += wi
    top = max(agg.values())
    argmax = [c for c, v in agg.items() if abs(v - top) < 1e-9]
    cin = sum(1 for c in argmax if c in correct_classes)
    return cin / len(argmax)


def maxconf_util(canon, correct, score):
    """best-of-n: pick the single sample with the highest score; 1 if correct."""
    s = np.asarray(score, dtype=float)
    if np.all(np.isnan(s)):
        return float("nan")
    i = int(np.nanargmax(s))
    return float(correct[i] > 0.5)


def softmax_w(score, tau):
    s = np.asarray(score, dtype=float)
    s = np.where(np.isnan(s), np.nanmin(s) - 10, s)
    s = s / tau
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def pooled_auc(scores, labels):
    """AUC = P(score|correct > score|wrong), Mann-Whitney, pooled over samples."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    # (simple version; tie effect negligible on continuous logprobs)
    r_pos = ranks[y > 0.5].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rows = load()
    by_band = defaultdict(list)
    for r in rows:
        by_band[r["band"]].append(r)
    say("# Phase 1a — can the model's own confidence select the answer?\n")
    say(f"Re-decoded {len(rows)} problems with token logprobs "
        f"(misselected={len(by_band['misselected'])}, control={len(by_band['control'])}), "
        f"grpo, n=64, T=1.0.\n")

    # ---- reproduction of the band on fresh samples ----
    mis = by_band["misselected"]
    solv = sum(1 for r in mis if sum(r["correct_mask"]) >= 1)
    say("## Fresh-decode reproduction (mis-selected band)\n")
    say(f"- of {len(mis)} band problems, {solv} ({100*solv/len(mis):.0f}%) still "
        f"have >=1 correct sample (coverage reproduces).\n")

    # ---- selector accuracy by band ----
    say("## Selector accuracy (single decision over 64 fresh samples)\n")
    selectors = {
        "oracle (any correct)": lambda r: 1.0 if sum(r["correct_mask"]) >= 1 else 0.0,
        "majority vote": lambda r: plurality_util(r["canonicals"], r["correct_mask"]),
        "best-of-n: max mean_lp": lambda r: maxconf_util(r["canonicals"], r["correct_mask"], r["mean_lp"]),
        "best-of-n: max tail_lp": lambda r: maxconf_util(r["canonicals"], r["correct_mask"], r["tail_lp"]),
        "best-of-n: max sum_lp": lambda r: maxconf_util(r["canonicals"], r["correct_mask"], r["cum_lp"]),
        "conf-weighted vote (mean_lp, tau=.3)":
            lambda r: plurality_util(r["canonicals"], r["correct_mask"], softmax_w(r["mean_lp"], 0.3)),
        "conf-weighted vote (mean_lp, tau=.1)":
            lambda r: plurality_util(r["canonicals"], r["correct_mask"], softmax_w(r["mean_lp"], 0.1)),
    }
    bands = ["misselected", "control"]
    say("| selector | " + " | ".join(bands) + " | all |")
    say("|" + "---|" * (len(bands) + 2))
    acc = {}
    for name, fn in selectors.items():
        acc[name] = {}
        cells = []
        for b in bands + ["all"]:
            rr = rows if b == "all" else by_band[b]
            vals = [fn(r) for r in rr]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            a = float(np.mean(vals))
            acc[name][b] = a
            cells.append(f"{a:.3f}")
        say(f"| {name} | " + " | ".join(cells) + " |")
    say("")
    # %-of-gap recovered on the band (vs majority, toward oracle)
    base = acc["majority vote"]["misselected"]
    orac = acc["oracle (any correct)"]["misselected"]
    say(f"On the band: majority={base:.3f}, oracle={orac:.3f}, "
        f"gap={orac-base:.3f}. %-of-gap recovered by confidence:\n")
    say("| selector | band acc | %-gap recovered |")
    say("|---|---|---|")
    for name in selectors:
        if name in ("oracle (any correct)", "majority vote"):
            continue
        rec = (acc[name]["misselected"] - base) / (orac - base) if orac > base else float("nan")
        say(f"| {name} | {acc[name]['misselected']:.3f} | {rec*100:.1f}% |")
    say("")

    # ---- calibration: is confidence higher for correct samples? ----
    say("## Does confidence separate correct from wrong samples? (band)\n")
    all_scores = {"mean_lp": [], "tail_lp": []}
    all_labels = []
    perprob_auc = []
    for r in mis:
        y = np.asarray(r["correct_mask"])
        if 1 <= y.sum() <= len(y) - 1:
            a = pooled_auc(r["mean_lp"], y)
            if not np.isnan(a):
                perprob_auc.append(a)
        for key in all_scores:
            all_scores[key].extend(r[key])
        all_labels.extend(r["correct_mask"])
    lab = np.asarray(all_labels)
    for key in all_scores:
        s = np.asarray(all_scores[key], dtype=float)
        m = ~np.isnan(s)
        cmean = s[m & (lab > 0.5)].mean()
        wmean = s[m & (lab <= 0.5)].mean()
        auc = pooled_auc(s, lab)
        say(f"- **{key}**: mean(correct)={cmean:.3f}, mean(wrong)={wmean:.3f}, "
            f"gap={cmean-wmean:+.3f}, pooled sample-level AUC={auc:.3f}")
    say(f"- **within-problem AUC** (mean_lp, avg over {len(perprob_auc)} band "
        f"problems with both classes): {np.mean(perprob_auc):.3f}  "
        f"(0.5 = confidence is useless / model is confidently-wrong)\n")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(selectors.keys())
    xs = np.arange(len(names))
    ax[0].barh(xs, [acc[n]["misselected"] for n in names], color="#b4436c")
    ax[0].axvline(acc["majority vote"]["misselected"], color="k", ls="--", lw=1, label="majority")
    ax[0].axvline(acc["oracle (any correct)"]["misselected"], color="g", ls="--", lw=1, label="oracle")
    ax[0].set_yticks(xs); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlabel("accuracy on mis-selected band"); ax[0].legend(fontsize=8)
    ax[0].set_title("Can confidence beat majority vote on the band?")
    # confidence distributions
    s = np.asarray(all_scores["mean_lp"], dtype=float); m = ~np.isnan(s)
    ax[1].hist(s[m & (lab > 0.5)], bins=40, alpha=0.6, density=True, label="correct samples", color="#4c9f70")
    ax[1].hist(s[m & (lab <= 0.5)], bins=40, alpha=0.6, density=True, label="wrong samples", color="#9aa0a6")
    ax[1].set_xlabel("mean token logprob (confidence)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Confidence: correct vs wrong samples (band)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_phase1a.png"), dpi=130)
    plt.close(fig)
    say("Figure: fig_phase1a.png")

    with open(os.path.join(OUT, "PHASE1a.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n[wrote", os.path.join(OUT, "PHASE1a.md"), "]")


if __name__ == "__main__":
    main()
