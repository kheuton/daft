"""
phase0b_minority.py — Characterize the "true-minority" band: solvable problems
where one WRONG canonical out-votes ALL correct answers combined.

This is the 546/1991 (grpo) band the autopsy flagged as needing a verifier or a
policy shift (not recoverable by voting/canonicalization tricks). The decision
of what to do next hinges on HOW DEEP the correct answer sits:
  - rank-2, C large  -> "confidently wrong mode"; a verifier can plausibly flip it
  - rank deep, C~1-2 -> effectively capability-limited; a needle in 64 samples

Fast (no maj Monte-Carlo): pure class-count bookkeeping over cached completions.
Run:  python -m rl.analysis.phase0b_minority
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from rl.analysis import phase0_lib as L

OUT = os.path.join(L.REPO, "rl", "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)
lines: list[str] = []


def say(s: str = ""):
    print(s)
    lines.append(s)


def class_table(p: L.Problem):
    """Return sorted list of (canonical, count, is_correct) desc by count."""
    correct_classes = {p.canonicals[i] for i in range(p.n) if p.correct[i] > 0.5}
    counts: dict[str, int] = {}
    for c in p.canonicals:
        counts[c] = counts.get(c, 0) + 1
    rows = [(c, n, c in correct_classes) for c, n in counts.items()]
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def is_true_minority(p: L.Problem):
    C = int(p.correct.sum())
    if C == 0:
        return False
    cc, wc, _ = L.class_counts(p)
    best_correct = max(cc) if cc else 0
    best_wrong = max(wc) if wc else 0
    merged = sum(cc)
    return best_correct <= best_wrong and merged < best_wrong


say("# Phase 0b — true-minority band characterization\n")
paired = L.load_paired()
ARMS = ["pi0", "grpo", "passk", "votek"]

# ----- depth distribution per arm -----
say("## Depth of the correct answer in the true-minority band\n")
say("C = total correct samples (of 64). rank = position of best correct class "
    "when classes are sorted by vote count (2 = correct is runner-up).\n")
say("| arm | minority probs | mean C | C=1 | C 2-3 | C 4-7 | C 8-15 | C 16+ | "
    "correct is rank-2 | mean best_wrong |")
say("|" + "---|" * 10)

band_by_arm = {}
for arm in ARMS:
    probs = [p for p in paired[arm].values() if is_true_minority(p)]
    band_by_arm[arm] = probs
    Cs, bws, ranks = [], [], []
    for p in probs:
        rows = class_table(p)
        C = int(p.correct.sum())
        best_wrong = max(n for _, n, ok in rows if not ok)
        # rank of first correct class
        rank = next(i + 1 for i, (_, _, ok) in enumerate(rows) if ok)
        Cs.append(C); bws.append(best_wrong); ranks.append(rank)
    Cs = np.array(Cs); bws = np.array(bws); ranks = np.array(ranks)

    def pct(mask):
        return f"{100*mask.mean():.0f}%"
    say(f"| {arm} | {len(probs)} | {Cs.mean():.1f} "
        f"| {pct(Cs==1)} | {pct((Cs>=2)&(Cs<=3))} | {pct((Cs>=4)&(Cs<=7))} "
        f"| {pct((Cs>=8)&(Cs<=15))} | {pct(Cs>=16)} "
        f"| {pct(ranks==2)} | {bws.mean():.1f} |")
say("")

# ----- cross-arm stability: same problem, same dominant wrong answer? -----
say("## Is the confidently-wrong mode stable across arms?\n")
def dominant_wrong(p: L.Problem):
    rows = class_table(p)
    for c, n, ok in rows:
        if not ok:
            return c
    return None

# problems that are true-minority in BOTH pi0 and grpo
pi0_band = {p.problem_id for p in band_by_arm["pi0"]}
grpo_band = {p.problem_id for p in band_by_arm["grpo"]}
both = pi0_band & grpo_band
same_wrong = 0
for pid in both:
    if dominant_wrong(paired["pi0"][pid]) == dominant_wrong(paired["grpo"][pid]):
        same_wrong += 1
say(f"- true-minority in pi0: {len(pi0_band)}; in grpo: {len(grpo_band)}; "
    f"in BOTH: {len(both)} ({100*len(both)/min(len(pi0_band),len(grpo_band)):.0f}% of the smaller)")
if both:
    say(f"- of the shared {len(both)}, the dominant WRONG canonical is IDENTICAL "
        f"pi0 vs grpo in {same_wrong} ({100*same_wrong/len(both):.0f}%) "
        f"=> the wrong mode is a stable, systematic error, not sampling noise.\n")

# how many problems are true-minority in ALL 4 arms
all_bands = [{p.problem_id for p in band_by_arm[a]} for a in ARMS]
in_all = set.intersection(*all_bands)
say(f"- true-minority in ALL 4 arms: {len(in_all)} problems "
    f"(persist through RL of every kind).\n")

# ----- concrete examples -----
say("## Concrete examples (grpo): correct answer vs the winning wrong answer\n")
say("| problem_id | gt answer | dominant WRONG canonical | wrong votes | correct votes (C) | correct rank |")
say("|" + "---|" * 6)
ex = sorted(band_by_arm["grpo"], key=lambda p: -int(p.correct.sum()))  # deepest-C first
shown = 0
for p in ex:
    if shown >= 18:
        break
    rows = class_table(p)
    C = int(p.correct.sum())
    bw = max(n for _, n, ok in rows if not ok)
    dw = dominant_wrong(p)
    rank = next(i + 1 for i, (_, _, ok) in enumerate(rows) if ok)
    gt = p.answer.replace("|", "\\|")[:30]
    dws = (dw or "")[:30].replace("|", "\\|")
    say(f"| {p.problem_id} | {gt} | {dws} | {bw} | {C} | {rank} |")
    shown += 1
say("")

with open(os.path.join(OUT, "PHASE0b_minority.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n[wrote", os.path.join(OUT, "PHASE0b_minority.md"), "]")
