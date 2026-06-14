"""
run_phase0.py — Run the Phase-0 selection-gap analysis and emit tables + figures.

Outputs (under rl/analysis/outputs/):
  PHASE0.md                  human-readable report
  coverage.csv               pass@k / maj@k / gap per arm
  autopsy.csv                gap decomposition per arm
  selectors.csv              selector accuracy + %-gap-recovered per arm
  fig_curves.png             pass@k & maj@k scaling per arm
  fig_autopsy.png            gap decomposition stacked bar
  fig_selectors.png          selector ceiling per arm

Run from repo root:
  python -m rl.analysis.run_phase0
"""

from __future__ import annotations

import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.analysis import phase0_lib as L

OUT = os.path.join(L.REPO, "rl", "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)

ARMS = ["pi0", "grpo", "passk", "votek"]
report_lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    report_lines.append(s)


# ==========================================================================
# Load
# ==========================================================================
say("# DAFT Phase 0 — selection-gap analysis\n")
paired = L.load_paired()
N = len(next(iter(paired.values())))
say(f"Paired pool: **{N} problems** present in all {len(ARMS)} runs "
    f"(n=64 samples/problem, T=1.0). Recomputed from completions.jsonl.\n")
problems = {arm: list(paired[arm].values()) for arm in ARMS}


# ==========================================================================
# 1. Coverage / vote table (authoritative recompute)
# ==========================================================================
say("## 1. Coverage vs. deployed selection\n")
cov = {}
for arm in ARMS:
    ps = problems[arm]
    pk = L.passk_curve(ps)
    mk = L.majk_curve(ps)
    cov[arm] = {"pass": pk, "maj": mk}

hdr = "| arm | pass@1 | pass@8 | pass@32 | pass@64 | maj@8 | maj@16 | maj@32 | **gap(p64-m32)** |"
say(hdr)
say("|" + "---|" * 9)
for arm in ARMS:
    pk, mk = cov[arm]["pass"], cov[arm]["maj"]
    gap = pk[64] - mk[32]
    say(f"| {arm} | {pk[1]:.3f} | {pk[8]:.3f} | {pk[32]:.3f} | {pk[64]:.3f} "
        f"| {mk[8]:.3f} | {mk[16]:.3f} | {mk[32]:.3f} | **{gap:.3f}** |")
say("")
say("`gap` = correct answers that ARE generated (pass@64) but the deployed "
    "majority vote discards (maj@32). This is the selection prize.\n")

with open(os.path.join(OUT, "coverage.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "pass@1", "pass@8", "pass@32", "pass@64",
                "maj@8", "maj@16", "maj@32", "gap_p64_m32"])
    for arm in ARMS:
        pk, mk = cov[arm]["pass"], cov[arm]["maj"]
        w.writerow([arm, pk[1], pk[8], pk[32], pk[64], mk[8], mk[16], mk[32],
                    pk[64] - mk[32]])


# ==========================================================================
# 2. Gap autopsy
# ==========================================================================
say("## 2. Gap autopsy — why does the vote miss correct answers?\n")
say("Each solvable-but-mis-selected problem is one of:")
say("- **(c) fragmentation** — correct answers exist in a *majority* but are "
    "split across canonical forms (merging them wins). Fixable by canonicalization.")
say("- **(a) tie** — merged-correct ties the top wrong bloc. Fixable by tie-break.")
say("- **(b) true minority** — a wrong answer is genuinely more popular even "
    "after merging correct forms. Needs a real selector/verifier or policy change.\n")

aut = {arm: L.autopsy(problems[arm]) for arm in ARMS}
say("| arm | total | dead(p64=0) | solvable | won outright | **mis-selected** "
    "| (c)frag | (a)tie | (b)minority |")
say("|" + "---|" * 9)
for arm in ARMS:
    a = aut[arm]
    say(f"| {arm} | {a.n_total} | {a.n_dead} | {a.n_solvable} | {a.n_won} "
        f"| **{a.n_misselected}** | {a.n_frag} | {a.n_tie} | {a.n_minority} |")
say("")
# % of the pool each category represents (= max maj-accuracy points recoverable)
say("As fraction of the pool (≈ maj-accuracy points each category is worth):\n")
say("| arm | dead | (c)frag | (a)tie | (b)minority | (c)+(a) PRM-free recoverable |")
say("|" + "---|" * 6)
for arm in ARMS:
    a = aut[arm]
    say(f"| {arm} | {a.n_dead/a.n_total:.3f} | {a.n_frag/a.n_total:.3f} "
        f"| {a.n_tie/a.n_total:.3f} | {a.n_minority/a.n_total:.3f} "
        f"| **{(a.n_frag+a.n_tie)/a.n_total:.3f}** |")
say("")

with open(os.path.join(OUT, "autopsy.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "total", "dead", "solvable", "won", "misselected",
                "frag_c", "tie_a", "minority_b"])
    for arm in ARMS:
        a = aut[arm]
        w.writerow([arm, a.n_total, a.n_dead, a.n_solvable, a.n_won,
                    a.n_misselected, a.n_frag, a.n_tie, a.n_minority])


# ==========================================================================
# 3. PRM-free selector ceiling
# ==========================================================================
say("## 3. How much of the gap does a PRM-free selector recover?\n")
say("All selectors use the full 64 samples, one decision/problem. "
    "`merged-correct ceiling*` is label-dependent (upper bound on correct-side "
    "canonicalization). %-recovered = (sel − plain) / (oracle − plain).\n")

sel = {arm: L.selector_accuracy(problems[arm]) for arm in ARMS}
names = list(L.SELECTORS.keys())
say("| arm | " + " | ".join(names) + " |")
say("|" + "---|" * (len(names) + 1))
for arm in ARMS:
    row = " | ".join(f"{sel[arm][n]:.3f}" for n in names)
    say(f"| {arm} | {row} |")
say("")

say("**%-of-gap recovered** by each realizable PRM-free selector:\n")
realizable = ["maj, drop-unparseable", "maj, tiebreak-shortest", "merged-correct ceiling*"]
say("| arm | " + " | ".join(realizable) + " |")
say("|" + "---|" * (len(realizable) + 1))
for arm in ARMS:
    plain = sel[arm]["plain maj (deployed)"]
    oracle = sel[arm]["oracle@64 (coverage ceiling)"]
    denom = oracle - plain
    cells = []
    for n in realizable:
        rec = (sel[arm][n] - plain) / denom if denom > 1e-9 else float("nan")
        cells.append(f"{rec*100:.1f}%")
    say(f"| {arm} | " + " | ".join(cells) + " |")
say("")

with open(os.path.join(OUT, "selectors.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm"] + names)
    for arm in ARMS:
        w.writerow([arm] + [sel[arm][n] for n in names])


# ==========================================================================
# 4. Fair re-analysis: arms at matched per-problem correctness
# ==========================================================================
say("## 4. Fair re-analysis — do shaped arms select better at equal skill?\n")
say("Round 1's 'control wins everything' is confounded: the control moved "
    "correctness furthest. Pooled per-problem OLS: maj@32 ~ pass@1 + pass@1² + "
    "arm dummies (grpo = reference). A positive arm coefficient = higher deployed "
    "metric at *equal per-problem correctness*. Cluster-bootstrap CI over "
    "problem_ids (2000 resamples).\n")

# Build design matrix pooled over arms (paired by problem_id).
pids = sorted(paired["grpo"].keys())
arm_index = {a: i for i, a in enumerate(ARMS)}
rows_pass1, rows_maj32, rows_arm, rows_pid = [], [], [], []
for arm in ARMS:
    for pid in pids:
        p = paired[arm][pid]
        rows_pass1.append(p.correct.mean())
        rows_maj32.append(L.maj_at_k(p, 32))
        rows_arm.append(arm)
        rows_pid.append(pid)
pass1 = np.array(rows_pass1)
maj32 = np.array(rows_maj32)
arm_arr = np.array(rows_arm)
pid_arr = np.array(rows_pid)

dummy_arms = ["pi0", "passk", "votek"]  # grpo reference


def design(mask_pass1, mask_arm):
    X = [np.ones_like(mask_pass1), mask_pass1, mask_pass1 ** 2]
    for a in dummy_arms:
        X.append((mask_arm == a).astype(float))
    return np.column_stack(X)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


X_full = design(pass1, arm_arr)
beta_full = ols(maj32, X_full)
coef_names = ["intercept", "pass1", "pass1^2"] + [f"arm[{a}]" for a in dummy_arms]

# cluster bootstrap by problem_id
rng = np.random.default_rng(0)
uniq_pids = np.array(pids)
# pre-index rows by pid for fast resampling
idx_by_pid = {pid: np.where(pid_arr == pid)[0] for pid in uniq_pids}
B = 2000
boot = np.zeros((B, len(beta_full)))
for b in range(B):
    samp = rng.choice(uniq_pids, size=len(uniq_pids), replace=True)
    sel_idx = np.concatenate([idx_by_pid[pid] for pid in samp])
    boot[b] = ols(maj32[sel_idx], design(pass1[sel_idx], arm_arr[sel_idx]))
lo = np.percentile(boot, 2.5, axis=0)
hi = np.percentile(boot, 97.5, axis=0)

say("| term | coef | 95% CI |")
say("|---|---|---|")
for i, nm in enumerate(coef_names):
    star = " **" if (lo[i] > 0 or hi[i] < 0) and nm.startswith("arm") else ""
    say(f"| {nm} | {beta_full[i]:+.4f}{star} | [{lo[i]:+.4f}, {hi[i]:+.4f}]{star.strip()} |")
say("")
say("Interpretation: arm[passk]/arm[votek] > 0 (CI excludes 0) would mean the "
    "shaped objective yields a more *selectable* sample distribution than plain "
    "GRPO at the same per-problem correctness — the decision-aware signal Round 1 "
    "couldn't see because the control simply moved further.\n")


# ==========================================================================
# Figures
# ==========================================================================
# Fig 1: pass@k & maj@k curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ks_p = [1, 2, 4, 8, 16, 32, 64]
ks_m = [1, 2, 4, 8, 16, 32]
for arm in ARMS:
    ax[0].plot(ks_p, [cov[arm]["pass"][k] for k in ks_p], marker="o", label=arm)
    ax[1].plot(ks_m, [cov[arm]["maj"][k] for k in ks_m], marker="o", label=arm)
for a, t in zip(ax, ["pass@k (coverage)", "maj@k (deployed)"]):
    a.set_xscale("log", base=2); a.set_xlabel("k"); a.set_title(t)
    a.grid(alpha=0.3); a.legend()
ax[0].set_ylabel("accuracy")
fig.suptitle("Coverage rises with k; the deployed vote lags far below the ceiling")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_curves.png"), dpi=130)
plt.close(fig)

# Fig 2: gap decomposition stacked bar (fraction of pool)
fig, axx = plt.subplots(figsize=(7.5, 4.5))
cats = ["won", "(c)frag", "(a)tie", "(b)minority", "dead"]
colors = ["#4c9f70", "#f2c14e", "#f78154", "#b4436c", "#9aa0a6"]
bottoms = np.zeros(len(ARMS))
for cat, col in zip(cats, colors):
    vals = []
    for arm in ARMS:
        a = aut[arm]
        m = {"won": a.n_won, "(c)frag": a.n_frag, "(a)tie": a.n_tie,
             "(b)minority": a.n_minority, "dead": a.n_dead}
        vals.append(m[cat] / a.n_total)
    axx.bar(ARMS, vals, bottom=bottoms, label=cat, color=col)
    bottoms += np.array(vals)
axx.set_ylabel("fraction of pool")
axx.set_title("Where the maj-vote accuracy goes\n(c)+(a) = canonicalization/tie = PRM-free recoverable")
axx.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_autopsy.png"), dpi=130)
plt.close(fig)

# Fig 3: selector ceiling
fig, axx = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(len(ARMS))
width = 0.16
for i, nm in enumerate(names):
    axx.bar(x + (i - 2) * width, [sel[arm][nm] for arm in ARMS], width, label=nm)
axx.set_xticks(x); axx.set_xticklabels(ARMS); axx.set_ylabel("accuracy")
axx.set_title("Selector ceiling: oracle vs deployed vs PRM-free fixes")
axx.legend(fontsize=7.5, ncol=2)
axx.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_selectors.png"), dpi=130)
plt.close(fig)

say("Figures: fig_curves.png, fig_autopsy.png, fig_selectors.png")

with open(os.path.join(OUT, "PHASE0.md"), "w") as fh:
    fh.write("\n".join(report_lines) + "\n")
print("\n[wrote outputs to", OUT, "]")
