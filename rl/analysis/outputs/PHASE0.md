# DAFT Phase 0 — selection-gap analysis

Paired pool: **1991 problems** present in all 4 runs (n=64 samples/problem, T=1.0). Recomputed from completions.jsonl.

## 1. Coverage vs. deployed selection

| arm | pass@1 | pass@8 | pass@32 | pass@64 | maj@8 | maj@16 | maj@32 | **gap(p64-m32)** |
|---|---|---|---|---|---|---|---|---|
| pi0 | 0.317 | 0.629 | 0.759 | 0.803 | 0.458 | 0.492 | 0.512 | **0.291** |
| grpo | 0.332 | 0.643 | 0.776 | 0.825 | 0.476 | 0.511 | 0.531 | **0.294** |
| passk | 0.323 | 0.640 | 0.776 | 0.822 | 0.468 | 0.502 | 0.522 | **0.301** |
| votek | 0.324 | 0.639 | 0.769 | 0.816 | 0.471 | 0.504 | 0.523 | **0.293** |

`gap` = correct answers that ARE generated (pass@64) but the deployed majority vote discards (maj@32). This is the selection prize.

## 2. Gap autopsy — why does the vote miss correct answers?

Each solvable-but-mis-selected problem is one of:
- **(c) fragmentation** — correct answers exist in a *majority* but are split across canonical forms (merging them wins). Fixable by canonicalization.
- **(a) tie** — merged-correct ties the top wrong bloc. Fixable by tie-break.
- **(b) true minority** — a wrong answer is genuinely more popular even after merging correct forms. Needs a real selector/verifier or policy change.

| arm | total | dead(p64=0) | solvable | won outright | **mis-selected** | (c)frag | (a)tie | (b)minority |
|---|---|---|---|---|---|---|---|---|
| pi0 | 1991 | 392 | 1599 | 1032 | **567** | 1 | 34 | 532 |
| grpo | 1991 | 349 | 1642 | 1064 | **578** | 5 | 27 | 546 |
| passk | 1991 | 354 | 1637 | 1052 | **585** | 4 | 22 | 559 |
| votek | 1991 | 366 | 1625 | 1043 | **582** | 6 | 28 | 548 |

As fraction of the pool (≈ maj-accuracy points each category is worth):

| arm | dead | (c)frag | (a)tie | (b)minority | (c)+(a) PRM-free recoverable |
|---|---|---|---|---|---|
| pi0 | 0.197 | 0.001 | 0.017 | 0.267 | **0.018** |
| grpo | 0.175 | 0.003 | 0.014 | 0.274 | **0.016** |
| passk | 0.178 | 0.002 | 0.011 | 0.281 | **0.013** |
| votek | 0.184 | 0.003 | 0.014 | 0.275 | **0.017** |

## 3. How much of the gap does a PRM-free selector recover?

All selectors use the full 64 samples, one decision/problem. `merged-correct ceiling*` is label-dependent (upper bound on correct-side canonicalization). %-recovered = (sel − plain) / (oracle − plain).

| arm | oracle@64 (coverage ceiling) | plain maj (deployed) | maj, drop-unparseable | maj, tiebreak-shortest | merged-correct ceiling* |
|---|---|---|---|---|---|
| pi0 | 0.803 | 0.526 | 0.526 | 0.529 | 0.527 |
| grpo | 0.825 | 0.540 | 0.540 | 0.541 | 0.544 |
| passk | 0.822 | 0.534 | 0.534 | 0.538 | 0.536 |
| votek | 0.816 | 0.531 | 0.531 | 0.536 | 0.534 |

**%-of-gap recovered** by each realizable PRM-free selector:

| arm | maj, drop-unparseable | maj, tiebreak-shortest | merged-correct ceiling* |
|---|---|---|---|
| pi0 | 0.0% | 1.1% | 0.5% |
| grpo | 0.0% | 0.2% | 1.2% |
| passk | 0.0% | 1.5% | 0.8% |
| votek | 0.0% | 1.9% | 1.2% |

## 4. Fair re-analysis — do shaped arms select better at equal skill?

Round 1's 'control wins everything' is confounded: the control moved correctness furthest. Pooled per-problem OLS: maj@32 ~ pass@1 + pass@1² + arm dummies (grpo = reference). A positive arm coefficient = higher deployed metric at *equal per-problem correctness*. Cluster-bootstrap CI over problem_ids (2000 resamples).

| term | coef | 95% CI |
|---|---|---|
| intercept | -0.0515 | [-0.0582, -0.0449] |
| pass1 | +3.3631 | [+3.3063, +3.4210] |
| pass1^2 | -2.5117 | [-2.5832, -2.4414] |
| arm[pi0] | -0.0018 | [-0.0069, +0.0035] |
| arm[passk] | -0.0035 | [-0.0085, +0.0019] |
| arm[votek] | -0.0033 | [-0.0088, +0.0020] |

Interpretation: arm[passk]/arm[votek] > 0 (CI excludes 0) would mean the shaped objective yields a more *selectable* sample distribution than plain GRPO at the same per-problem correctness — the decision-aware signal Round 1 couldn't see because the control simply moved further.

Figures: fig_curves.png, fig_autopsy.png, fig_selectors.png
