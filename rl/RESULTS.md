# DAFT Round 1 — Results

**Date:** 2026-06-12
**Setup:** π₀ = vanilla-CE SFT of Qwen2.5-1.5B-Instruct on s1K. Three GRPO arms
from π₀, identical engine/data/budget, differing only in per-sample advantage:
`grpo` (Dr.GRPO control), `pass_at_k` (LOO coverage credit, k=8), `vote_k`
(exact majority-vote pivotality credit, k=8). G=16, 400 steps, lr 1e-6, β=0,
T=1.0, scale-calibrated to equal advantage-RMS. Eval: n=64 samples/problem at
T=1.0, pooled **1991 paired problems** (MATH-500 + held-out 1500-problem MATH
test slice), paired bootstrap (10k resamples) + McNemar. Exact subset-averaged
maj@n, unbiased pass@n. Full table: `rl/eval_outputs/analysis_3arm_pooled/`.

## Headline: negative for the core hypothesis

Δ vs π₀ (pts; **bold** = 95% CI excludes 0):

| metric | grpo (control) | pass_at_k | vote_k |
|---|---|---|---|
| pass@1 | **+1.43** | **+0.58** | **+0.68** |
| pass@8 | **+1.45** | **+1.11** | **+1.02** |
| pass@32 | **+1.69** | **+1.70** | **+0.95** |
| pass@64 | **+2.16** | **+1.91** | **+1.31** |
| maj@8 | **+1.81** | **+0.93** | **+1.22** |
| maj@16 | **+1.90** | **+1.04** | **+1.27** |
| maj@32 | **+1.88** | **+0.97** | **+1.08** |

All three arms significantly beat π₀ on every metric. **The plain GRPO control
is the best arm on every metric** — the decision-aware objectives did *not*
beat it. Hypothesis (matching the training objective to the deployed decision
rule improves the deployed metric) is **not supported in this regime**.

## Why — the regime never entered the failure mode

The premise is that standard RL sharpens the policy and *flattens* the pass@n
scaling curve, and that decision-aware training wins back that lost coverage.
At this gentle scale that collapse **never happened**: eval-time answer
diversity is essentially identical across all arms, including the control —

| run | mean correctness | distinct answers / 64 |
|---|---|---|
| π₀ | 0.317 | 20.26 |
| grpo | 0.332 | 20.62 |
| pass_at_k | 0.323 | 20.52 |
| vote_k | 0.324 | 20.64 |

With diversity preserved, the control's straightforward correctness gain lifts
*every* metric (pass@n and maj@n alike), and there is nothing for the
decision-aware shaping to win back. The shaped (sparser) advantages — ~38–41%
nonzero samples vs ~57% for grpo — simply moved the policy **less far** in 400
steps (roughly half the gain), so they are dominated.

## Faint confirmation the credit assignment works as designed

Head-to-head (`analysis_c_vs_b/`), each shaped arm is *relatively* better on
its own target metric, though below significance:
- `pass_at_k` leads `vote_k` on coverage: pass@32 +0.75pt [−0.11, +1.63].
- `vote_k` leads `pass_at_k` on the vote metric: maj@8 +0.29pt [−0.10, +0.67].

The shaping steers in the predicted directions; the effect is just too small at
this scale to overcome the control's larger overall movement.

## Takeaways → Round 2

1. **Train into the collapse regime.** The hypothesis lives where standard RL
   *hurts* coverage. 400 steps / lr 1e-6 / β=0 is too gentle — push to
   1500–2000 steps and/or lr 3e-6 until the control's pass@n curve flattens or
   drops, then re-test whether the decision-aware arms hold coverage.
2. **The control is a genuinely strong, diversity-preserving baseline at low
   compute** — itself a useful finding (cf. RLVR-collapse literature, which
   uses far more steps).
3. Infra, advantage math (unit-tested), decontamination, and the paired-eval
   harness are all validated and reusable for Round 2.
