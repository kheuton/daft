# Decision-Aware Fine-Tuning (DAFT): Design v2

*v2 incorporates an adversarial 3-lens review (theory/statistics, TRL+vLLM
systems, experimental design). Review-driven changes are marked [R].*

## 1. Problem statement

We deploy a policy `π_θ` under a fixed test-time compute budget: draw `n` i.i.d.
samples `y_1..y_n ~ π_θ(·|x)` at temperature T and apply a **decision rule** `d`
to pick a final answer. Utility is `u(d(y_1..n), a*) ∈ {0,1}`. Decision rules in
our eval harness (`sal` best-of-n + processing):

| rule | definition | role |
|---|---|---|
| `oracle` (pass@n) | correct if **any** sample correct | coverage upper bound |
| `maj` | majority vote over canonicalized answers | selector-free deployment |
| `prm_bon` | argmax PRM score | current "best_of_n" metric |
| `prm_wmaj` | PRM-score-weighted vote | current best metric |

Decision-aware objective: `J_n(θ) = E_x E_{y_1..n} [u(d(y_1..n), a*)]`.
Standard SFT/GRPO optimize ≈`J_1` (mode-seeking); the gap to `J_n` is the value
of answer-distribution diversity. Our own data (qwen-1.5B-s1, MATH-500 subset):
PRM best-of-n is non-monotone (0.22@1 → 0.30@64 → 0.18@128) while PRM-weighted
vote scales 0.22 → 0.50 — the decision rule changes the value of samples, and
the noisy-selector regime means coverage alone is not the target; the deployed
rule is.

## 2. Critique of "reward if any of n generations is correct"

The objective is right — `E[max_i c_i]` *is* `J_n` for the oracle rule — but
broadcasting the group reward `R = max_i c_i` to all samples fails on credit
assignment: within a group the gradient cannot distinguish the correct trace
from garbage (every member gets the same scalar), so wrong siblings of a correct
sample are reinforced; the estimator is unbiased only across many resamples →
hopeless variance at our scale. With a leave-one-out baseline it collapses to
**pivotal credit** `A_i = c_i · Π_{j≠i}(1−c_j)`: reward only the unique solver.
That is the fix — and it generalizes: compute each sample's **marginal
contribution to the deployed decision's utility**.

## 3. Objectives (all = GRPO engine, only the per-sample advantage differs)

Group of `G=16` completions per prompt; binary correctness `c_i`; `C = Σc_i`.
**[R] All arms run with `scale_rewards=False`** (Dr.GRPO-style; TRL's per-group
std-division would destroy the cross-group magnitude structure that *is* the
shaped objectives). **[R] Cross-arm scale control**: each arm's advantages are
divided by a *static per-arm constant* calibrated once on the shared pilot
rollouts so all arms start at equal advantage RMS (equal effective LR, equal
KL-to-advantage ratio). Advantage RMS is logged every step. A 0.5×/2× lr
sensitivity check runs on one shaped arm before full seeds.

### A1 `grpo` (control)
`A_i = c_i − mean(c)` (unscaled, [R] no std division — same estimator family as
the shaped arms; canonical GRPO std-scaling is a *worse* control because it
adds per-group difficulty reweighting only to this arm).

### A2 `pass_at_k` (coverage-aware, k=8)
Unbiased pass@k estimator from the group: `R̂(C,G) = 1 − C(G−C,k)/C(G,k)`.
LOO-difference credit ([R] verified unbiased in review; closed forms):

```
A_corr(C)  = (k/(G−k)) · C(G−C,k)/C(G,k)            (i correct)
A_wrong(C) = C(G−1−C,k)/C(G−1,k) − C(G−C,k)/C(G,k)  (i wrong, ≤ 0)
```

[R] This is a LOO variant of (not identical to) Pass@k-Training (Chen et al.
2508.10751) / PKPO (Walder & Karkhanis); cite as such. Tests: brute-force subset
enumeration AND a numeric gradient-direction test on a toy categorical policy
(`E[A_i ∇logπ] ∝ ∇ pass@k`). [R] Dead zone: at G=16,k=8 all advantages ≈ 0
outside C∈[1,7]; prompt sampling for this arm is stratified toward that band
using pilot C-counts, and frac-zero-advantage-groups is logged every step.

### A3 `vote_k` (decision-marginal credit for the deployed rule, k=8)
`A_i = E_{S∋i,|S|=k}[u(maj(S))] − E_{S∌i,|S|=k}[u(maj(S))]`, subsets drawn from
the group. [R] Computed **exactly** (no Monte Carlo): `u(maj(S))` depends only
on the answer-count vector, so enumerate via hypergeometric DP over count
vectors; zero estimator variance, milliseconds per group. Hard precondition
`k ≤ G−1`. [R] Tie-break = expected utility under uniform random tie-break:
`u = 1[correct ∈ argmax set]/|argmax set|` (symmetric; equals eval's
first-occurrence rule in expectation under exchangeability).
[R] **Decision-rule fidelity**: vote grouping uses the *same* canonicalization
as eval (`sal` `memoized_canonical_form`), and unparseable/truncated answers
form a real voting bloc in both training and eval, consistently.
[R] Known dead zone: prompts where the correct answer can never win a plurality
give all-zero advantages (gradient starvation precisely on hard prompts). Pilot
logs flippability; if nonzero-advantage sample fraction < ~15%, the arm becomes
hybrid `A = A_vote + 0.25·A_passk` (decided on pilot data, before full runs).

### A4 (deferred) `prm_bon_k` — needs in-loop 7B PRM; only if A3 shows signal,
and as offline scoring of cached rollouts, not inside the synchronous step.

### B (parallel cheap track, optional) budget-conditioned / multi-trace SFT.

## 4. What the comparison isolates

Same engine, same prompts, same generation budget, same pilot-calibrated
advantage scale, same KL anchor — only the credit assignment (implied
objective) differs. [R] Compute parity is *reported* (total generated tokens
per arm), not assumed. num_iterations=1 pinned (clipping inert, estimators
exact); per-arm KL/length diagnostics noted as loss-normalization-sensitive.

## 5. Experimental design

- **Base π_0 [R]**: retrain a **vanilla-CE SFT** on s1K with the existing s1
  pipeline (`use_custom_loss=false`, no SWA; ~40 min on 4 GPUs). The legacy
  checkpoints all used top-k-CE (k=128) which truncates exactly the tail
  diversity DAFT exploits — a diversity-suppressed prior risks a false null.
  Legacy ckpt kept as a secondary base for a robustness check.
- **RL prompts**: MATH train (7500, `DigitalLearningGmbH/MATH-lighteval`),
  [R] decontaminated (exact + 8-gram normalized) against MATH-500 AND s1K
  (s1K verifiably contains 85 MATH items, ≥2 exact MATH-500 matches — report
  counts). Learnability filter: π_0 G=16 rollouts at T=1.0, keep 1 ≤ C ≤ 15;
  store per-prompt C, distinct-answer counts, vote@8 flippability → used for
  (i) arm-b stratification, (ii) advantage-scale calibration, (iii) π_0
  diversity baseline, (iv) power analysis.
- **Arms**: a/b/c above; G=16, T_train=1.0, **max_completion 2048** [R],
  KL β=1e-3 to π_0 (+100-step β=0 ablation on one shaped arm [R]),
  num_iterations=1, lr 1e-6 const, batch geometry pinned: 4 GPUs ×
  per_device_bs 4 × grad_accum 4 = 64 completions = 4 groups/step, asserted
  divisible by G [R]. ~400 steps. Seed 0 pilots first; ≥2 seeds for the final
  comparison [R].
- **Eval [R]** (power was a blocker: 500 problems → MDE ~5-8pts vs predicted
  2-4pt effects):
  - Generation: n=64 samples/problem, T=1.0 (primary; 0.8 secondary),
    max_tokens 4096, models: π_0 + 3 arms.
  - Sets: MATH-500 (primary, plus decontaminated-subset reporting) **and** a
    level-stratified 1500-problem slice of the remaining MATH test split
    (no PRM needed there → cheap power for pass@n / maj@n).
  - Metrics: unbiased pass@n (n ≤ 32 from 64 samples; pass@64 flagged as raw),
    **exact** subset-averaged maj@n (same hypergeometric enumeration as
    training), PRM-bo@n + PRM-wmaj@n on MATH-500 only (≥100 subsets/problem,
    not first-n [R: sal's `completions[:n]` is a single ordered subset).
  - Analysis: **paired** per-problem differences — paired bootstrap CIs +
    McNemar; pre-registered primary endpoints: (b)−(a) on unbiased pass@8;
    (c)−(a) on subset-averaged maj@8, pooled eval set. All else exploratory.
  - π_0 curve on every plot (RLVR literature predicts the control *flattens*
    pass@n vs base — that's the phenomenon, not a bug).
  - Power gate [R]: after pilot, simulate MDE from per-problem correct-counts;
    expand the MATH-test slice if MDE > expected effect.

## 6. Systems plan (the barrier and its fix)

Prior attempts stalled because the `s1` env (trl 0.12) predates GRPOTrainer and
TRL↔vLLM integration. **[R] Pin set: `trl==0.18.x` (colocate mode landed in
0.18.0 — 0.17 is server-only; verified against source), `vllm==0.8.5.post1`,
`transformers==4.51.3`, `torch==2.6.0`** in new env `daft_rl`.

- Launch: `accelerate launch --num_processes=4` with **DDP** (1.5B needs no
  sharding; do NOT reuse the torchrun+FSDP harness — NCCL/process-group
  conflicts with colocated vLLM) [R].
- vLLM colocate: TP=1 per rank (4 replicas), `gpu_memory_utilization≈0.35`.
- [R] **No clean hook exists**: advantages are inline in
  `_generate_and_score_completions`. We subclass and override that method —
  copied from the pinned version, advantage block replaced (gather correctness
  + canonical answers via `gather_object`, compute `rl/advantages.py`, no
  normalization). A drift test asserts the upstream method's source hash so a
  TRL bump can't silently bypass us; an end-to-end test feeds a synthetic batch
  and asserts the loss consumes exactly our advantages (incl. all-correct group
  → all-zero) [R].
- [R] Grader: pebble process-pool timeouts (signal.SIGALRM does not fire off
  main thread inside the trainer); canonical-form cache; truncated completions
  counted incorrect consistently in reward AND group statistics
  (`mask_truncated_completions=False`); truncation-rate tripwire >30%.
- [R] Entropy tripwire: callback checkpoints+aborts if mean token entropy drops
  below 0.5× its step-10 value for 20 consecutive steps. Entropy lever
  pre-decided: raise `epsilon_high` (clip-higher) before touching β.
- Checkpointing: save_steps=50, save_total_limit=2, auto-resume in sbatch;
  final HF-format dir consumed directly by local eval (no Hub round-trip) [R].
- SLURM [R]: real nodes only (`pax*` — the configs' `cc1gpu/s1cmp` nodelists
  don't exist on this cluster; A100s here are 80GB, A6000s 48GB). Pilots/smoke:
  partition `hugheslab` (8× A6000). Full runs: `gpu` with
  `--constraint=a100-80G`, overflow `preempt` with requeue+resume.
- [R] **One-step timing smoke test on the target node before any multi-hour
  job**; SLURM `--time` set from measured step time × steps × 1.5.

## 7. Module layout

```
rl/
  DESIGN.md            this file
  advantages.py        grpo / pass_at_k (closed form) / vote_k (exact DP)
                       + brute-force reference impls (tests only)
  rewards.py           correctness + canonicalization (shared with eval),
                       pebble-timeout grading, answer side-channel
  train_grpo.py        GRPOTrainer subclass (pinned-version method override),
                       geometry/scale assertions, diagnostics, tripwires
  prepare_data.py      MATH train load, decontamination vs MATH-500+s1K,
                       filter-sweep consumption → train JSONL + stats JSON
  filter_sweep.py      π_0 rollouts on MATH train (vLLM offline), C counts,
                       diversity, flippability; also yields pilot calibration
  calibrate_scale.py   per-arm static advantage-scale constants from sweep
  quick_eval.py        n=64 generation + pass@n / exact maj@n curves (no PRM)
  analyze.py           paired bootstrap/McNemar, scaling plots, power calc
  configs/             per-arm YAML (geometry, k, scale constants, seeds)
  slurm/               sbatch templates: filter-sweep, train, quick-eval
  tests/               advantage math, fidelity (canonicalization parity with
                       eval), trainer-hook end-to-end, grader timeout
  env/                 SETUP.md + requirements lock
```

## 8. Risks (updated [R])

| risk | mitigation |
|---|---|
| TRL internals drift breaks override | source-hash drift test; exact pin |
| advantage renormalized accidentally | end-to-end loss-consumes-our-A test |
| vote_k gradient starvation | pilot signal-density gate → hybrid +0.25·pass@k |
| pass_at_k dead zone C∉[1,7] | stratified sampling; zero-adv fraction logged |
| entropy collapse | tripwire + clip-higher lever; β=0 ablation |
| underpowered eval | pooled 2000-problem eval; paired tests; power gate |
| contamination (s1K↔MATH-500 verified) | decontaminate RL set; report subset |
| cu124 vs driver | node smoke test first; fallback = cu121 same-logic stack |
| sympy grader hang/thread | pebble pool; cache; unit test off main thread |
| step-time blowout | 2048 cap; 1-step timing gate; --time from measurement |
```
