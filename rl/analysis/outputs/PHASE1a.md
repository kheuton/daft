# Phase 1a — can the model's own confidence select the answer?

Re-decoded 878 problems with token logprobs (misselected=578, control=300), grpo, n=64, T=1.0.

## Fresh-decode reproduction (mis-selected band)

- of 578 band problems, 491 (85%) still have >=1 correct sample (coverage reproduces).

## Selector accuracy (single decision over 64 fresh samples)

| selector | misselected | control | all |
|---|---|---|---|
| oracle (any correct) | 0.849 | 1.000 | 0.901 |
| majority vote | 0.142 | 0.948 | 0.417 |
| best-of-n: max mean_lp | 0.156 | 0.670 | 0.331 |
| best-of-n: max tail_lp | 0.088 | 0.620 | 0.270 |
| best-of-n: max sum_lp | 0.157 | 0.623 | 0.317 |
| conf-weighted vote (mean_lp, tau=.3) | 0.173 | 0.960 | 0.442 |
| conf-weighted vote (mean_lp, tau=.1) | 0.201 | 0.957 | 0.459 |

On the band: majority=0.142, oracle=0.849, gap=0.708. %-of-gap recovered by confidence:

| selector | band acc | %-gap recovered |
|---|---|---|
| best-of-n: max mean_lp | 0.156 | 2.0% |
| best-of-n: max tail_lp | 0.088 | -7.6% |
| best-of-n: max sum_lp | 0.157 | 2.2% |
| conf-weighted vote (mean_lp, tau=.3) | 0.173 | 4.4% |
| conf-weighted vote (mean_lp, tau=.1) | 0.201 | 8.3% |

## Does confidence separate correct from wrong samples? (band)

- **mean_lp**: mean(correct)=-0.382, mean(wrong)=-0.429, gap=+0.047, pooled sample-level AUC=0.572
- **tail_lp**: mean(correct)=-0.326, mean(wrong)=-0.449, gap=+0.123, pooled sample-level AUC=0.610
- **within-problem AUC** (mean_lp, avg over 491 band problems with both classes): 0.589  (0.5 = confidence is useless / model is confidently-wrong)

Figure: fig_phase1a.png
