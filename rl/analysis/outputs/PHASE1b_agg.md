# Phase 1b — PRM ceiling with proper per-step scoring

band=578 control=300. Steps/completion now: mean=45.6, median=43, max=495 (was 1 before the split fix).

## Aggregation sweep (rule = PRM-weighted vote p=2, drop-empty)

| aggregation | band | control | pool maj | band %-gap | AUC | pool Δ |
|---|---|---|---|---|---|---|
| last | 0.211 | 0.997 | 0.594 | 19.5% | 0.711 | +0.054 |
| min | 0.242 | 0.957 | 0.582 | 22.7% | 0.569 | +0.042 |
| prod | 0.279 | 0.940 | 0.583 | 26.4% | 0.660 | +0.043 |
| mean | 0.081 | 1.000 | 0.558 | 6.3% | 0.607 | +0.018 |

**best aggregation: last** (pool 0.594). Rule sweep under it:

| rule (last agg) | band | control | pool maj | band %-gap | pool Δ |
|---|---|---|---|---|---|
| best-of-n (drop-empty) | 0.337 | 0.873 | 0.565 | 32.4% | +0.025 |
| wvote p=1 | 0.163 | 1.000 | 0.582 | 14.6% | +0.042 |
| wvote p=2 | 0.211 | 0.997 | 0.594 | 19.5% | +0.054 |
| wvote p=4 | 0.270 | 0.997 | 0.611 | 25.5% | +0.071 |

Compare to whole-sequence PRM (1-step, prior): wvote p=2 -> band 0.356, pool 0.622 (+8.2pt), AUC 0.690.
