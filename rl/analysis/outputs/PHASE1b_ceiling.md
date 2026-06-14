# Phase 1b ceiling — best realizable PRM decision rule (no retraining)

band=578 control=300; oracle=1.0 on band, plain majority=0.020.

| rule | band | control | pool maj | band %-gap | pool Δ vs 0.540 |
|---|---|---|---|---|---|
| plain majority | 0.020 | 1.000 | 0.540 | 0.0% | +0.000 |
| PRM best-of-n (all) | 0.318 | 0.883 | 0.564 | 30.4% | +0.024 |
| PRM best-of-n (drop-empty) | 0.341 | 0.910 | 0.585 | 32.7% | +0.045 |
| PRM-wvote p=1 | 0.294 | 0.990 | 0.614 | 28.0% | +0.074 |
| PRM-wvote p=2 | 0.356 | 0.970 | 0.622 | 34.3% | +0.082 |
| PRM-wvote p=4 | 0.355 | 0.960 | 0.616 | 34.2% | +0.076 |
| PRM-wvote p=8 | 0.343 | 0.950 | 0.607 | 32.9% | +0.067 |
| thresh-maj tau=0.5 | 0.233 | 0.973 | 0.587 | 21.7% | +0.047 |
| thresh-maj tau=0.7 | 0.183 | 0.969 | 0.571 | 16.6% | +0.031 |
| thresh-maj tau=0.9 | 0.112 | 0.959 | 0.545 | 9.4% | +0.005 |
| maj + PRM tiebreak | 0.036 | 1.000 | 0.545 | 1.7% | +0.005 |

- **best pool maj overall**: PRM-wvote p=2 -> 0.622 (+0.082 vs majority)
- **best 'safe' rule (control>=0.97)**: PRM-wvote p=2 -> pool 0.622 (+0.082), band 0.356 (34.3% of gap)

Note: this is the ceiling for the EXISTING PRM with 'last' aggregation. min/prod aggregation and a held-out judge are separate (need re-score / new model).
