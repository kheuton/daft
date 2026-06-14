# Phase 1b — PRM verifier ceiling on the selection gap

Scored 878 cached problems with Qwen2.5-Math-PRM-7B (misselected=578, control=300); these are the exact samples the band was defined on, so oracle=1.0 on the band.

## Selector accuracy

| selector | misselected band | control | all |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| majority vote | 0.020 | 1.000 | 0.355 |
| PRM best-of-n | 0.318 | 0.883 | 0.511 |
| PRM-weighted vote (p=1) | 0.289 | 0.987 | 0.527 |
| PRM-weighted vote (p=4) | 0.336 | 0.960 | 0.549 |

Band gap = oracle(1.000) - majority(0.020) = 0.980. %-of-gap recovered by the PRM:

- PRM best-of-n: band acc 0.318  ->  30.4% of gap
- PRM-weighted vote (p=1): band acc 0.289  ->  27.4% of gap
- PRM-weighted vote (p=4): band acc 0.336  ->  32.2% of gap

## PRM discrimination: correct vs wrong samples (band)

- pooled sample-level AUC: 0.735
- within-problem AUC (avg over 578 band problems): **0.690**  (Phase 1a confidence was 0.589; 0.5=useless)

## Where does the PRM help? (band stratified by correct-answer rank)

| stratum | n | majority | PRM best-of-n | PRM-wvote p=4 |
|---|---|---|---|---|
| rank-2 (near miss) | 128 | 0.053 | 0.547 | 0.586 |
| rank 3-5 | 176 | 0.000 | 0.358 | 0.386 |
| rank 6+ (deep tail) | 263 | 0.000 | 0.179 | 0.179 |

Figure: fig_phase1b.png
