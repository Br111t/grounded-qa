# V5I activation comparison: tanh vs GELU vs linear

Matched architecture, random initialization seed, TRAIN/DEV, schedule, optimizer,
objective, decoder and evaluation cadence. Only activation changes.
One fixed seed; results are exploratory, not a multiple-seed comparison.
No reserved EVAL accessed. New checkpoints are independent of original V5I.

| Variant | DEV exact | Paired both-correct | Correct switches | Unsupported | False abstentions | TRAIN exact | Optimization s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 35/48 | 9.744 |
| gelu | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 24/48 | 7.044 |
| linear | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 29/48 | 6.835 |

## Interpretation

- The linear fusion still has question-dependent encoder hidden states; only its direct
  additive pooled-question term cancels in within-encoder token rankings.
- Relative scores and paired-switch metrics take precedence over raw logit magnitude.
- Record Brier/answerability, losses and field confusion in per-variant metrics.json.
- DEV is exploratory; do not select an EVAL winner without preregistration.
- Compare runtimes only with similar machine load. One seed is not statistical evidence.
