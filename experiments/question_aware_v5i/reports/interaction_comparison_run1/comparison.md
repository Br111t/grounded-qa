# V5J interaction comparison: multiplicative vs bilinear vs FiLM

Matched encoder, seed, TRAIN/DEV, schedule, optimizer,
objective, decoder and evaluation cadence. Only the question-token interaction changes; parameter counts vary.
One fixed seed; results are exploratory, not a multiple-seed comparison.
No reserved EVAL accessed. New checkpoints are independent of original V5I.

| Variant | Parameters | DEV exact | Paired both-correct | Correct switches | Unsupported | False abstentions | TRAIN exact | Optimization s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multiplicative | 132931 | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 31/48 | 9.596 |
| bilinear | 132867 | 36/48 | 12/24 | 12/12 | 12/12 | 0 | 36/48 | 6.844 |
| film | 132995 | 25/48 | 1/24 | 1/12 | 12/12 | 0 | 36/48 | 6.780 |

## Interpretation

- All three methods introduce an explicit multiplicative question-token interaction.
- Compare matched dataset, decoder, schedule, and shared backbone initialization.
- Parameter counts differ; interpret gains alongside computation and complexity.
- Relative scores and paired-switch metrics take precedence over raw logit magnitude.
- Record Brier/answerability, losses and field confusion in per-variant metrics.json.
- DEV is exploratory; do not select an EVAL winner without preregistration.
- Compare runtimes only with similar machine load. One seed is not statistical evidence.
