# V5K frozen bilinear + joint-score a′

Frozen V5J encoder and heads; only lambda and bias_delta learned using TRAIN BCE.
Joint-score feature is computed without gold requested-field information.
Original 0.5 answerability threshold. DEV exploratory. Reserved EVAL untouched.

TRAIN-only feature mean=33.105030; std=17.844799
lambda=3.868728; bias_delta=-0.081653

| Metric | Baseline DEV | a′ DEV |
| --- | ---: | ---: |
| dev_exact | 36 | 45 |
| answerable_correct | 36 | 33 |
| unanswerable_correct | 0 | 12 |
| unsupported_answer_errors | 12 | 0 |
| false_abstentions | 0 | 3 |
| paired_both_correct | 12 | 21 |
| correct_switches | 12 | 10 |
| brier_score | 0.22682206663945473 | 0.07273007236644556 |

## Success interpretation

- Aim: improve missing-target abstention while preserving answerable extraction and 12/12 field switches.
- Unsupported-answer errors count mistakes; lower is better.
- This is a TRAIN-fitted post-hoc calibration of a frozen model, not end-to-end training.
- The diagnostic examined DEV before this experiment; these findings remain exploratory.
- Scores may exploit syntactic differences in field availability; test a separately registered distribution later.
