# V5M — Learned question-conditioned null answer

Frozen V5J bilinear. A 129-parameter linear null head is TRAIN-fitted
from pooled contextual question and BOS vectors. Single-question inference;
no paired question, answer label, or requested field as an input.
Decision: sigmoid(max candidate joint − learned null score) >= 0.5.
TRAIN-only per-dimension feature normalization; fixed decision threshold.
DEV exploratory, EVAL sealed.

Null-head parameters: 129; TRAIN BCE: 2.82098 → 0.00025

| Metric | V5K DEV reference | V5M DEV |
| --- | ---: | ---: |
| dev_exact | 45 | 44 |
| answerable_correct | 33 | 32 |
| unanswerable_correct | 12 | 12 |
| unsupported_answer_errors | 0 | 0 |
| false_abstentions | 3 | 4 |
| paired_both_correct | 21 | 20 |
| correct_switches | 10 | 11 |

## Interpretation

- New null head adds parameters, not a full encoder retraining.
- A high DEV score after repeated inspection does not establish generalization.
- The frozen encoder was trained on both questions; V5M inference uses only the current question.
- Do not change thresholds using DEV; inspect on a separately registered dataset after this.
