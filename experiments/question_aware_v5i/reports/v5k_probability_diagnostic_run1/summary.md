# V5K calibrated probability diagnostic

Checkpoint and calibration are read-only. Threshold 0.5 is the registered
decision rule. Other thresholds are *descriptive only* and must not be
selected using this repeatedly inspected DEV split.

## DEV misclassified at threshold 0.50

| Example | Requested | Top-scoring field | Original p | Calibrated p | Max joint |
| --- | --- | --- | ---: | ---: | ---: |
| v5g_dev_16_canonical_location | location | location | 0.94883 | 0.18331 | 13.121 |
| v5g_dev_15_canonical_location | location | location | 0.94860 | 0.33888 | 16.952 |
| v5g_dev_07_canonical_location | location | location | 0.94909 | 0.24679 | 14.842 |

DEV answerable probability range: 0.18331–0.98782
DEV missing-target probability range: 0.00414–0.13044

## Probability threshold sensitivity (descriptive only)

| Threshold | TRAIN correct | TRAIN false abstain | TRAIN unsupported | DEV correct | DEV false abstain | DEV unsupported |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 42/48 | 0 | 6 | 44/48 | 0 | 4 |
| 0.20 | 42/48 | 0 | 6 | 47/48 | 1 | 0 |
| 0.30 | 41/48 | 1 | 6 | 46/48 | 2 | 0 |
| 0.40 | 43/48 | 1 | 4 | 45/48 | 3 | 0 |
| 0.50 | 40/48 | 4 | 4 | 45/48 | 3 | 0 |
| 0.60 | 40/48 | 4 | 4 | 41/48 | 7 | 0 |
| 0.70 | 43/48 | 5 | 0 | 36/48 | 12 | 0 |
| 0.80 | 43/48 | 5 | 0 | 32/48 | 16 | 0 |
| 0.90 | 43/48 | 5 | 0 | 26/48 | 22 | 0 |

## Interpretation safeguards

- A false abstention is an answerable example with calibrated p below threshold.
- An unsupported answer is a missing-target example with p at or above threshold.
- Changing the threshold does not change candidate field rankings or extracted spans.
- Candidate counts and question/evidence structure may confound score calibration.
- Review TRAIN and DEV ranges without picking an optimal DEV threshold.
- The original V5J model and V5K calibration were not modified; sealed EVAL unused.
