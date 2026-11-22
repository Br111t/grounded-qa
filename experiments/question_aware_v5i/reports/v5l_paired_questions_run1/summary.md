# V5L — Paired-question comparison, diagnostic only

Paired canonical Leader/Location questions for the same evidence source.
**The pairing uses both questions and is not yet a single-question inference method.**
No learned parameters, new thresholds, or EVAL access.

| Split | Source condition | n | Mean (Location − Leader) joint | Min | Median | Max | Mean p difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | both_present | 12 | -18.036 | -34.807 | -15.059 | -7.265 | -0.1720 |
| train | leader_missing | 6 | 32.102 | 28.135 | 33.527 | 34.692 | 0.7821 |
| train | location_missing | 6 | -32.547 | -33.258 | -33.191 | -31.204 | -0.4602 |
| dev | both_present | 12 | 7.645 | -8.750 | 9.370 | 19.496 | 0.1382 |
| dev | leader_missing | 6 | 32.067 | 18.394 | 34.457 | 36.133 | 0.8201 |
| dev | location_missing | 6 | -17.002 | -17.269 | -16.915 | -16.782 | -0.6916 |

## Cautions

- Difference in raw scores is subject to question/field scale shift.
- In both-present sources, neither question should abstain; score difference is not evidence of absence.
- Never infer the requested field by a gold target label in deployment.
- Pairing and scenario labels are used for retrospective diagnostics only.
- Report descriptive group separation, not an optimized DEV classification threshold.
- This experiment informs a later single-question architecture; it does not establish generalization.
