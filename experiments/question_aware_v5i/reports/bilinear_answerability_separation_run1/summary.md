# Read-only bilinear: present vs missing field-score separation

AUC = 0.5 indicates no rank separation; 1.0 indicates perfect rank
separation with larger scores for answerable questions. AUC is descriptive,
computed independently on TRAIN and DEV; no threshold was optimized.
The requested-field label is NEVER passed to the model or feature summary.

| Feature | TRAIN AUC | DEV AUC | TRAIN present mean | TRAIN missing mean | DEV present mean | DEV missing mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answerability_logit | 0.815 | 0.762 | 2.928 | 2.915 | 2.922 | 2.911 |
| answer_probability | 0.815 | 0.762 | 0.949 | 0.949 | 0.949 | 0.948 |
| max_end | 0.924 | 0.944 | 21.378 | 5.402 | 22.038 | 6.539 |
| end_gap_top2 | 0.701 | 0.660 | 22.901 | 15.128 | 23.623 | 16.869 |
| max_joint | 0.949 | 1.000 | 40.903 | 9.713 | 28.255 | 4.909 |
| joint_gap_top2 | 0.731 | 0.748 | 42.216 | 25.625 | 32.441 | 21.188 |
| max_end_minus_mean | 0.889 | 0.896 | 16.238 | 7.564 | 16.525 | 8.435 |
| max_joint_minus_mean | 0.905 | 0.949 | 29.646 | 12.812 | 22.761 | 10.594 |

## Caveats

- Maximum end and joint scores are confidence proxies, NOT direct evidence that the requested field exists.
- Top-two gaps are zero when fewer than two candidate fields exist.
- Candidate count and field presence structure may confound group comparisons.
- Each source contributes two correlated questions; this exploratory AUC is not an independent-sample confidence interval.
- If features overlap heavily, adding them to answerability may not fix abstention.
- The reserved EVAL set was not accessed.
