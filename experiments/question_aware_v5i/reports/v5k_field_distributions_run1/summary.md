# V5K joint score and probability distributions by requested field

Read-only regrouping of already recorded TRAIN and DEV predictions.
Groups are for evaluation only: no target_field label is used in inference.
Threshold remains 0.50. No EVAL, optimization or checkpoint writes.

## TRAIN distributions

| Requested | Presence | n | Joint mean | Joint min | Joint median | Joint max | p mean | p min | p median | p max | Abstain errors | Unsupported errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leader | present | 18 | 50.681 | 47.029 | 49.089 | 55.923 | 0.9984 | 0.9972 | 0.9982 | 0.9996 | 0 | 0 |
| leader | missing | 6 | -1.365 | -12.388 | 1.873 | 3.653 | 0.0158 | 0.0009 | 0.0193 | 0.0279 | 0 | 0 |
| location | present | 18 | 31.124 | 15.809 | 34.740 | 41.118 | 0.8167 | 0.2848 | 0.9608 | 0.9901 | 4 | 0 |
| location | missing | 6 | 20.791 | 17.449 | 22.311 | 22.740 | 0.5389 | 0.3637 | 0.6192 | 0.6402 | 0 | 4 |

### Candidate-count and top-field breakdown

| Requested | Presence | Candidate counts | Top-field counts |
| --- | --- | --- | --- |
| leader | present | {'2': 6, '3': 12} | {'leader': 18} |
| leader | missing | {'2': 6} | {'location': 6} |
| location | present | {'2': 6, '3': 12} | {'location': 18} |
| location | missing | {'2': 6} | {'leader': 6} |

## DEV distributions

| Requested | Presence | n | Joint mean | Joint min | Joint median | Joint max | p mean | p min | p median | p max | Abstain errors | Unsupported errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leader | present | 18 | 24.723 | 20.576 | 24.411 | 28.144 | 0.7217 | 0.5304 | 0.7221 | 0.8516 | 0 | 0 |
| leader | missing | 6 | 0.328 | -5.273 | 0.418 | 4.353 | 0.0175 | 0.0041 | 0.0149 | 0.0325 | 0 | 0 |
| location | present | 18 | 31.788 | 13.121 | 34.495 | 40.259 | 0.8298 | 0.1833 | 0.9582 | 0.9878 | 3 | 0 |
| location | missing | 6 | 9.489 | 6.396 | 10.808 | 11.362 | 0.0983 | 0.0491 | 0.1177 | 0.1304 | 0 | 0 |

### Candidate-count and top-field breakdown

| Requested | Presence | Candidate counts | Top-field counts |
| --- | --- | --- | --- |
| leader | present | {'2': 6, '3': 12} | {'leader': 18} |
| leader | missing | {'2': 6} | {'location': 5, 'topic': 1} |
| location | present | {'2': 6, '3': 12} | {'location': 18} |
| location | missing | {'2': 6} | {'leader': 6} |

## DEV false abstentions, same registered 0.50 threshold

| Example | Requested | Correct top field? | Candidate count | Joint score | Original p | Calibrated p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| v5g_dev_16_canonical_location | location | True | 2 | 13.121 | 0.94883 | 0.18331 |
| v5g_dev_07_canonical_location | location | True | 3 | 14.842 | 0.94909 | 0.24679 |
| v5g_dev_15_canonical_location | location | True | 3 | 16.952 | 0.94860 | 0.33888 |

## Reading the results

- Compare present/missing *within each requested field*, not only pooled means.
- Check whether Location-present scores are systematically below Leader-present.
- Check candidate-count distributions, since missing-target sources have fewer fields.
- The pooled joint score may separate source structure, not semantic answerability.
- Full per-group rows plus p25/p75 are retained in metrics.json.
- DEV has been inspected repeatedly; no DEV-fitted threshold or new performance claim.
