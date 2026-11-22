# V5I fusion and tanh diagnostic

Read-only. TRAIN/DEV only. No training or reserved EVAL.

## Learned fusion weights

- Token block Frobenius norm: 3.899782
- Question block Frobenius norm: 3.528286

## Evidence-token activation diagnostics

| Metric (mean across examples) | TRAIN | DEV |
| --- | ---: | ---: |
| token_term_rms | 1.052483 | 1.053718 |
| question_term_rms | 0.760386 | 0.763387 |
| preactivation_mean_abs | 1.237735 | 1.240870 |
| preactivation_max_abs | 3.032812 | 3.028965 |
| tanh_activation_mean_abs | 0.732738 | 0.733283 |
| tanh_derivative_mean | 0.385449 | 0.384395 |
| fraction_abs_z_ge_2 | 0.118969 | 0.120965 |
| fraction_abs_z_ge_3 | 0.001734 | 0.001779 |
| fraction_derivative_below_0_1 | 0.216937 | 0.219678 |
| fraction_derivative_below_0_01 | 0.001934 | 0.002001 |
| question_to_token_rms_ratio | 0.722529 | 0.724477 |
| question_effect_after_tanh_rms | 0.414100 | 0.414521 |
| start_logit_question_effect_mean_abs | 0.561645 | 0.564811 |
| end_logit_question_effect_mean_abs | 1.104695 | 1.108511 |

## Interpretation guide

- High fraction `abs(z) >= 2` or derivative below 0.1 suggests tanh saturation.
- Low question/token RMS ratio suggests small question input relative to token input.
- Small post-tanh question effect despite a sizable question term suggests compression, possibly saturation.
- Question effects on logits do not necessarily imply a change in the selected field.
- This diagnostic does not establish the causal origin of generalization failure.
