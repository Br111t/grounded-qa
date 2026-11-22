# V5I TRAIN/DEV question-conditioning diagnostic

Read-only checkpoint, no training, no reserved EVAL access.

| Metric | TRAIN | DEV |
| --- | ---: | ---: |
| source_pairs | 24 | 24 |
| both_fields_present | 12 | 12 |
| correct_switches_among_both_present | 11 | 0 |
| mean_question_vector_l2 | 0.487159 | 0.453964 |
| mean_question_vector_cosine | 0.998200 | 0.998322 |
| mean_actual_max_field_score_delta | 18.600005 | 1.674022 |
| mean_fixed_encoder_leader_question_delta | 0.058974 | 0.067091 |
| mean_fixed_encoder_location_question_delta | 0.067675 | 0.070449 |
| mean_leader_question_margin | 6.827019 | -11.578827 |
| mean_location_question_margin | 12.247593 | 13.424295 |

## Interpret carefully

- Direct fusion sensitivity is measured by swapping the question vector
  while **holding encoder evidence states fixed**.
- Observed field-score differences additionally include changes in
  encoder hidden states when the question text changes.
- Nonzero sensitivity does not imply the correct field is selected.
- Both-field winners here exclude Topic and answerability; use the
  established V5I DEV metrics for actual final accuracy.
- Compare TRAIN versus DEV, then inspect per-source JSON for exceptions.
- A finding on this registered DEV set is exploratory, not holdout proof.
