# V5D logit and decoder diagnostic

V5D DEV SHA256: `5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd`

## Observations

| Model | Gold end ranked #1 | Gold end ranked top 5 | Raw wrong, joint correct | Raw correct, joint wrong |
|---|---:|---:|---:|---:|
| V5 | 34/96 | 40/96 | 0 | 27 |
| V5B | 38/96 | 54/96 | 0 | 6 |
| V5D | 22/96 | 79/96 | 0 | 12 |

## Representative predictions (deterministic selection)

Chosen by group and error category, sorted by example ID; these are not random samples. See `all_cases.json` for all 288 observations.

| Model | Group | Category | Expected | Predicted | Gold end rank |
|---|---|---|---|---|---:|
| V5 | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbaa. | Leader: baa. | 1 |
| V5 | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbaa. | Leader: baa. | 1 |
| V5 | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbbi. | Leader: bbi. | 1 |
| V5 | length4_vowel | exact | Leader: Dbca. | Leader: Dbca. | 1 |
| V5 | length4_vowel | exact | Leader: Dbea. | Leader: Dbea. | 1 |
| V5 | length4_vowel | exact | Leader: Dbfi. | Leader: Dbfi. | 1 |
| V5 | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: ba. | 21 |
| V5 | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: ba. | 21 |
| V5 | length4_consonant | raw_and_selected_wrong | Leader: Dbbt. | Leader: bbt. Loca. | 29 |
| V5 | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbcea. | Leader: bcea. | 1 |
| V5 | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbdei. | Leader: bdei. | 1 |
| V5 | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbeea. | Leader: beea. | 1 |
| V5 | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: ba. | 5 |
| V5 | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: ba. | 4 |
| V5 | length5_vowel | raw_and_selected_wrong | Leader: Dbbei. | Leader: bbei. Loca. | 26 |
| V5 | length5_vowel | exact | Leader: Dbgea. | Leader: Dbgea. | 1 |
| V5 | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: ba. | 22 |
| V5 | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: ba. | 22 |
| V5 | length5_consonant | raw_and_selected_wrong | Leader: Dbbet. | Leader: bbet. Loca. | 26 |
| V5B | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbhi. | Leader: bhi. | 1 |
| V5B | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbka. | Leader: bka. | 1 |
| V5B | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbka. | Leader: bka. | 1 |
| V5B | length4_vowel | exact | Leader: Dbaa. | Leader: Dbaa. | 1 |
| V5B | length4_vowel | exact | Leader: Dbaa. | Leader: Dbaa. | 1 |
| V5B | length4_vowel | exact | Leader: Dbbi. | Leader: Dbbi. | 1 |
| V5B | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: Dba. | 5 |
| V5B | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: Dba. | 4 |
| V5B | length4_consonant | raw_and_selected_wrong | Leader: Dbbt. | Leader: Db. | 33 |
| V5B | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbhei. | Leader: bhei. | 1 |
| V5B | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbkea. | Leader: bkea. | 1 |
| V5B | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbkea. | Leader: bkea. | 1 |
| V5B | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: Dba. | 3 |
| V5B | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: Dba. | 3 |
| V5B | length5_vowel | raw_and_selected_wrong | Leader: Dbbei. | Leader: Db. | 5 |
| V5B | length5_vowel | exact | Leader: Dbcea. | Leader: Dbcea. | 1 |
| V5B | length5_vowel | exact | Leader: Dbdei. | Leader: Dbdei. | 1 |
| V5B | length5_vowel | exact | Leader: Dbeea. | Leader: Dbeea. | 1 |
| V5B | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: Dba. | 18 |
| V5B | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: Dba. | 17 |
| V5B | length5_consonant | raw_and_selected_wrong | Leader: Dbbet. | Leader: Db. | 21 |
| V5D | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbca. | Leader: Pytho. | 1 |
| V5D | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbdi. | Leader: di. | 1 |
| V5D | length4_vowel | decoder_disagrees_correct_raw_end | Leader: Dbea. | Leader: Pytho. | 1 |
| V5D | length4_vowel | raw_and_selected_wrong | Leader: Dbbi. | Leader: Roo. | 2 |
| V5D | length4_vowel | raw_and_selected_wrong | Leader: Dbca. | Leader: Dbca. Lo. | 3 |
| V5D | length4_vowel | raw_and_selected_wrong | Leader: Dbdi. | Leader: Roo. | 2 |
| V5D | length4_vowel | exact | Leader: Dbaa. | Leader: Dbaa. | 1 |
| V5D | length4_vowel | exact | Leader: Dbaa. | Leader: Dbaa. | 1 |
| V5D | length4_vowel | exact | Leader: Dbbi. | Leader: Dbbi. | 1 |
| V5D | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: Dba. | 4 |
| V5D | length4_consonant | raw_and_selected_wrong | Leader: Dban. | Leader: Dba. | 4 |
| V5D | length4_consonant | raw_and_selected_wrong | Leader: Dbbt. | Leader: Roo. | 2 |
| V5D | length4_consonant | exact | Leader: Dbbt. | Leader: Dbbt. | 2 |
| V5D | length4_consonant | exact | Leader: Dbgn. | Leader: Dbgn. | 2 |
| V5D | length4_consonant | exact | Leader: Dblt. | Leader: Dblt. | 2 |
| V5D | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbdei. | Leader: dei. | 1 |
| V5D | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbhei. | Leader: der: Dbhei. | 1 |
| V5D | length5_vowel | decoder_disagrees_correct_raw_end | Leader: Dbjei. | Leader: Topic: Ast. | 1 |
| V5D | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: Dba. | 3 |
| V5D | length5_vowel | raw_and_selected_wrong | Leader: Dbaea. | Leader: Dba. | 3 |
| V5D | length5_vowel | raw_and_selected_wrong | Leader: Dbcea. | Leader: Lo. | 5 |
| V5D | length5_vowel | abstention | Leader: Dbbei. | Insufficient information. | 14 |
| V5D | length5_vowel | abstention | Leader: Dbiea. | Insufficient information. | 5 |
| V5D | length5_vowel | exact | Leader: Dbbei. | Leader: Dbbei. | 2 |
| V5D | length5_vowel | exact | Leader: Dbgea. | Leader: Dbgea. | 2 |
| V5D | length5_vowel | exact | Leader: Dbhei. | Leader: Dbhei. | 1 |
| V5D | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: Dba. | 11 |
| V5D | length5_consonant | raw_and_selected_wrong | Leader: Dbaen. | Leader: Dba. | 11 |
| V5D | length5_consonant | raw_and_selected_wrong | Leader: Dbcen. | Leader: Lo. | 18 |
| V5D | length5_consonant | abstention | Leader: Dbbet. | Insufficient information. | 11 |
| V5D | length5_consonant | abstention | Leader: Dbien. | Insufficient information. | 16 |
| V5D | length5_consonant | exact | Leader: Dbbet. | Leader: Dbbet. | 2 |
| V5D | length5_consonant | exact | Leader: Dblet. | Leader: Dblet. | 2 |
| V5D | length5_consonant | exact | Leader: Dblet. | Leader: Dblet. | 2 |

## Interpretation (hypotheses, not conclusions)

- If the correct end often ranks #2–5, investigate scoring calibration and joint decoding before changing the encoder.
- If the correct end consistently ranks far below the top candidates, investigate representation learning or the end-supervision objective.
- If the raw end ranks #1 but the final span is wrong, inspect start scores and joint-score competition; do not blame the end head alone.
- Compare the same examples across checkpoints; no causal claims about gradient conflict or data balance follow from this diagnostic.

No training, checkpoint updates, threshold changes, or final holdout access.
