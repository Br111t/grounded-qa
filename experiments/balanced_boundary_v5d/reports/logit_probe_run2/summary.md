# V5D logit and decoder diagnostic

V5D DEV SHA256: `5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd`

## Observations

| Model | Gold end ranked #1 | Gold end ranked top 5 | Raw wrong, joint correct | Raw correct, joint wrong |
|---|---:|---:|---:|---:|
| V5 | 34/96 | 40/96 | 0 | 27 |
| V5B | 38/96 | 54/96 | 0 | 6 |
| V5D | 22/96 | 79/96 | 11 | 12 |

## Score margins (best minus gold; logits, not probabilities)

| Model | End gap <= 1 | End gap <= 3 | Joint gap <= 1 | Joint gap <= 3 | Median end gap | Median joint gap |
|---|---:|---:|---:|---:|---:|---:|
| V5 | 34/96 | 37/96 | 8/96 | 11/96 | 5.193 | 10.5927 |
| V5B | 38/96 | 39/96 | 37/96 | 39/96 | 6.0271 | 5.4959 |
| V5D | 28/96 | 68/96 | 26/96 | 55/96 | 1.7908 | 2.386 |

## Representative predictions (deterministic selection)

Chosen by group and error category, sorted by example ID; these are not random samples. See `all_cases.json` for all 288 observations.

| Model | Group | Exact | Recovery | Expected | Predicted | Gold end rank | End gap | Joint gap |
|---|---|---|---|---|---|---:|---:|---:|
| V5 | length4_vowel | False | False | Leader: Dbaa. | Leader: baa. | 1 | 0.000 | 3.698 |
| V5 | length4_vowel | False | False | Leader: Dbaa. | Leader: baa. | 1 | 0.000 | 3.759 |
| V5 | length4_vowel | False | False | Leader: Dbbi. | Leader: bbi. | 1 | 0.000 | 10.279 |
| V5 | length4_vowel | True | False | Leader: Dbca. | Leader: Dbca. | 1 | 0.000 | 0.000 |
| V5 | length4_vowel | True | False | Leader: Dbea. | Leader: Dbea. | 1 | 0.000 | 0.000 |
| V5 | length4_vowel | True | False | Leader: Dbfi. | Leader: Dbfi. | 1 | 0.000 | 0.000 |
| V5 | length4_consonant | False | False | Leader: Dban. | Leader: ba. | 21 | 12.788 | 16.393 |
| V5 | length4_consonant | False | False | Leader: Dban. | Leader: ba. | 21 | 12.842 | 16.511 |
| V5 | length4_consonant | False | False | Leader: Dbbt. | Leader: bbt. Loca. | 29 | 6.718 | 16.876 |
| V5 | length5_vowel | False | False | Leader: Dbcea. | Leader: bcea. | 1 | 0.000 | 1.888 |
| V5 | length5_vowel | False | False | Leader: Dbdei. | Leader: bdei. | 1 | 0.000 | 9.093 |
| V5 | length5_vowel | False | False | Leader: Dbeea. | Leader: beea. | 1 | 0.000 | 1.595 |
| V5 | length5_vowel | False | False | Leader: Dbaea. | Leader: ba. | 5 | 8.842 | 12.278 |
| V5 | length5_vowel | False | False | Leader: Dbaea. | Leader: ba. | 4 | 8.884 | 12.259 |
| V5 | length5_vowel | False | False | Leader: Dbbei. | Leader: bbei. Loca. | 26 | 7.130 | 13.397 |
| V5 | length5_vowel | True | False | Leader: Dbgea. | Leader: Dbgea. | 1 | 0.000 | 0.000 |
| V5 | length5_consonant | False | False | Leader: Dbaen. | Leader: ba. | 22 | 13.086 | 16.482 |
| V5 | length5_consonant | False | False | Leader: Dbaen. | Leader: ba. | 22 | 13.103 | 16.434 |
| V5 | length5_consonant | False | False | Leader: Dbbet. | Leader: bbet. Loca. | 26 | 7.197 | 13.430 |
| V5B | length4_vowel | False | False | Leader: Dbhi. | Leader: bhi. | 1 | 0.000 | 0.731 |
| V5B | length4_vowel | False | False | Leader: Dbka. | Leader: bka. | 1 | 0.000 | 0.608 |
| V5B | length4_vowel | False | False | Leader: Dbka. | Leader: bka. | 1 | 0.000 | 0.934 |
| V5B | length4_vowel | True | False | Leader: Dbaa. | Leader: Dbaa. | 1 | 0.000 | 0.000 |
| V5B | length4_vowel | True | False | Leader: Dbaa. | Leader: Dbaa. | 1 | 0.000 | 0.000 |
| V5B | length4_vowel | True | False | Leader: Dbbi. | Leader: Dbbi. | 1 | 0.000 | -0.000 |
| V5B | length4_consonant | False | False | Leader: Dban. | Leader: Dba. | 5 | 10.733 | 10.733 |
| V5B | length4_consonant | False | False | Leader: Dban. | Leader: Dba. | 4 | 10.761 | 10.761 |
| V5B | length4_consonant | False | False | Leader: Dbbt. | Leader: Db. | 33 | 7.377 | 7.377 |
| V5B | length5_vowel | False | False | Leader: Dbhei. | Leader: bhei. | 1 | 0.000 | 1.073 |
| V5B | length5_vowel | False | False | Leader: Dbkea. | Leader: bkea. | 1 | 0.000 | 0.979 |
| V5B | length5_vowel | False | False | Leader: Dbkea. | Leader: bkea. | 1 | 0.000 | 0.785 |
| V5B | length5_vowel | False | False | Leader: Dbaea. | Leader: Dba. | 3 | 9.003 | 9.003 |
| V5B | length5_vowel | False | False | Leader: Dbaea. | Leader: Dba. | 3 | 8.982 | 8.982 |
| V5B | length5_vowel | False | False | Leader: Dbbei. | Leader: Db. | 5 | 2.983 | 2.983 |
| V5B | length5_vowel | True | False | Leader: Dbcea. | Leader: Dbcea. | 1 | 0.000 | 0.000 |
| V5B | length5_vowel | True | False | Leader: Dbdei. | Leader: Dbdei. | 1 | 0.000 | 0.000 |
| V5B | length5_vowel | True | False | Leader: Dbeea. | Leader: Dbeea. | 1 | 0.000 | 0.000 |
| V5B | length5_consonant | False | False | Leader: Dbaen. | Leader: Dba. | 18 | 12.827 | 12.827 |
| V5B | length5_consonant | False | False | Leader: Dbaen. | Leader: Dba. | 17 | 12.755 | 12.755 |
| V5B | length5_consonant | False | False | Leader: Dbbet. | Leader: Db. | 21 | 6.864 | 6.864 |
| V5D | length4_vowel | False | False | Leader: Dbca. | Leader: Pytho. | 1 | 0.000 | 2.150 |
| V5D | length4_vowel | False | False | Leader: Dbdi. | Leader: di. | 1 | 0.000 | 4.570 |
| V5D | length4_vowel | False | False | Leader: Dbea. | Leader: Pytho. | 1 | 0.000 | 2.089 |
| V5D | length4_vowel | False | False | Leader: Dbbi. | Leader: Roo. | 2 | 4.057 | 4.653 |
| V5D | length4_vowel | False | False | Leader: Dbca. | Leader: Dbca. Lo. | 3 | 2.863 | 2.863 |
| V5D | length4_vowel | False | False | Leader: Dbdi. | Leader: Roo. | 2 | 3.915 | 4.474 |
| V5D | length4_vowel | True | False | Leader: Dbaa. | Leader: Dbaa. | 1 | 0.000 | 0.000 |
| V5D | length4_vowel | True | False | Leader: Dbaa. | Leader: Dbaa. | 1 | 0.000 | 0.000 |
| V5D | length4_vowel | True | False | Leader: Dbbi. | Leader: Dbbi. | 1 | 0.000 | 0.000 |
| V5D | length4_consonant | False | False | Leader: Dban. | Leader: Dba. | 4 | 1.938 | 1.631 |
| V5D | length4_consonant | False | False | Leader: Dban. | Leader: Dba. | 4 | 1.984 | 1.641 |
| V5D | length4_consonant | False | False | Leader: Dbbt. | Leader: Roo. | 2 | 5.377 | 5.896 |
| V5D | length4_consonant | True | True | Leader: Dbbt. | Leader: Dbbt. | 2 | 0.366 | 0.000 |
| V5D | length4_consonant | True | True | Leader: Dbgn. | Leader: Dbgn. | 2 | 1.624 | 0.000 |
| V5D | length4_consonant | True | True | Leader: Dblt. | Leader: Dblt. | 2 | 0.691 | 0.000 |
| V5D | length5_vowel | False | False | Leader: Dbdei. | Leader: dei. | 1 | 0.000 | 4.957 |
| V5D | length5_vowel | False | False | Leader: Dbhei. | Leader: der: Dbhei. | 1 | 0.000 | 0.857 |
| V5D | length5_vowel | False | False | Leader: Dbjei. | Leader: Topic: Ast. | 1 | 0.000 | 0.925 |
| V5D | length5_vowel | False | False | Leader: Dbaea. | Leader: Dba. | 3 | 1.241 | 1.171 |
| V5D | length5_vowel | False | False | Leader: Dbaea. | Leader: Dba. | 3 | 1.269 | 1.163 |
| V5D | length5_vowel | False | False | Leader: Dbcea. | Leader: Lo. | 5 | 5.202 | 5.339 |
| V5D | length5_vowel | False | False | Leader: Dbbei. | Insufficient information. | 14 | 6.695 | 8.104 |
| V5D | length5_vowel | False | False | Leader: Dbiea. | Insufficient information. | 5 | 5.933 | 7.299 |
| V5D | length5_vowel | True | True | Leader: Dbbei. | Leader: Dbbei. | 2 | 1.844 | 0.000 |
| V5D | length5_vowel | True | True | Leader: Dbgea. | Leader: Dbgea. | 2 | 1.336 | 0.000 |
| V5D | length5_vowel | True | False | Leader: Dbhei. | Leader: Dbhei. | 1 | 0.000 | -0.000 |
| V5D | length5_consonant | False | False | Leader: Dbaen. | Leader: Dba. | 11 | 2.254 | 2.112 |
| V5D | length5_consonant | False | False | Leader: Dbaen. | Leader: Dba. | 11 | 2.292 | 2.111 |
| V5D | length5_consonant | False | False | Leader: Dbcen. | Leader: Lo. | 18 | 6.397 | 6.508 |
| V5D | length5_consonant | False | False | Leader: Dbbet. | Insufficient information. | 11 | 6.360 | 7.784 |
| V5D | length5_consonant | False | False | Leader: Dbien. | Insufficient information. | 16 | 7.204 | 8.568 |
| V5D | length5_consonant | True | True | Leader: Dbbet. | Leader: Dbbet. | 2 | 1.661 | 0.000 |
| V5D | length5_consonant | True | True | Leader: Dblet. | Leader: Dblet. | 2 | 1.494 | 0.000 |
| V5D | length5_consonant | True | True | Leader: Dblet. | Leader: Dblet. | 2 | 1.483 | 0.000 |
| V5D | length4_consonant | True | True | Leader: Dblt. | Leader: Dblt. | 2 | 0.688 | 0.000 |

## Interpretation (hypotheses, not conclusions)

- If the correct end often ranks #2–5, investigate scoring calibration and joint decoding before changing the encoder.
- If the correct end consistently ranks far below the top candidates, investigate representation learning or the end-supervision objective.
- If the raw end ranks #1 but the final span is wrong, inspect start scores and joint-score competition; do not blame the end head alone.
- Compare the same examples across checkpoints; no causal claims about gradient conflict or data balance follow from this diagnostic.

No training, checkpoint updates, threshold changes, or final holdout access.
