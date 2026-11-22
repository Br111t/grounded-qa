# Complete-field decoder · DEV comparison

No training. Same model weights and inputs in both arms.
A change can still be unchanged if both answers are wrong.

| Model | Baseline exact | Field exact | Improved | Regressed | Unchanged | Changed predictions |
|---|---:|---:|---:|---:|---:|---:|
| V5 | 7/96 | 88/96 | 81 | 0 | 15 | 89 |
| V5B | 32/96 | 96/96 | 64 | 0 | 32 | 64 |
| V5D | 21/96 | 78/96 | 57 | 0 | 39 | 71 |

## By group

| Model | Group | Baseline exact | Field exact | Improved | Regressed | Unchanged |
|---|---|---:|---:|---:|---:|---:|
| V5 | length4_consonant | 0/24 | 21/24 | 21 | 0 | 3 |
| V5 | length4_vowel | 6/24 | 24/24 | 18 | 0 | 6 |
| V5 | length5_consonant | 0/24 | 21/24 | 21 | 0 | 3 |
| V5 | length5_vowel | 1/24 | 22/24 | 21 | 0 | 3 |
| V5B | length4_consonant | 0/24 | 24/24 | 24 | 0 | 0 |
| V5B | length4_vowel | 21/24 | 24/24 | 3 | 0 | 21 |
| V5B | length5_consonant | 0/24 | 24/24 | 24 | 0 | 0 |
| V5B | length5_vowel | 11/24 | 24/24 | 13 | 0 | 11 |
| V5D | length4_consonant | 4/24 | 21/24 | 17 | 0 | 7 |
| V5D | length4_vowel | 9/24 | 21/24 | 12 | 0 | 12 |
| V5D | length5_consonant | 3/24 | 18/24 | 15 | 0 | 9 |
| V5D | length5_vowel | 5/24 | 18/24 | 13 | 0 | 11 |

## Selected paired outcomes (first 3 by ID in each change class)

| Model | Status | Expected | Baseline | Alternative |
|---|---|---|---|---|
| V5 | improved | Leader: Dban. | Leader: ba. | Leader: Dban. |
| V5 | improved | Leader: Dban. | Leader: ba. | Leader: Dban. |
| V5 | improved | Leader: Dbbt. | Leader: bbt. Loca. | Leader: Dbbt. |
| V5 | unchanged | Leader: Dbcn. | Leader: bcn. Loca. | Leader: Python. |
| V5 | unchanged | Leader: Dben. | Leader: ben. Loca. | Leader: Python. |
| V5 | unchanged | Leader: Dbkn. | Leader: b. | Leader: Python. |
| V5B | improved | Leader: Dban. | Leader: Dba. | Leader: Dban. |
| V5B | improved | Leader: Dban. | Leader: Dba. | Leader: Dban. |
| V5B | improved | Leader: Dbbt. | Leader: Db. | Leader: Dbbt. |
| V5B | unchanged | Leader: Dbaa. | Leader: Dbaa. | Leader: Dbaa. |
| V5B | unchanged | Leader: Dbaa. | Leader: Dbaa. | Leader: Dbaa. |
| V5B | unchanged | Leader: Dbbi. | Leader: Dbbi. | Leader: Dbbi. |
| V5D | improved | Leader: Dban. | Leader: Dba. | Leader: Dban. |
| V5D | improved | Leader: Dban. | Leader: Dba. | Leader: Dban. |
| V5D | improved | Leader: Dbbt. | Leader: Roo. | Leader: Dbbt. |
| V5D | unchanged | Leader: Dbbt. | Leader: Dbbt. | Leader: Dbbt. |
| V5D | unchanged | Leader: Dbcn. | Leader: Pytho. | Leader: Python. |
| V5D | unchanged | Leader: Dben. | Leader: Pytho. | Leader: Python. |

## Interpretation constraints

The alternative only constrains candidate spans to complete field values. It does **not** identify the field requested by the question. It is a deliberately narrow synthetic-grammar decoder experiment.
DEV is exploratory; HOLDOUT is single-use and cannot be used to choose another decoder rule without a new untouched test set.
