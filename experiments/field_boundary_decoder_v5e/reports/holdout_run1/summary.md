# Complete-field decoder · HOLDOUT comparison

No training. Same model weights and inputs in both arms.
A change can still be unchanged if both answers are wrong.

| Model | Baseline exact | Field exact | Improved | Regressed | Unchanged | Changed predictions |
|---|---:|---:|---:|---:|---:|---:|
| V5 | 12/108 | 90/108 | 78 | 0 | 30 | 96 |
| V5B | 12/108 | 57/108 | 45 | 0 | 63 | 91 |
| V5D | 20/108 | 76/108 | 56 | 0 | 52 | 88 |

## By group

| Model | Group | Baseline exact | Field exact | Improved | Regressed | Unchanged |
|---|---|---:|---:|---:|---:|---:|
| V5 | length4_consonant | 0/24 | 18/24 | 18 | 0 | 6 |
| V5 | length4_vowel | 0/24 | 24/24 | 24 | 0 | 0 |
| V5 | length5_consonant | 0/24 | 18/24 | 18 | 0 | 6 |
| V5 | length5_vowel | 0/24 | 18/24 | 18 | 0 | 6 |
| V5 | missing | 12/12 | 12/12 | 0 | 0 | 12 |
| V5B | length4_consonant | 0/24 | 6/24 | 6 | 0 | 18 |
| V5B | length4_vowel | 0/24 | 24/24 | 24 | 0 | 0 |
| V5B | length5_consonant | 0/24 | 0/24 | 0 | 0 | 24 |
| V5B | length5_vowel | 0/24 | 15/24 | 15 | 0 | 9 |
| V5B | missing | 12/12 | 12/12 | 0 | 0 | 12 |
| V5D | length4_consonant | 0/24 | 15/24 | 15 | 0 | 9 |
| V5D | length4_vowel | 8/24 | 21/24 | 13 | 0 | 11 |
| V5D | length5_consonant | 0/24 | 14/24 | 14 | 0 | 10 |
| V5D | length5_vowel | 0/24 | 14/24 | 14 | 0 | 10 |
| V5D | missing | 12/12 | 12/12 | 0 | 0 | 12 |

## Selected paired outcomes (first 3 by ID in each change class)

| Model | Status | Expected | Baseline | Alternative |
|---|---|---|---|---|
| V5 | improved | Leader: Ebaa. | Leader: baa. | Leader: Ebaa. |
| V5 | improved | Leader: Eban. | Leader: ba. | Leader: Eban. |
| V5 | improved | Leader: Ebaea. | Leader: ba. | Leader: Ebaea. |
| V5 | unchanged | Leader: Ebcn. | Leader: bcn. Loca. | Leader: Python. |
| V5 | unchanged | Leader: Ebcea. | Leader: bcea. Loca. | Leader: Python. |
| V5 | unchanged | Leader: Ebcen. | Leader: bcen. Loca. | Leader: Python. |
| V5B | improved | Leader: Ebaa. | Leader: baa. | Leader: Ebaa. |
| V5B | improved | Leader: Eban. | Leader: ba. | Leader: Eban. |
| V5B | improved | Leader: Ebaea. | Leader: baea. | Leader: Ebaea. |
| V5B | unchanged | Leader: Ebaen. | Leader: ba. | Leader: Room Cedar. |
| V5B | unchanged | Leader: Eban. | Leader: ba. | Leader: Room Pine. |
| V5B | unchanged | Leader: Ebaen. | Leader: ba. | Leader: Room Pine. |
| V5D | improved | Leader: Ebaa. | Leader: Ebaa. Lo. | Leader: Ebaa. |
| V5D | improved | Leader: Eban. | Leader: Eban. Lo. | Leader: Eban. |
| V5D | improved | Leader: Ebaea. | Leader: Lo. | Leader: Ebaea. |
| V5D | unchanged | Leader: Ebbi. | Leader: Ebbi. | Leader: Ebbi. |
| V5D | unchanged | Leader: Ebca. | Leader: Pytho. | Leader: Python. |
| V5D | unchanged | Leader: Ebcn. | Leader: Pytho. | Leader: Python. |

## Interpretation constraints

The alternative only constrains candidate spans to complete field values. It does **not** identify the field requested by the question. It is a deliberately narrow synthetic-grammar decoder experiment.
DEV is exploratory; HOLDOUT is single-use and cannot be used to choose another decoder rule without a new untouched test set.
