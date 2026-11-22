# V5D boundary error diagnostic

**Read-only · V5D development data · 96 examples per checkpoint**

## Overall

| Model | Exact | Raw start | Raw end | Selected span | Abstentions |
|---|---:|---:|---:|---:|---:|
| V5 | 7/96 | 19/96 | 34/96 | 7/96 | 0 |
| V5B | 32/96 | 84/96 | 38/96 | 32/96 | 0 |
| V5D | 21/96 | 51/96 | 22/96 | 21/96 | 4 |

## Per-condition breakdown

| Model | Group | Exact | Raw start | Raw end |
|---|---|---:|---:|---:|
| V5 | length4_vowel | 6/24 | 6/24 | 24/24 |
| V5 | length4_consonant | 0/24 | 6/24 | 0/24 |
| V5 | length5_vowel | 1/24 | 4/24 | 10/24 |
| V5 | length5_consonant | 0/24 | 3/24 | 0/24 |
| V5B | length4_vowel | 21/24 | 21/24 | 24/24 |
| V5B | length4_consonant | 0/24 | 21/24 | 0/24 |
| V5B | length5_vowel | 11/24 | 21/24 | 14/24 |
| V5B | length5_consonant | 0/24 | 21/24 | 0/24 |
| V5D | length4_vowel | 9/24 | 12/24 | 16/24 |
| V5D | length4_consonant | 4/24 | 13/24 | 0/24 |
| V5D | length5_vowel | 5/24 | 13/24 | 6/24 |
| V5D | length5_consonant | 3/24 | 13/24 | 0/24 |

## End-position offset distributions

Offsets are predicted end minus gold end, measured in model byte-token positions (inclusive). Negative = early; positive = late. Selected offsets exclude abstentions.

### V5
- length4_vowel: raw {'0': 24}; selected {'0': 24}
- length4_consonant: raw {'-21': 3, '-9': 6, '-2': 7, '-1': 2, '4': 3, '6': 3}; selected {'-2': 10, '-1': 2, '4': 3, '6': 9}
- length5_vowel: raw {'-22': 1, '-10': 3, '-3': 4, '-2': 2, '0': 10, '6': 4}; selected {'-3': 4, '-2': 2, '0': 11, '6': 7}
- length5_consonant: raw {'-22': 3, '-10': 6, '-3': 7, '-2': 2, '6': 6}; selected {'-3': 9, '-2': 3, '6': 12}
- Raw end correct but selected span wrong: 27; raw end wrong but selected span correct: 0

### V5B
- length4_vowel: raw {'0': 24}; selected {'0': 24}
- length4_consonant: raw {'-9': 3, '-2': 10, '-1': 3, '6': 8}; selected {'-2': 13, '-1': 3, '6': 8}
- length5_vowel: raw {'-3': 6, '-2': 2, '0': 14, '6': 2}; selected {'-3': 6, '-2': 2, '0': 14, '6': 2}
- length5_consonant: raw {'-22': 3, '-10': 3, '-3': 13, '-2': 3, '6': 2}; selected {'-3': 19, '-2': 3, '6': 2}
- Raw end correct but selected span wrong: 6; raw end wrong but selected span correct: 0

### V5D
- length4_vowel: raw {'-21': 1, '0': 16, '4': 4, '15': 3}; selected {'-15': 2, '-3': 3, '0': 12, '4': 4, '15': 3}
- length4_consonant: raw {'-22': 6, '-21': 3, '-20': 3, '4': 9, '15': 3}; selected {'-20': 1, '-15': 2, '-3': 3, '-1': 2, '0': 5, '4': 7, '12': 1, '15': 3}
- length5_vowel: raw {'-23': 6, '-4': 7, '0': 6, '4': 2, '14': 3}; selected {'-4': 9, '-2': 2, '0': 7, '4': 2, '12': 1, '14': 1}
- length5_consonant: raw {'-23': 6, '-22': 3, '-21': 3, '-4': 7, '4': 2, '14': 3}; selected {'-21': 1, '-4': 10, '-2': 2, '0': 4, '4': 3, '12': 1, '14': 1}
- Raw end correct but selected span wrong: 12; raw end wrong but selected span correct: 11

## Interpretation

**Observation:** Compare offset concentration and disagreements between raw heads and jointly decoded spans. See `predictions.csv` for individual cases.

**Hypotheses to assess:** A consistent -1 offset could indicate a systematic early boundary; widely dispersed offsets would suggest a broader position-selection problem. Neither is proved by these metrics alone.

**Next test:** Design a new pre-registered holdout or a controlled training experiment only after reviewing these errors. Avoid repeatedly optimizing against this DEV set.

No training, threshold tuning, V5C reuse or final-holdout access.
