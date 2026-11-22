# V5F · Counterfactual question sensitivity

Exploratory and read-only; unchanged evidence for each question pair.
Models were trained primarily on Leader questions. Different question types are out-of-distribution tests, not expected trained competencies.

## Selected fields by question type

| Model | Question | Leader | Location | Topic | Abstained | Correct target |
|---|---|---:|---:|---:|---:|---:|
| V5 | leader_original | 3 | 0 | 1 | 0 | 3/4 |
| V5 | leader_paraphrase | 3 | 0 | 1 | 0 | 3/4 |
| V5 | leader_reworded | 3 | 0 | 0 | 1 | 3/4 |
| V5 | location | 3 | 0 | 1 | 0 | 0/4 |
| V5 | topic | 3 | 0 | 0 | 1 | 0/4 |
| V5 | unrelated | 3 | 0 | 1 | 0 | 0/4 |
| V5B | leader_original | 2 | 1 | 0 | 1 | 2/4 |
| V5B | leader_paraphrase | 2 | 1 | 0 | 1 | 2/4 |
| V5B | leader_reworded | 2 | 1 | 0 | 1 | 2/4 |
| V5B | location | 2 | 2 | 0 | 0 | 2/4 |
| V5B | topic | 2 | 1 | 0 | 1 | 0/4 |
| V5B | unrelated | 2 | 1 | 0 | 1 | 1/4 |
| V5D | leader_original | 2 | 1 | 1 | 0 | 2/4 |
| V5D | leader_paraphrase | 2 | 1 | 0 | 1 | 2/4 |
| V5D | leader_reworded | 2 | 0 | 0 | 2 | 2/4 |
| V5D | location | 2 | 1 | 0 | 1 | 1/4 |
| V5D | topic | 2 | 1 | 0 | 1 | 0/4 |
| V5D | unrelated | 2 | 1 | 0 | 1 | 1/4 |

## Paired changes against original Leader question

| Model | Question | Field selection changed | Target advantage (individual examples) |
|---|---|---:|---|
| V5 | leader_paraphrase | 0/4 | 12.80, 19.74, 1.71, -5.26 |
| V5 | leader_reworded | 1/4 | 10.23, 19.42, 1.80, -5.39 |
| V5 | location | 0/4 | -25.39, -20.02, -1.37, -10.15 |
| V5 | topic | 1/4 | -9.67, -20.05, -8.06, 4.81 |
| V5 | unrelated | 0/4 | n/a, n/a, n/a, n/a |
| V5B | leader_paraphrase | 0/4 | 23.48, 11.94, -6.33, -1.95 |
| V5B | leader_reworded | 0/4 | 23.23, 11.80, -6.60, -2.11 |
| V5B | location | 1/4 | -23.95, -11.62, 6.57, 1.92 |
| V5B | topic | 0/4 | -23.33, -14.55, -7.54, -3.68 |
| V5B | unrelated | 0/4 | n/a, n/a, n/a, n/a |
| V5D | leader_paraphrase | 1/4 | 1.40, 11.72, -1.81, -5.00 |
| V5D | leader_reworded | 2/4 | 1.61, 12.40, -1.62, -4.76 |
| V5D | location | 1/4 | -7.74, -11.90, 1.81, -0.87 |
| V5D | topic | 1/4 | -1.94, -14.31, -4.67, 0.76 |
| V5D | unrelated | 1/4 | n/a, n/a, n/a, n/a |

## Reading these results

- A changed selected field alone is not evidence of understanding.
- A paraphrase should preserve the target field, but score stability is a separate observable measure.
- An unrelated question has no evidence-grounded answer; abstention is desirable but not a learned capability we can assume.
- This is a small, knowingly out-of-distribution diagnostic on seen and unseen name families; not an accuracy claim.
- The prior V5E holdout is consumed and is not reused here.
