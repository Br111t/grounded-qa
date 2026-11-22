# V5G: Data and Development Prediction Summary

DEV only; the reserved evaluation set is not accessed.

## Training data

24 evidence sources and 48 question rows.

| Question | Answerable | Missing target |
| --- | --- | --- |
| leader | 18 | 6 |
| location | 18 | 6 |

**Field presence in evidence**

| Fields present | Sources |
| --- | --- |
| Leader+Location+Topic | 12 |
| Leader+Topic | 6 |
| Location+Topic | 6 |

## Development data

24 evidence sources and 48 question rows.

| Question | Answerable | Missing target |
| --- | --- | --- |
| leader | 18 | 6 |
| location | 18 | 6 |

**Field presence in evidence**

| Fields present | Sources |
| --- | --- |
| Leader+Location+Topic | 12 |
| Leader+Topic | 6 |
| Location+Topic | 6 |

## Development predictions

Exact accuracy: **24/48 (50.0%)**.
Both questions correct on same evidence: **0/24**.
Correct field switches when both fields exist: **0/12**.
False abstentions: **12**.

| Asked for | Predicted | Count |
| --- | --- | --- |
| leader | abstain | 12 |
| leader | location | 12 |
| location | abstain | 12 |
| location | location | 12 |

### Example: Wrong-field selection

Source: `v5g_dev_02`

Evidence: Fictional Workshop W0001. Leader: Gbct. Location: Room Birch 27. Topic: Gardening.

Question: What is the leader of Workshop W0001?

Expected: **Leader: Gbct.**

Actual: **Location: Room Birch 27**

Answerability probability: 0.9925

| Candidate field | Joint logit score |
| --- | --- |
| Location | 13.249 |
| Leader | -0.576 |
| Topic | -2.966 |

### Example: False abstention

Source: `v5g_dev_00`

Evidence: Fictional Workshop W0001. Location: Room Maple 25. Topic: Python.

Question: Where is Workshop W0001 located?

Expected: **Location: Room Maple 25.**

Actual: **Insufficient information.**

Answerability probability: 0.0083

| Candidate field | Joint logit score |
| --- | --- |
| Location | 12.564 |
| Topic | -9.201 |

### Example: Correct extraction

Source: `v5g_dev_02`

Evidence: Fictional Workshop W0001. Leader: Gbct. Location: Room Birch 27. Topic: Gardening.

Question: Where is Workshop W0001 located?

Expected: **Location: Room Birch 27.**

Actual: **Location: Room Birch 27**

Answerability probability: 0.9923

| Candidate field | Joint logit score |
| --- | --- |
| Location | 13.212 |
| Leader | -1.335 |
| Topic | -2.483 |

### Example: Correct abstention

Source: `v5g_dev_00`

Evidence: Fictional Workshop W0001. Location: Room Maple 25. Topic: Python.

Question: What is the leader of Workshop W0001?

Expected: **Insufficient information.**

Actual: **Insufficient information.**

Answerability probability: 0.0056

| Candidate field | Joint logit score |
| --- | --- |
| Location | 12.680 |
| Topic | -9.233 |

## Interpretation notes

A correct abstention differs from an incorrect refusal to answer.
Joint-logit scores are not calibrated probabilities.
This report does not measure unseen paraphrases, unrelated questions or Topic transfer.
Do not run reserved EVAL until the model and evaluator are frozen.
