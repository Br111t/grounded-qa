# V5G training-set prediction diagnostic

Same frozen checkpoint, same decoder, same 0.5 answerability threshold.
TRAIN is in-sample. DEV is out-of-sample; no reserved EVAL was accessed.

## Overall results

| Metric | TRAIN | DEV |
| --- | --- | --- |
| Exact accuracy | 26/48 (54.2%) | 24/48 (50.0%) |
| Both questions correct | 2/24 | 0/24 |
| Correct switches on complete pairs | 2/12 | 0/12 |
| False abstentions | 12 | 12 |

## Training-set predictions by requested field

| Question | Correct | Total |
| --- | ---: | ---: |
| Leader | 9 | 24 |
| Location | 17 | 24 |

## Training-set confusion counts

| Requested | Predicted | Count |
| --- | --- | ---: |
| leader | abstain | 12 |
| leader | leader | 3 |
| leader | location | 9 |
| location | abstain | 12 |
| location | leader | 1 |
| location | location | 11 |

## Training-set performance by evidence structure

| Fields available | Correct | Questions | False abstentions |
| --- | ---: | ---: | ---: |
| Leader+Location+Topic | 14 | 24 | 0 |
| Leader+Topic | 6 | 12 | 6 |
| Location+Topic | 6 | 12 | 6 |

## Paired question comparison (same evidence)

| Source | Fields present | Leader requested → selected | Location requested → selected | Both correct |
| --- | --- | --- | --- | --- |
| v5g_train_00 | Location | abstain | abstain | no |
| v5g_train_01 | Leader | abstain | abstain | no |
| v5g_train_02 | Leader, Location | location | location | no |
| v5g_train_03 | Leader, Location | location | location | no |
| v5g_train_04 | Location | abstain | abstain | no |
| v5g_train_05 | Leader | abstain | abstain | no |
| v5g_train_06 | Leader, Location | location | location | no |
| v5g_train_07 | Leader, Location | location | location | no |
| v5g_train_08 | Location | abstain | abstain | no |
| v5g_train_09 | Leader | abstain | abstain | no |
| v5g_train_10 | Leader, Location | leader | location | yes |
| v5g_train_11 | Leader, Location | leader | location | yes |
| v5g_train_12 | Location | abstain | abstain | no |
| v5g_train_13 | Leader | abstain | abstain | no |
| v5g_train_14 | Leader, Location | location | location | no |
| v5g_train_15 | Leader, Location | location | location | no |
| v5g_train_16 | Location | abstain | abstain | no |
| v5g_train_17 | Leader | abstain | abstain | no |
| v5g_train_18 | Leader, Location | location | location | no |
| v5g_train_19 | Leader, Location | leader | leader | no |
| v5g_train_20 | Location | abstain | abstain | no |
| v5g_train_21 | Leader | abstain | abstain | no |
| v5g_train_22 | Leader, Location | location | location | no |
| v5g_train_23 | Leader, Location | location | location | no |

## How to interpret

- TRAIN strong, DEV weak: suspect generalization, value patterns, or evidence-format shortcuts.
- TRAIN weak, DEV weak: first inspect training targets, optimization, sampling, and decoder behavior.
- Both-correct accuracy matters more than overall accuracy for question-aware field selection.
- Candidate logits and answerability scores for every row are in predictions.json.
- Training-set performance is diagnostic, not an estimate of generalization.
- This script does not alter any existing checkpoint or consume the reserved evaluation set.
