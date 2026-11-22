# V5G sampling ablation: equal exposure per question

From-scratch seed 42, original V5 span model, original loss, same
train/dev datasets, same complete-field decoder, Adam lr=0.001.
Only the example-selection schedule changes.
DEV monitored for diagnosis only; reserved EVAL untouched.

## Exposure counts

| Category | Unique examples | Training updates |
| --- | ---: | ---: |
| leader/answerable | 18 | 450 |
| leader/unanswerable | 6 | 150 |
| location/answerable | 18 | 450 |
| location/unanswerable | 6 | 150 |

Training SHA256: \`493d939ab6207799536670e1b90dfa07e67e08265274423fe87d7e96348771b3\`

## Whole-split diagnostics (not single-step training losses)

| Step | Partition | Answer BCE | Start CE | End CE | Both correct pairs | Correct switches | Exact accuracy | False abstentions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | train | 0.6634 | 4.3496 | 4.7140 | 1/24 | 0/12 | 33.3% | 2 |
| 0 | dev | 0.6620 | 4.4746 | 4.7913 | 2/24 | 0/12 | 33.3% | 3 |
| 120 | train | 0.5759 | 0.8765 | 1.9661 | 0/24 | 0/12 | 50.0% | 0 |
| 120 | dev | 0.5758 | 3.7504 | 2.1076 | 0/24 | 0/12 | 50.0% | 0 |
| 240 | train | 0.6090 | 0.8194 | 2.1827 | 0/24 | 0/12 | 50.0% | 0 |
| 240 | dev | 0.6092 | 4.3612 | 3.0257 | 0/24 | 0/12 | 50.0% | 0 |
| 360 | train | 0.5946 | 0.7989 | 1.8478 | 0/24 | 0/12 | 50.0% | 0 |
| 360 | dev | 0.5948 | 4.6463 | 2.3118 | 0/24 | 0/12 | 50.0% | 0 |
| 480 | train | 0.6177 | 0.7979 | 1.8279 | 0/24 | 0/12 | 50.0% | 0 |
| 480 | dev | 0.6177 | 4.9085 | 2.5249 | 0/24 | 0/12 | 50.0% | 0 |
| 600 | train | 0.6092 | 0.6523 | 1.3511 | 0/24 | 0/12 | 50.0% | 0 |
| 600 | dev | 0.6094 | 5.4130 | 1.9927 | 0/24 | 0/12 | 50.0% | 0 |
| 720 | train | 0.6001 | 0.7480 | 1.3048 | 1/24 | 1/12 | 52.1% | 0 |
| 720 | dev | 0.6002 | 4.9507 | 2.1294 | 0/24 | 0/12 | 50.0% | 0 |
| 840 | train | 0.6336 | 0.9909 | 1.8973 | 0/24 | 0/12 | 50.0% | 0 |
| 840 | dev | 0.6336 | 4.8522 | 2.8344 | 0/24 | 0/12 | 50.0% | 0 |
| 960 | train | 0.5776 | 0.4140 | 0.9261 | 2/24 | 2/12 | 54.2% | 0 |
| 960 | dev | 0.5778 | 5.0731 | 1.6869 | 0/24 | 0/12 | 50.0% | 0 |
| 1080 | train | 0.6476 | 0.4757 | 1.1376 | 1/24 | 1/12 | 52.1% | 0 |
| 1080 | dev | 0.6476 | 5.4654 | 2.1291 | 0/24 | 0/12 | 50.0% | 0 |
| 1200 | train | 0.6489 | 0.2517 | 1.2038 | 2/24 | 2/12 | 54.2% | 0 |
| 1200 | dev | 0.6491 | 4.8846 | 2.3740 | 0/24 | 0/12 | 50.0% | 0 |

## Interpretation

Start/end CE are computed on answerable examples only.
Paired accuracy includes correct abstention for missing targets.
Correct switches require both Leader and Location in evidence.
No evaluator, model architecture, loss weights or test partition was changed.
Compare the final TRAIN/DEV values with the original V5G baseline;
a change in performance is not evidence of semantic paraphrase generalization.
