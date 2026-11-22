# V5G: Answerability versus start/end losses

Read-only, dropout disabled. Same registered TRAIN/DEV data,
seeded pre-training initialization and saved 1,200-step checkpoint.
This compares endpoints, not intermediate training steps.

## TRAIN (48 questions)

| Metric | Initialization | Trained | Change (trained − initial) |
| --- | ---: | ---: | ---: |
| answerability_bce_all | 0.66344 | 1.18103 | 0.51759 |
| answerability_bce_answerable | 0.63508 | 1.57282 | 0.93774 |
| answerability_bce_unanswerable | 0.74852 | 0.00564 | -0.74287 |
| start_ce_answerable | 4.34955 | 0.54015 | -3.80940 |
| end_ce_answerable | 4.71395 | 1.41800 | -3.29595 |
| extraction_ce_answerable | 4.53175 | 0.97908 | -3.55268 |
| mean_total_objective_all | 4.06225 | 1.91533 | -2.14692 |
| answerability_accuracy_all | 0.72917 | 0.75000 | 0.02083 |
| start_head_accuracy_answerable | 0.00000 | 0.66667 | 0.66667 |
| end_head_accuracy_answerable | 0.00000 | 0.50000 | 0.50000 |
| both_heads_accuracy_answerable | 0.00000 | 0.41667 | 0.41667 |

### Trained loss by question and answerability

| Group | Rows | Answerability BCE | Start CE | End CE |
| --- | ---: | ---: | ---: | ---: |
| answerability:answerable | 36 | 1.57282 | 0.54015 | 1.41800 |
| answerability:unanswerable | 12 | 0.00564 | n/a | n/a |
| field:leader | 24 | 1.23430 | 0.97468 | 1.97876 |
| field:location | 24 | 1.12775 | 0.10563 | 0.85724 |

## DEV (48 questions)

| Metric | Initialization | Trained | Change (trained − initial) |
| --- | ---: | ---: | ---: |
| answerability_bce_all | 0.66197 | 1.19003 | 0.52806 |
| answerability_bce_answerable | 0.62976 | 1.58479 | 0.95503 |
| answerability_bce_unanswerable | 0.75861 | 0.00574 | -0.75287 |
| start_ce_answerable | 4.47457 | 4.31523 | -0.15935 |
| end_ce_answerable | 4.79129 | 2.05390 | -2.73738 |
| extraction_ce_answerable | 4.63293 | 3.18457 | -1.44836 |
| mean_total_objective_all | 4.13667 | 3.57845 | -0.55822 |
| answerability_accuracy_all | 0.72917 | 0.75000 | 0.02083 |
| start_head_accuracy_answerable | 0.00000 | 0.50000 | 0.50000 |
| end_head_accuracy_answerable | 0.00000 | 0.30556 | 0.30556 |
| both_heads_accuracy_answerable | 0.00000 | 0.25000 | 0.25000 |

### Trained loss by question and answerability

| Group | Rows | Answerability BCE | Start CE | End CE |
| --- | ---: | ---: | ---: | ---: |
| answerability:answerable | 36 | 1.58479 | 4.31523 | 2.05390 |
| answerability:unanswerable | 12 | 0.00574 | n/a | n/a |
| field:leader | 24 | 1.23774 | 8.61954 | 2.41523 |
| field:location | 24 | 1.14232 | 0.01091 | 1.69257 |

## Interpretation guardrails

- Lower cross-entropy (CE/BCE) is better; higher head accuracy is better.
- Start and end CE are defined **only on answerable examples**.
- The actual training objective averages start/end CE, and adds it to answerability BCE.
- Mean total objective weights extraction by the fraction of answerable examples.
- Start/end argmax accuracy is not equivalent to complete-field decoder accuracy.
- The untrained comparison is reconstructed with the stored initial seed;
 it does not recover the original optimizer or intermediate checkpoints.
- A decreasing loss does not establish semantic question conditioning.
- Endpoint losses alone cannot prove one task's gradient updates hurt the other;
 gradient/ablation experiments would be needed for that causal claim.
- Reserved EVAL was neither generated nor examined.
