# V5D corrected logit diagnostic

Reconstructed from saved prediction logits, without retraining. Correctness and recovery are independent booleans.

## Correctness and recovery

| Model | Exact | Gold end top 5 | Recovery | Lost correct end | Abstained |
|---|---:|---:|---:|---:|---:|
| V5 | 7/96 | 40/96 | 0 | 27 | 0 |
| V5B | 32/96 | 54/96 | 0 | 6 | 0 |
| V5D | 21/96 | 79/96 | 11 | 12 | 4 |

## Raw end and joint-span score margins

Margin = best score minus gold score (logits, NOT probabilities). Lower positive values mean closer competition.

| Model | End gap <= 1 | End gap <= 3 | Joint gap <= 1 | Joint gap <= 3 | Median end gap | Median joint gap |
|---|---:|---:|---:|---:|---:|---:|
| V5 | 34/96 | 37/96 | 8/96 | 11/96 | 5.193 | 10.5927 |
| V5B | 38/96 | 39/96 | 37/96 | 39/96 | 6.0271 | 5.4959 |
| V5D | 28/96 | 68/96 | 26/96 | 55/96 | 1.7908 | 2.386 |

## Recovered examples

| Model | Expected | Predicted | End rank | End margin | Joint margin |
|---|---|---|---:|---:|---:|
| V5D | Leader: Dbbt. | Leader: Dbbt. | 2 | 0.366 | 0.000 |
| V5D | Leader: Dbgn. | Leader: Dbgn. | 2 | 1.624 | 0.000 |
| V5D | Leader: Dblt. | Leader: Dblt. | 2 | 0.691 | 0.000 |
| V5D | Leader: Dblt. | Leader: Dblt. | 2 | 0.688 | 0.000 |
| V5D | Leader: Dbbet. | Leader: Dbbet. | 2 | 1.661 | 0.000 |
| V5D | Leader: Dblet. | Leader: Dblet. | 2 | 1.494 | 0.000 |
| V5D | Leader: Dblet. | Leader: Dblet. | 2 | 1.483 | 0.000 |
| V5D | Leader: Dbbei. | Leader: Dbbei. | 2 | 1.844 | 0.000 |
| V5D | Leader: Dbgea. | Leader: Dbgea. | 2 | 1.336 | 0.000 |
| V5D | Leader: Dblei. | Leader: Dblei. | 2 | 1.643 | 0.000 |

## Caveats

This is DEV-only descriptive measurement. Gold joint-span rank cannot be reconstructed from top-five joint candidates alone. The original exclusive `error_category` field is retained for provenance but must not be used to count recoveries.
No final holdout accessed or checkpoint modified.
