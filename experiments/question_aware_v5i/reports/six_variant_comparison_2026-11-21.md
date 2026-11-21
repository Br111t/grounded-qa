# V5I–V5J controlled forward-pass comparison: six variants

Recorded 2026-11-21 from the completed local run output shared in the project discussion. This is a consolidated **exploratory DEV** record, not a claim of new reruns or sealed EVAL results.

## Data and common training protocol

All six runs use the registered V5G question-aware canonical Leader/Location examples: 48 TRAIN and 48 DEV questions (24 paired question sets on 24 distinct evidence sources in each split). TRAIN leader names begin with **F**, DEV leader names with **G**. The evidence sources are disjoint. Each includes 12 complete-source pairs and 12 incomplete-source pairs (6 missing Leader, 6 missing Location).

Common setup: encoder configuration, byte tokenizer, candidate-field decoder, training objective (answerability BCE + average answerable-only start/end CE + 0.5 × paired hinge ranking penalty with margin 1), uniform sampling, 1,200 updates, Adam learning rate 0.001, seed 42, gradient clipping at norm 1, and DEV measurements every 120 updates. Only the fusion/interaction formulation changes. Parameter counts differ in V5J. The two experiment scripts were executed separately, so wall-clock timings are indicative only.

## Final metrics: six approaches

**Important**: the `unsupported_answers` count is the number of **errors** on missing-target questions, not successful abstentions. **12/12 means every missing-target DEV question was answered incorrectly.**

| Experiment | Variant | Params | TRAIN exact | DEV exact | DEV paired both-correct | DEV correct switches on complete-source pairs | DEV unsupported-answer errors | DEV false abstentions | Optimization seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V5I activation | tanh | 132,931 | 35/48 | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 9.744 |
| V5I activation | GELU | 132,931 | 24/48 | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 7.044 |
| V5I activation | Linear | 132,931 | 29/48 | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 6.835 |
| V5J interaction | Multiplicative | 132,931 | 31/48 | 24/48 | 0/24 | 0/12 | 12/12 | 0 | 9.596 |
| V5J interaction | Bilinear | 132,867 | 36/48 | **36/48** | **12/24** | **12/12** | 12/12 | 0 | 6.844 |
| V5J interaction | FiLM | 132,995 | 36/48 | 25/48 | 1/24 | 1/12 | 12/12 | 0 | 6.780 |

DEV accuracies: tanh 50.0%, GELU 50.0%, Linear 50.0%, Multiplicative 50.0%, Bilinear **75.0%**, FiLM 52.1%.

### Equations implemented

Let `H_i` denote a contextual token state, `q` the question-byte mean-pooled encoder state and `z_i = W_f [H_i;q] + b_f`. Three V5I activation variants share:

- **tanh**: `H'_i = H_i + tanh(z_i)`
- **GELU**: `H'_i = H_i + GELU(z_i)`
- **linear**: `H'_i = H_i + z_i`

All three use linear start/end heads on `H'_i`.

V5J changes the interaction instead:

- **Multiplicative**: `H'_i = H_i + (W_h H_i) ⊙ (W_q q) + b_f`. Start/end/answerability heads receive `H'`.
- **Bilinear**: `start_i = start_head(H_i) + H_i^T W_start q`; `end_i = end_head(H_i) + H_i^T W_end q`. Answerability uses the unchanged `H_BOS`.
- **FiLM**: `(γ(q), β(q)) = Linear(q)`; `H'_i = H_i + γ(q) ⊙ H_i + β(q)`. Start/end/answerability heads receive `H'`.

These are the **implemented** formulae, not arbitrary activation swaps on an old checkpoint.

## Training timeline signal: bilinear

| Step | DEV exact | Correct complete-source switches | Unsupported-answer errors |
| ---: | ---: | ---: | ---: |
| 0 | 20/48 | 0/12 | 10/12 |
| 120 | 12/48 | 0/12 | 0/12 |
| 240 | 12/48 | 0/12 | 0/12 |
| 360 | 24/48 | 0/12 | 12/12 |
| 480 | 36/48 | 12/12 | 12/12 |
| 600 | 32/48 | 8/12 | 12/12 |
| 720 | 33/48 | 9/12 | 12/12 |
| 840 | 36/48 | 12/12 | 12/12 |
| 960 | 36/48 | 12/12 | 12/12 |
| 1080 | 36/48 | 12/12 | 12/12 |
| 1200 | 36/48 | 12/12 | 12/12 |

**Interpretation**: the explicit question-dependent bilinear span scores improve observed DEV field selection under this dataset and seed. No variant resolves DEV missing-target answerability. The repeated 36/48 score should not be read as fully solved question-aware QA; on this split the 12 incomplete-source queries asking for absent fields remain wrong.

## Experiment-control notes and next step

- The numbers above come from user-provided local console outputs; this consolidated document does not independently inspect the local `metrics.json` files or saved checkpoints.
- Do not call 12/12 `unsupported_answers` a success.
- The variant definitions are not equal in parameter count; the encoder/training/decoder are the matched controls.
- The results use one random seed and DEV repeatedly; preserve sealed EVAL and do not claim statistically established superiority.
- **Next separate dataset experiment**: preserve different names and evidence between splits, but draw name initials from the same distribution in TRAIN and DEV, avoiding the deliberate F-versus-G shift. Keep this registered V5I–V5J comparison as the original-split baseline; do not modify its data or checkpoints.
- For a new dataset, rerun at least bilinear and the original tanh control with matched training settings, and consider all three V5J variants if attribution across interaction methods remains the goal.
