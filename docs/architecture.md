# Architecture

grounded-qa is a local PyTorch learning project exploring question-aware answers grounded in supplied `information` and `question` fields. The repository retains multiple experimental approaches; they are not one interchangeable checkpoint format.

## Shared Python code

| Directory | Purpose |
|---|---|
| [`src/grounded_qa/models/`](../src/grounded_qa/models/) | Initial encoder-decoder model, V4 copy model, and V5 span model |
| [`src/grounded_qa/tokenization/`](../src/grounded_qa/tokenization/) | Tokenization |
| [`src/grounded_qa/prompting/`](../src/grounded_qa/prompting/) | Prompt formatting and previews |
| [`src/grounded_qa/training/`](../src/grounded_qa/training/) | Training and checkpoint utilities |
| [`src/grounded_qa/inference/`](../src/grounded_qa/inference/) | Prediction and model inspection |
| [`scripts/data/`](../scripts/data/) | Shared data generation, splitting, and validation |
| [`scripts/diagnostics/`](../scripts/diagnostics/) | Shared diagnostics |

## Model approaches

- **Initial encoder-decoder Transformer:** predicts output tokens. The earliest experiments build the vocabulary and model from scratch.
- **V4/V4B copy model:** retains generated answers while adding a learned source-copy route; V4B tests training signals for alignment and copy-gate behavior.
- **V5/V5B span extractor:** selects start and end byte positions in supplied evidence and predicts answerability. This differs from autoregressive answer generation. V5B augments training with variable-length names.
- **V5C:** a read-only diagnostic of how the two span-model checkpoints generalize across name length and final-character class; no training.

Each experiment's [README](../experiments/) documents its individual hypothesis, controls, and limitations. For the latest experiment, see [CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md). The original step-by-step implementation narrative is preserved in [the learning journal](history/learning_journal.md).
