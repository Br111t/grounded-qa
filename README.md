# grounded-qa

A personal Python/PyTorch learning project for building and evaluating **question-aware models that answer from supplied evidence**. The work begins with a small Transformer trained from scratch and progresses through source-copying and span-extraction experiments. No pretrained model is required for the recorded experiments.

The research goal is to understand when a model uses the supplied information, when it memorizes training patterns, and when it should respond with **`Insufficient information.`** Results come from narrow synthetic datasets, not a general-purpose QA benchmark.

## Start here

- **[Current experiment](CURRENT_EXPERIMENT.md)** — what is being investigated now, entry points, and most recent verified results.
- **[Architecture](docs/architecture.md)** — the different model families and shared Python modules.
- **[Dataset design](docs/dataset_design.md)** — source, training, development, and holdout boundaries.
- **[Training and checkpoints](docs/training_and_checkpoints.md)** — environment, regression tests, loading models, and verified backups.
- **[Original learning journal](docs/history/learning_journal.md)** — the complete, unchanged original README, preserved for educational history.

## Repository map

```text
grounded-qa/
├── CURRENT_EXPERIMENT.md      # Active research pointer
├── docs/                      # Architecture, data, checkpoints, history
├── experiments/               # Experiment notes, scripts, datasets, checkpoints*
├── src/grounded_qa/           # Shared models, training, inference, prompting, tokenization
├── scripts/                   # Shared data utilities and diagnostics
├── data/                      # Shared source, training, holdout data
├── tests/                     # Regression tests and experiment tests
├── notebooks/                 # Notebook-based exploration
└── requirements.txt
```

`* .pt` checkpoint weights are local, excluded from Git, and backed up separately.

## Experiment index

| Experiment | Focus |
|---|---|
| [Foundations](experiments/foundations/) | Initial datasets, tokenizer, and model learning |
| [Compact prompt V1](experiments/compact_prompt_v1/README.md) | Prompt-length and question-sensitivity comparison |
| [Evidence sensitivity V1](experiments/evidence_sensitivity_v1/README.md) | Fixed-question, changing-evidence baseline |
| [Evidence sensitivity V2 (and V2B)](experiments/evidence_sensitivity_v2/README.md) | Separate development partitions and class-aware sampling |
| [Evidence sensitivity V3](experiments/evidence_sensitivity_v3/README.md) | Known, recombined, unseen, and missing-evidence cases |
| [Evidence copying V4](experiments/evidence_copy_v4b/evidence_copy_v4/README.md) | Pointer-generator source copying |
| [Evidence copying V4B](experiments/evidence_copy_v4b/README.md) | Alignment and copy-gate supervision |
| [Span extractor V5](experiments/span_extractor_v5/README.md) | Start/end span selection and answerability |
| [Variable span V5B](experiments/variable_span_v5b/README.md) | Training with varied answer lengths |
| [Span boundary probe V5C](experiments/span_boundary_probe_v5c/README.md) | Read-only factorial test of span-end generalization |

Each experiment README contains its original motivation, controls, commands, and limitations. Its historical results are not replaced by the latest experiment's metrics.

## Quick start (Windows PowerShell)

From the repository root, with Python and a virtual environment available:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

On October 31, 2026, **93 regression tests passed** and both saved V5/V5B checkpoints loaded for the V5C read-only evaluation. For the complete V5C command and results, follow [the active experiment](CURRENT_EXPERIMENT.md).

**Do not** run a training command against an existing checkpoint path or use final holdout data for training. The project keeps experimental model weights and observations distinct from application code.

## Answering contract

- **Answerable:** answer using the provided information.
- **Partially answerable:** answer supported parts; label missing parts `Insufficient information.`
- **Unanswerable:** return `Insufficient information.`
- Do not infer unsupported facts from outside knowledge.

The original learning sequence, including examples and step-by-step commands from the first implementation, remains available in [the archived learning journal](docs/history/learning_journal.md). Its commands reflect the original layout and are **historical**, not current execution instructions.
