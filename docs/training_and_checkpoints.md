# Training and checkpoints

All commands below run from the repository root in PowerShell inside the project's virtual environment.

## Environment and tests

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

Experiment-specific commands, seeds, data paths, and checkpoint formats are documented in their own READMEs. Do not start a historical training command against a filename that already exists.

## Checkpoint handling

- `.pt` files are **local** and ignored by Git; pushing source code does not back them up.
- Each model family may require its matching loader and checkpoint metadata.
- Keep existing checkpoints in their corresponding `experiments/.../checkpoints/` directories when possible; historical exceptions exist.
- Do not overwrite completed checkpoints, and do not treat an evaluation as a training step.
- Older checkpoint metadata can contain the original paths used when models were trained. The V5C evaluator accepts the known registered original and reorganized paths; the checkpoint weights themselves were not changed.

## Verified local backup (October 31, 2026)

A local backup copied **24** `.pt` checkpoint files, preserving their relative experiment paths. The user verified **24 of 24** original/backup SHA-256 hashes matched. The backup is outside the repository at:

`C:\Users\britt\grounded-qa-checkpoint-backup-20261031-121020`

This protects against accidental repository changes, **not** physical drive failure. An additional copy to a separate drive or secure storage is recommended.

## Reproducing the most recent read-only comparison

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c --checkpoint experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt --compare-checkpoint experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt
```

This command was executed successfully with both existing checkpoints and all 96 V5C development examples. See [the V5C README](../experiments/span_boundary_probe_v5c/README.md) for results and limitations.
