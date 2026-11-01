# V5D boundary-error investigation (read-only)

Goal: explain why V5/V5B/V5D fail to select correct span ends for
consonant-ending names, without retraining or changing model weights.

This diagnostic reads ONLY the V5D DEV partition. It evaluates three
checkpoints on identical 96 records, predicts before consulting reference
offsets, and reports:

- Independent start/end head accuracy and signed byte-token end deltas.
- Jointly selected span accuracy and signed end deltas (excluding abstentions).
- Cases where the raw end is correct but joint selection is wrong, or vice versa.
- Per-example outputs with identifiers, predictions, ground truth and offsets.

The independent raw head can be wrong even when the joint decoder selects a
correct answer; the reverse can also happen. All offsets here are model
byte-token offsets, not character offsets. No model training occurs.

## Run from repo root (PowerShell)

```powershell
git fetch origin
git switch experiment/v5d-boundary-error-diagnostics
$env:PYTHONPATH = "$PWD\src"
python -m unittest tests.experiments.test_v5d_boundary_errors -v
python -m unittest discover -s tests -t . -p "test_*.py" -v

python -m experiments.balanced_boundary_v5d.diagnose_boundary_errors `
  --v5 experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt `
  --v5b experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt `
  --v5d experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt `
  --output-dir experiments/balanced_boundary_v5d/reports/boundary_probe_run1
```

New directory only, never overwrite an existing report. Files saved:
`metrics.json`, `predictions.csv`, `summary.md`.
Check full regression status before interpreting the findings. Checkpoint
files must be present locally; they are intentionally absent from Git.

## Interpretation

A negative end delta means early termination; a positive value means a
late boundary. Look for concentration of offsets by group and changes
between independent heads and jointly decoded spans. This is descriptive
DEV diagnostics, **not** proof of causal effects or final holdout accuracy.
Keep V5C locked and do not tune repeatedly to this development partition.
