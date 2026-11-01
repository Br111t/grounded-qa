# V5D — Balanced boundary training (controlled experiment)

## Hypothesis

V5C found strong end-boundary failures for consonant-ending names and some
length sensitivity. V5D tests whether balancing 4/5-character names crossed
with vowel/consonant endings in TRAIN improves *unseen* span selection.

This is a synthetic, restricted QA experiment, not evidence of general
language understanding.

## Immutable boundaries

- V3 original TRAIN (168 rows) remains unchanged.
- New V5D TRAIN: 96 answerable examples, 24 per cell.
- New V5D DEV: 96 answerable examples, 24 per cell, distinct name stems.
- V5C's locked 96 examples remain diagnostic-only and are NEVER used to train.
- Existing V5/V5B weights, saved checkpoints, and final holdout are untouched.
- The candidate uses the same encoder and loss as V5/V5B; random initialization
  with seed 42, 3,000 steps, default 0.33 missing-class sampling.
- The original V3 unanswerable training examples remain included.
- This is **not** an example-count-matched causal control against V5B (264
  vs 240 rows). Run an equal-count control before attributing gains solely
  to terminal-character balance.
- Both partitions share a synthetic context grammar, so passing DEV does
  not prove real-world generalization or arbitrary-question capability.

## Quick start (PowerShell, repository root)

```powershell
git switch experiment/balanced-boundaries-v5d
$env:PYTHONPATH = "$PWD\src"
python -m unittest tests.experiments.test_balanced_boundary_v5d -v
python -m unittest discover -s tests -t . -p "test_*.py" -v
python -m experiments.balanced_boundary_v5d.generate_data_v5d
python -m experiments.balanced_boundary_v5d.train_v5d --checkpoint experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt
python -m experiments.balanced_boundary_v5d.evaluate_v5d --checkpoint experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt --output-dir experiments/balanced_boundary_v5d/reports/v5d_seed42_run1
```

For a fair direct comparison on the *same V5D DEV examples*, also evaluate
the existing V5 and V5B checkpoints with
`evaluate_v5d.py --checkpoint ... --output-dir ...`, selecting distinct
report directory names for each. Do not overwrite any report directory.

## Reporting

Each run writes `metrics.json` and `summary.md` with checkpoint and data
hashes; overall and per-cell exact/head scores; false abstentions and
span errors; plus an interpretation prompt. Reports are immutable per path.
Save a new path for each run. Keep observations distinct from hypotheses.
This evaluation is DEV only: using its outcomes to choose successive models
will eventually overfit DEV. Reserve an untouched final holdout for one
precommitted selection.

## Next investigations

1. Verify generator/tests and train one independent V5D checkpoint.
2. Compare V5, V5B, and V5D on exactly the same V5D DEV rows.
3. If balanced examples help, run a **quantity-matched** additional-data
   control before assigning causality to vowel/consonant balance.
4. Investigate generalization across alternate contexts and unseen question
   wording separately. Keep V6 recurrent/Hopfield work independent.

This branch intentionally does not modify `CURRENT_EXPERIMENT.md` until
the data, training script, and evaluator have all been executed and reviewed.
