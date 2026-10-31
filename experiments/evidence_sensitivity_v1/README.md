# Evidence sensitivity V1: fixed-question, variable-evidence experiment

> **Current repository layout:** Run commands from the repository root with
> `$env:PYTHONPATH = "$PWD\src"` set in PowerShell. The test suite uses
> `python -m unittest discover -s tests -t . -p "test_*.py" -v`.
> Saved `.pt` checkpoints are local, ignored by Git, and must not be overwritten.
> Historical branch-switch commands in older research notes are not needed on
> `refactor/repository-organization`.

**Question:** Can our original encoder-decoder Transformer use varying supplied evidence to choose a different leader, rather than recalling Mira?

The dataset is generated locally using `scripts/data/generate_evidence_data.py`. Keep the question, workshop identity, and location fixed; vary only the leader, including one missing-leader case.

## Run from the project root in PowerShell

The existing data is **local and untracked**; it was not added to GitHub by this experiment.

```powershell
python -m scripts.data.validate_data experiments/compact_prompt_v1/data/evidence_sensitivity_v1.jsonl

python -m unittest discover -s tests -t . -p "test_evidence_training.py" -v

# Diagnostic training on all 7 examples from fresh random weights:
python -m grounded_qa.training.train `
  --experiment-data experiments/compact_prompt_v1/data/evidence_sensitivity_v1.jsonl `
  --prompt-version compact_v1 `
  --steps 500 `
  --seed 42 `
  --checkpoint experiments/evidence_sensitivity_v1/evidence_sensitivity_v1_500_steps.pt

python -m grounded_qa.inference.predict `
  --checkpoint experiments/evidence_sensitivity_v1/evidence_sensitivity_v1_500_steps.pt `
  --data experiments/compact_prompt_v1/data/evidence_sensitivity_v1.jsonl
```

**Note:** `predict.py` displays the first row of each answerability class by default.
For this dataset, that is only two examples. To evaluate all seven, pass each
`--example-id` separately, or use its `--source-ids` argument with the
seven IDs: `evidence_001 evidence_002 evidence_003 evidence_004 evidence_005 evidence_006 evidence_007`.

The new `--experiment-data` training option is deliberately restricted to
`experiments/compact_prompt_v1/data/evidence_sensitivity_v1.jsonl` and independently checks each
fixed-question answer against the supplied leader. It selects **all 7 rows**
without requiring `--source-id`. Passing this dataset via ordinary `--data`
still fails the original training-source guard, by design.

This is a fitting experiment, **not evidence of generalization**. Do not modify
the final holdout or reuse it for training. After fitting, change an unrelated
information field and test whether predictions are still correct. Afterward,
design a new dataset with novel evidence combinations for evaluation.
