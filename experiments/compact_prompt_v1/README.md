# Compact prompt V1: controlled training experiment

> **Current repository layout:** Run commands from the repository root with
> `$env:PYTHONPATH = "$PWD\src"` set in PowerShell. The test suite uses
> `python -m unittest discover -s tests -t . -p "test_*.py" -v`.
> Saved `.pt` checkpoints are local, ignored by Git, and must not be overwritten.
> Historical branch-switch commands in older research notes are not needed on
> `refactor/repository-organization`.

## Research question

Does replacing the long instruction prompt with a compact information/question/answer
prompt improve question-sensitive generation while all other variables stay fixed?

## Controls

- Existing data: `experiments/foundations/data/practice_wordings.jsonl` (20 questions, one workshop).
- Original long prompt: `original` (default for existing checkpoints).
- Treatment: `compact_v1`.
- Transformer architecture, tokenizer, answers, source, optimizer and training order unchanged.
- Start from fresh random weights using seed 42. Do not fine-tune the old checkpoint.
- Original baseline: 4,000 updates, loss 0.0818, exact generation 2/20.
- Preserve the existing 4,000-step checkpoint and do not touch the final holdout.

## Running from the repository root in PowerShell

```powershell
python -m scripts.data.validate_data experiments/foundations/data/practice_wordings.jsonl

# Short scout: NOT comparable in training budget to the 4,000-update baseline.
python -m grounded_qa.training.train `
  --data experiments/foundations/data/practice_wordings.jsonl `
  --source-id generated_workshop_0001 `
  --prompt-version compact_v1 `
  --seed 42 `
  --steps 500 `
  --checkpoint experiments/compact_prompt_v1/checkpoints/workshop_0001_compact_v1_500_steps.pt

python -m grounded_qa.inference.predict `
  --checkpoint experiments/compact_prompt_v1/checkpoints/workshop_0001_compact_v1_500_steps.pt `
  --data experiments/foundations/data/practice_wordings.jsonl `
  --source-id generated_workshop_0001

python experiments/compact_prompt_v1/diagnose.py `
  --checkpoint experiments/compact_prompt_v1/checkpoints/workshop_0001_compact_v1_500_steps.pt

# Run 4,000 updates separately only if diagnostics justify the CPU cost:
# python -m grounded_qa.training.train ... --steps 4000 --checkpoint experiments/compact_prompt_v1/checkpoints/workshop_0001_compact_v1_4000_steps.pt
```

To compare the baseline under the exact same diagnostic:

```powershell
python experiments/compact_prompt_v1/diagnose.py `
  --checkpoint experiments/foundations/checkpoints/workshop_0001_wordings_4000_steps.pt
```

`predict.py` and `check_model.py` automatically use the version recorded in the
checkpoint. Original checkpoints omit `prompt_version` and remain compatible.

## Interpretation

The 500-update run is a cheap directional scout, not a fair 4,000-update
comparison. Track full-answer exact matches and how token probabilities change
when ONLY the question changes. Never infer English comprehension from a low
teacher-forced training loss or matching the single repeated information source.
