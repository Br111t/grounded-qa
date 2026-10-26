# Compact prompt V1: controlled training experiment

## Research question

Does replacing the long instruction prompt with a compact information/question/answer
prompt improve question-sensitive generation while all other variables stay fixed?

## Controls

- Existing data: `data/training/practice_wordings.jsonl` (20 questions, one workshop).
- Original long prompt: `original` (default for existing checkpoints).
- Treatment: `compact_v1`.
- Transformer architecture, tokenizer, answers, source, optimizer and training order unchanged.
- Start from fresh random weights using seed 42. Do not fine-tune the old checkpoint.
- Original baseline: 4,000 updates, loss 0.0818, exact generation 2/20.
- Preserve the existing 4,000-step checkpoint and do not touch the final holdout.

## Running from the repository root in PowerShell

```powershell
python src/validate_data.py data/training/practice_wordings.jsonl

# Short scout: NOT comparable in training budget to the 4,000-update baseline.
python src/train.py `
  --data data/training/practice_wordings.jsonl `
  --source-id generated_workshop_0001 `
  --prompt-version compact_v1 `
  --seed 42 `
  --steps 500 `
  --checkpoint models/workshop_0001_compact_v1_500_steps.pt

python src/predict.py `
  --checkpoint models/workshop_0001_compact_v1_500_steps.pt `
  --data data/training/practice_wordings.jsonl `
  --source-id generated_workshop_0001

python experiments/compact_prompt_v1/diagnose.py `
  --checkpoint models/workshop_0001_compact_v1_500_steps.pt

# Run 4,000 updates separately only if diagnostics justify the CPU cost:
# python src/train.py ... --steps 4000 --checkpoint models/workshop_0001_compact_v1_4000_steps.pt
```

To compare the baseline under the exact same diagnostic:

```powershell
python experiments/compact_prompt_v1/diagnose.py `
  --checkpoint models/workshop_0001_wordings_4000_steps.pt
```

`predict.py` and `check_model.py` automatically use the version recorded in the
checkpoint. Original checkpoints omit `prompt_version` and remain compatible.

## Interpretation

The 500-update run is a cheap directional scout, not a fair 4,000-update
comparison. Track full-answer exact matches and how token probabilities change
when ONLY the question changes. Never infer English comprehension from a low
teacher-forced training loss or matching the single repeated information source.
