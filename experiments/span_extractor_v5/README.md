# V5 — Question-conditioned Transformer span extraction

## Why this experiment exists

The best V4B attention-aligned pointer-generator copied familiar names
but still scored 0/24 on wholly unseen names. Increasing V4B training
from 3,000 to 6,000 updates did not help:

| Metric | V4B 3000 | V4B 6000 |
|---|---:|---:|
| Train | 141/168 | 147/168 |
| Known | 42/48 | 40/48 |
| Recombined | 8/24 | 8/24 |
| Unseen | 0/24 | 0/24 |
| Missing | 12/12 | 12/12 |
| Development | 62/108 | 60/108 |

V5 replaces byte-by-byte answer generation with an **encoder-only
Transformer**. Its heads predict (1) start position, (2) end position
and (3) answerability. At inference the chosen byte span is copied
literally from Information, instead of predicting each answer byte.

V5 has no pretrained model and does not load any V4B weights.
Unlike V4B, V5 does not require a generated EOS or an autoregressive decoder.

## Controlled design

- Reuse **exactly** the V3 train/known/recombined/unseen/missing JSONL files.
- Identical fixed question and compact_v1 prompt.
- Maintain the V2/V3 class-aware 33% missing-evidence sampling schedule.
- Seed 42; 3,000 independent updates; batch size 1; CPU; Adam 0.001.
- Train only using the registered data/training/evidence_sensitivity_v3_train.jsonl.
- Use source span targets from the validated **training reference only**.
- At inference, model receives only the Information and Question text.
- A mask allows span endpoints inside **Information**, never the Question.
- Fixed answerability threshold 0.5. A predicted missing answer returns
  precisely 'Insufficient information.'.
- For the fixed V3 benchmark, a selected span is rendered 'Leader: {span}.'.
  This formatting is benchmark-specific; it is not a general answer generator.
- The extractor can copy source values it has never seen during training,
  provided it learns the correct start/end positions.
- The max extracted span is 16 UTF-8 bytes; V3 name lengths fit this.
- The final original holdout is not read.

## Pull and run tests (PowerShell, repository root)

    git status --short
    git fetch origin
    git switch --track origin/experiment/span-extractor-v5

    python -m unittest discover -s tests -p "test_span_v5.py" -v
    python -m unittest discover -s tests -p "test_evidence_v3.py" -v

Stop if tests fail; don't train until the issue is addressed.

## Train

    python src/train_span_v5.py --experiment-data data/training/evidence_sensitivity_v3_train.jsonl --steps 3000 --loss-every 120 --seed 42 --missing-ratio 0.33 --checkpoint models/evidence_span_v5_3000_steps.pt

Training begins from fresh weights, not V4B checkpoint weights. The V5
checkpoint format is intentionally independent from the earlier
encoder-decoder checkpoint format. No optimizer state is saved.

## Evaluate

    python src/evaluate_span_v5.py --checkpoint models/evidence_span_v5_3000_steps.pt --group all

The evaluator reuses the V3 partition and leakage checks. It reports exact
full-answer matches for training, known, recombined, unseen, missing, and
the overall development set (excluding training), plus false abstentions
and unsupported answers on missing rows.

Interpretation note: span extraction is suited to copying substrings
present in evidence. It does not naturally solve synthesis or reasoning
that requires generating a new answer. These initial results use one
synthetic fixed-question dataset, not general-world QA or final holdout.

## Source files

- src/span_model_v5.py — model, training targets, prediction and checkpoint I/O
- src/train_span_v5.py — V3 train-only training program
- src/evaluate_span_v5.py — same partitions as V4B, exact-match evaluator
- tests/test_span_v5.py — masks, targets, gradients, extraction and checkpoint tests

All V4B files and checkpoints remain unchanged.
