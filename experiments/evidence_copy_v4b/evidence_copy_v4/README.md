# V4 — Pointer-generator evidence copying

> **Current repository layout:** Run commands from the repository root with
> `$env:PYTHONPATH = "$PWD\src"` set in PowerShell. The test suite uses
> `python -m unittest discover -s tests -t . -p "test_*.py" -v`.
> Saved `.pt` checkpoints are local, ignored by Git, and must not be overwritten.
> Historical branch-switch commands in older research notes are not needed on
> `refactor/repository-organization`.

**Research question:** Can an explicit learned source-copy pathway improve
exact reproduction of leader values that did not appear as complete training
answers? Compare against the unchanged V3B standard Transformer checkpoints.

## Architecture

The original `TinyGroundedQA` and its weights/checkpoint formats remain
available. V4 adds a separate `CopyGroundedQA` model variant.

- The original Transformer encoder and decoder calculate contextual hidden states.
- A new learned query/key attention produces a distribution over **source byte
  positions**, ignoring padding and EOS.
- Repeated input bytes are summed into the same vocabulary probability using
  `scatter_add`.
- A learned sigmoid gate combines vocabulary generation and source copying at
  every answer position.
- The model outputs normalized **log probabilities**. Existing cross-entropy
  loss and greedy generation still work without special decoding rules.
- EOS must be generated; it is never copied.

The pointer reads from the **entire compact prompt**, not a hardcoded leader
substring. Successful unseen-name copying is a hypothesis to test, not a
guaranteed feature. The model is trained from random weights, not pretrained.

## What stays fixed

- Identical **V3 dataset** (168 train; 48 known, 24 recombined, 24 unseen,
  12 missing development examples).
- Fixed question and `compact_v1` prompt.
- Seed 42, Adam learning rate 0.001, CPU batch size 1.
- Class-aware 33% missing-evidence sampling.
- 3,000 updates for the first controlled comparison with V3's 3,000-step run.
- The final holdout remains untouched.

## Before training (PowerShell in repo root)

```powershell
git status --short
git fetch origin
git switch refactor/repository-organization

python -m unittest discover -s tests -t . -p "test_copy_model_v4.py" -v
python -m unittest discover -s tests -t . -p "test_evidence_v3.py" -v
python -m unittest discover -s tests -t . -p "test_compact_prompt.py" -v

python -m scripts.data.validate_data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl
```

The V3 dataset must already exist locally. No new dataset generation is needed.

## Train V4 from random weights

```powershell
python src/grounded_qa/training/train.py `
  --experiment-data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl `
  --model-variant pointer_v4 `
  --prompt-version compact_v1 `
  --sampling balanced `
  --missing-ratio 0.33 `
  --steps 3000 `
  --loss-every 120 `
  --seed 42 `
  --checkpoint experiments/evidence_copy_v4b/evidence_copy_v4/checkpoints/evidence_copy_v4_3000_steps.pt
```

Do **not** overwrite any previous checkpoint. V4 checkpoint metadata records
`model_variant=pointer_v4`, so `load_checkpoint` can restore the appropriate
architecture. Checkpoint loading defaults to the old `TinyGroundedQA` for
legacy checkpoints without a `model_variant` field.

## Evaluate training and development partitions

```powershell
python -m experiments.evidence_sensitivity_v3.evaluate_evidence_v3 `
  --checkpoint experiments/evidence_copy_v4b/evidence_copy_v4/checkpoints/evidence_copy_v4_3000_steps.pt `
  --group all
```

The V3 evaluator is reused intentionally: it loads the model from its own
checkpoint metadata, verifies the V3 data partitions, then compares the
*same* training, known, recombined, unseen, and missing reference answers.

Baseline at 3,000 steps (standard Transformer):
- Train: 47/168 (28.0%)
- Known: 7/48 (14.6%)
- Recombined: 0/24
- Unseen: 0/24
- Missing: 12/12
- Development total: 19/108 (17.6%)

V3B at 6,000 steps improved only slightly; do not interpret that unequal
training budget as the primary V4 comparison.

## Interpretation

Report exact generated answers and failure examples, not only teacher-forced
loss. A trained pointer-generator may still choose familiar generated names,
copy irrelevant prompt bytes, or produce no valid EOS. We must measure whether
V4 improves **recombined and unseen name accuracy** while retaining missing
evidence behavior. Passing six regression tests does not prove correct answers.


## V4 copy-path diagnostics

The existing src/check_model.py now supports an optional --copy-details flag.
The flag is V4-only and never updates weights or checkpoints.

After pulling the branch, run the focused tests:

    python -m unittest discover -s tests -t . -p "test_copy_diagnostics_v4.py" -v
    python -m unittest discover -s tests -t . -p "test_copy_model_v4.py" -v

Inspect an incorrect known name:

    python -m grounded_qa.inference.check_model --data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_known.jsonl --example-id evidence_v3_known_0001_q01 --checkpoint experiments/evidence_copy_v4b/evidence_copy_v4/checkpoints/evidence_copy_v4_3000_steps.pt --copy-details

Inspect an entirely unseen leader name:

    python -m grounded_qa.inference.check_model --data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_unseen.jsonl --example-id evidence_v3_unseen_0001_q01 --checkpoint experiments/evidence_copy_v4b/evidence_copy_v4/checkpoints/evidence_copy_v4_3000_steps.pt --copy-details

The original diagnostic still reports teacher-forced loss. The new section
independently traces greedy generated leader-name characters, using only
the MODEL'S OWN previous predicted tokens as the decoder prefix.

For each generated alphabetic byte of a 'Leader: ...' answer it prints:
- Generation and copying gate fractions.
- Vocabulary, copy, and combined probabilities plus weighted contributions.
- Copy attention mass on the source leader-value region, and on the
  leader-value source bytes that match the generated byte.
- Three top-attended source positions, with their source characters.
- Reference next byte probability only while output remains prefix-aligned
  with the reference; the reference never affects generation.

Because attention and copy distribution range over the full compact prompt,
a high copy fraction does not mean the model copied the correct Leader field.
The shown attention is V4's explicit copy-pointer attention, not the
original Transformer multi-head cross-attention.

If the model does not generate a name, the command still prints its final
greedy answer. The extension is only a diagnostic; it does not modify
normal forward outputs or checkpoint formats.
