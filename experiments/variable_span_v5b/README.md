# V5B — variable-length span selection

## Question

V5 reached 78/108 (72.2%) on original V3 development, including
24/24 recombined 4-character names but 0/24 entirely unfamiliar
5-character names. Does varying TRAINING name lengths improve held-out
start/end boundaries, while keeping the encoder-only architecture intact?

## Design and limitations

- Architecture, source masking, answerability threshold 0.5, question,
  optimizer (Adam 0.001), seed 42, batch size 1, and 3,000 updates stay fixed.
- Original V3 files remain completely unchanged; the V5B checkpoint is new.
- Original V3 **train** contains 168 rows (144 answerable 4-letter names,
  24 missing evidence). V5B adds 72 fresh answerable rows: 24 each
  containing 3-, 5-, and 6-character leader names (8 names per length
  with 3 contexts each). V5B therefore trains on 240 rows.
- Two **development-only** groups each contain 24 rows (12 names with
  2 contexts): **familiar5** (5-letter familiar-pattern names withheld
  from training) and **novel4** (4-letter names using withheld starting
  syllables). They never enter the optimizer.
- Existing **V3 recombined** provides unfamiliar complete names, length 4;
  **V3 unseen** provides unfamiliar starting syllables and length 5.
- This creates a practical 2x2 length/pattern comparison using original
  dev partitions plus the two new groups.
- All V5B data is generated deterministically with seed 42 and strict
  validation. The generator will never overwrite changed existing files.
- Because V5B has MORE training rows, this is an augmentation experiment,
  NOT a perfectly size-matched causal isolation of length variation.
  Training frequencies and exposure per row also change. Further
  size-matched experiments or repeated seeds may be needed.
- V5B still uses synthetic fixed-question leader fields; successes do
  not establish general-purpose question answering.
- Model still learns the start and end positions; no hardcoded trimming at
  punctuation has been added.

## Data groups

| Group | Pattern | Name length | Purpose |
|---|---|---|---|
| V3 train | familiar names | 4 | original train |
| V5B extra_train | familiar syllables | 3, 5, 6 | additional train |
| V3 recombined | familiar syllables, new full names | 4 | dev |
| V5B familiar5 | familiar syllables, new full names | 5 | dev |
| V5B novel4 | withheld starting syllables | 4 | dev |
| V3 unseen | withheld starting syllables | 5 | dev |
| V3 missing | leader absent | n/a | dev |

## Steps (PowerShell, repository root)

1. Pull the branch:

    git status --short
    git fetch origin
    git switch --track origin/experiment/variable-span-v5b

2. Run new tests, then regenerate only new V5B files:

    python -m unittest discover -s tests -t . -p "test_variable_span_v5b.py" -v
    python -m unittest discover -s tests -t . -p "test_span_v5.py" -v
    python -m experiments.variable_span_v5b.generate_span_data_v5b

3. **Evaluate V5 baseline first** with start/end diagnostics on the
   extended development partitions. This does not modify weights:

    python -m experiments.variable_span_v5b.evaluate_span_v5b --checkpoint experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt --group all

4. Train fresh V5B checkpoint, retaining ALL previous checkpoints:

    python -m experiments.variable_span_v5b.train_span_v5b --experiment-data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl --extra-train experiments/variable_span_v5b/data/evidence_span_v5b_extra_train.jsonl --steps 3000 --loss-every 120 --seed 42 --missing-ratio 0.33 --checkpoint experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt

5. Evaluate the same groups on the new V5B model:

    python -m experiments.variable_span_v5b.evaluate_span_v5b --checkpoint experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt --group all

## Interpretation

The evaluator reports:

- Exact-match answers and unsupported answers for missing evidence.
- **Independent** start/end head argmax correctness, even when the
  answerability head abstains.
- Correct selected start, selected end, complete span, and any overlap.
- Overextended and truncated spans; false abstentions.

Compare the ORIGINAL five V3 groups separately from the two extra
dev groups; do not interpret the combined 156-row development total
as a direct replacement for the original 108-row benchmark. For the
two-by-two length/pattern diagnostic, compare V3 recombined, V5B
familiar5, V5B novel4, and V3 unseen. This isolates measured
behaviors by group, while acknowledging data augmentation confounds.

The reference answers are used to score the model only AFTER
predictions. Training never reads development/final-holdout files.

## Implementation

- `experiments/variable_span_v5b/generate_span_data_v5b.py`: fixed new train/dev data generation
- `experiments/variable_span_v5b/train_span_v5b.py`: same V5 model, extra training samples
- `experiments/variable_span_v5b/evaluate_span_v5b.py`: start/end and answerability diagnostics
- `tests/experiments/test_variable_span_v5b.py`: isolation and boundary regression tests
