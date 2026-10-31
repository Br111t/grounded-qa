# V4B: supervised copy attention versus copy gate

> **Current repository layout:** Run commands from the repository root with
> `$env:PYTHONPATH = "$PWD\src"` set in PowerShell. The test suite uses
> `python -m unittest discover -s tests -t . -p "test_*.py" -v`.
> Saved `.pt` checkpoints are local, ignored by Git, and must not be overwritten.
> Historical branch-switch commands in older research notes are not needed on
> `refactor/repository-organization`.

## Research question

V4 can copy a familiar answer from the correct source position, but its gate
can override the copy route; for unfamiliar names it may point to unrelated
source characters. This experiment isolates TWO auxiliary training signals.

- Alignment supervision: at each TRAINING leader-name output byte, increase
  the pointer's probability of that byte's exact source position.
- Copy-gate supervision: at the same output bytes, increase the probability
  of choosing the source-copy route instead of vocabulary generation.

No changes to the original model's inference API or architecture, V3 dataset
or train/dev partition, compact_v1 prompt, sampling schedule, learning rate,
seed, or main next-token cross-entropy loss. The auxiliary positions are
derived from validated TRAINING rows only. No development or holdout input
is used to build targets; no source positions are saved in checkpoints.

## Implementation details

All controls are optional additions to src/grounded_qa/training/train.py.

    --alignment-weight 0.5
    --copy-gate-weight 0.5

A flag is independently active only when greater than 0.0. Defaults are
zero, preserving all existing standard Transformer / V4 training behavior.
Positive auxiliary weights are restricted to the V4 pointer variant,
registered V3 training file, and compact_v1 prompt format.

Training minimizes:

    total = answer_cross_entropy
          + alignment_weight * mean(-log(P_pointer(correct source byte position)))
          + copy_gate_weight * mean(-log(P(copy route)))

The auxiliary losses are averaged over the leader-name output bytes only,
not the 'Leader: ' prefix, punctuation, or EOS. For unanswerable training
examples, the auxiliary losses are zero and ordinary answer loss remains.
Full-dataset losses printed by the existing training script are the
original answer cross-entropy, NOT the total auxiliary objective.

Test that alignments work for each validated training row, that unrelated
parts do not receive auxiliary gradients, and that normal V4 inference
is identical with the new forward_components method.

## Pull and test (PowerShell, repository root)

    git status --short
    git fetch origin
    git switch refactor/repository-organization

    python -m unittest discover -s tests -t . -p "test_evidence_copy_v4b.py" -v
    python -m unittest discover -s tests -t . -p "test_copy_model_v4.py" -v
    python -m unittest discover -s tests -t . -p "test_copy_diagnostics_v4.py" -v
    python -m unittest discover -s tests -t . -p "test_evidence_v3.py" -v

V3 dataset files must remain present locally; do not regenerate or change them.
Regression tests must pass before starting the longer runs.

## Three independent experiments

Run EACH command from the repository root, on the new V4B branch. All are
fresh random-weight 3,000-update runs with seed 42 and the same 33% missing
sampling ratio. Their filenames are unique and intentionally different from
the saved V4 baseline experiments/evidence_copy_v4b/evidence_copy_v4/checkpoints/evidence_copy_v4_3000_steps.pt.

### A: Alignment supervision only

    python -m grounded_qa.training.train --experiment-data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl --model-variant pointer_v4 --prompt-version compact_v1 --sampling balanced --missing-ratio 0.33 --steps 3000 --loss-every 120 --seed 42 --alignment-weight 0.5 --copy-gate-weight 0 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_alignment_3000.pt

### B: Gate supervision only

    python -m grounded_qa.training.train --experiment-data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl --model-variant pointer_v4 --prompt-version compact_v1 --sampling balanced --missing-ratio 0.33 --steps 3000 --loss-every 120 --seed 42 --alignment-weight 0 --copy-gate-weight 0.5 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_gate_3000.pt

### C: Both signals

    python -m grounded_qa.training.train --experiment-data experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl --model-variant pointer_v4 --prompt-version compact_v1 --sampling balanced --missing-ratio 0.33 --steps 3000 --loss-every 120 --seed 42 --alignment-weight 0.5 --copy-gate-weight 0.5 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_both_3000.pt

Evaluate each saved checkpoint (same five V3 partitions, original reference
answers, free-running exact match):

    python -m experiments.evidence_sensitivity_v3.evaluate_evidence_v3 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_alignment_3000.pt --group all
    python -m experiments.evidence_sensitivity_v3.evaluate_evidence_v3 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_gate_3000.pt --group all
    python -m experiments.evidence_sensitivity_v3.evaluate_evidence_v3 --checkpoint experiments/evidence_copy_v4b/checkpoints/evidence_copy_v4b_both_3000.pt --group all

Follow with the existing --copy-details diagnostic on the Xezen and Tato
examples to inspect pointer attention and gate choices.

## Comparison table (fill after the runs)

| Category | V4 baseline | Alignment | Gate | Both |
|---|---:|---:|---:|---:|
| Training | 61/168 | — | — | — |
| Known | 13/48 | — | — | — |
| Recombined | 0/24 | — | — | — |
| Unseen | 0/24 | — | — | — |
| Missing | 11/12 | — | — | — |
| Dev total | 24/108 | — | — | — |

Do not interpret a low teacher-forced loss as generalization. Keep old
checkpoints untouched and compare exact free-running answers. Initial
auxiliary weights 0.5 are an experimental choice, not tuned hyperparameters.
