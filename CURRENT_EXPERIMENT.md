# Current experiment: V5H — Question-aware paired-ranking loss

**Status:** Code and tests committed; local tests and training not yet run.
**Working branch:** experiment/question-aware-ranking-v5h
**Draft PR:** https://github.com/Br111t/grounded-qa/pull/9
**Base branch:** experiment/question-aware-field-v5g (not develop).

## Why V5H?

The completed V5G uniform-sampling ablation reached 24/48 DEV exact,
0/24 paired both-correct and 0/12 correct field switches. All 12
missing-target DEV questions received unsupported answers. Eleven of the
12 both-present sources showed relative field-score shifts in the right
direction, but Location still won on both questions.

The next controlled question is whether explicit field-ranking supervision
can improve question-aware Leader vs Location switching without increasing
model size or changing the inference decoder.

## What changes?

Only TRAIN adds a hinge penalty when both Leader and Location are present:
max(0, margin - (requested_joint_score - rival_joint_score)).
The coefficient is 0.5 and the raw-score margin is 1.0.

Architecture, dataset, seed 42, 1,200 uniform updates, Adam lr=0.001,
base answerability BCE, start/end CE and complete-field decoder remain as
in the completed V5G uniform baseline. Reserved EVAL remains sealed.

The V5H checkpoint is new and separate. V5G weights and prior artifacts
are not modified.

## Where to work

- [V5H plan and commands](experiments/question_aware_v5h/README.md)
- [V5H training code](experiments/question_aware_v5h/train_paired_ranking.py)
- [V5H regression tests](tests/experiments/test_paired_ranking_v5h.py)
- [V5G original uniform sampling code](experiments/question_aware_v5g/train_uniform_sampling.py)
- [V5C prior completed boundary probe](experiments/span_boundary_probe_v5c/README.md)

## Next local steps (Windows PowerShell, repository root)

Run the six new V5H loss-contract tests first:

~~~powershell
git fetch origin
git switch experiment/question-aware-ranking-v5h
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests/experiments -p "test_paired_ranking_v5h.py" -v
~~~

If they pass, run the unchanged V5G sampling tests, then follow the
V5H README's training command and share its summary.md. Its script accepts
the existing V5G uniform metrics.json read-only for a matched comparison.

**Do not** generate or inspect the reserved EVAL partition, overwrite a
.pt checkpoint, or reset/delete untracked V5G data and reports.

## Research guardrails

The primary DEV signal is paired both-correct and correct field switches,
not single-question 50% accuracy. Independently report false abstentions,
unsupported answers, field confusion and BCE/start/end losses. The trainer
also records optimization time and total time including diagnostics; V5G's
prior report did not capture time, so a measured runtime comparison with
that past run is unavailable.

This is a new experiment, not a retrospective edit to the V5G or V5C
results. It builds on the controlled-computation approach in the
[MITx Computational Thinking coursework](https://github.com/Coursework-Archive/math-through-code/tree/main/mitx-computational-thinking).
