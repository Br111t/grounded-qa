# V5E: complete-field boundary decoder (reversible, no training)

**Goal:** Test whether restricting candidate spans to complete evidence-field values
improves exact question-answer extraction versus unchanged V5/V5B/V5D decoding.

**What is fixed in advance:**
- Each model's checkpoint weights, answerability threshold (0.5), question,
  inputs and reference answers remain unchanged.
- The baseline is the original unconstrained highest-scoring joint span.
- The alternative enumerates *all* syntactically complete `Label: Value.`
  fields within evidence; it **does not hard-code Leader** or learn the correct
  question-to-field mapping. It scores each complete value using the same
  start-logit + end-logit rule. Missing or oversized candidates abstain.
- Both model predictions are made before the gold reference is checked.
- Exact accuracy is primary; paired improvements, regressions, unchanged,
  and raw text changes are reported separately.
- No decoder changes after freezing and before the one-time holdout test.

**Important limitation:** This is deliberately a strong *syntactic* prior
on a tightly structured synthetic corpus. It is not evidence that the model
learned transferable question understanding or can process free-form evidence.
Even holdout uses the same synthetic context grammar, but new E-initial names
and 12 missing-evidence cases. Only generalization within this grammar is tested.

## Step A — Baseline-preserving DEV comparison

From the repository root in PowerShell:

```powershell
git fetch origin
git switch experiment/field-boundary-decoder-v5e
$env:PYTHONPATH = "$PWD\src"
python -m unittest tests.experiments.test_field_boundary_decoder_v5e -v
python -m unittest discover -s tests -t . -p "test_*.py" -v

python -m experiments.field_boundary_decoder_v5e.compare_decoders `
  --phase dev `
  --v5 experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt `
  --v5b experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt `
  --v5d experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt `
  --baseline-cases experiments/balanced_boundary_v5d/reports/logit_probe_run2/all_cases.json `
  --output-dir experiments/field_boundary_decoder_v5e/reports/dev_run1
```

DEV validates that the saved baseline predictions are byte-for-byte
unchanged for all 288 model/example pairs. If any prediction differs,
execution stops before publishing. Save and review these immutable artifacts:
`all_pairs.json`, `metrics.json`, `summary.md`, `freeze.json`.
Review regressions as carefully as improvements; complete syntax may select
the wrong field (for example `Topic`) even while returning a valid span.

## Step B — Freeze BEFORE accessing holdout labels/predictions

The DEV command writes the code/checkpoint SHA256 identities into
`dev_run1/freeze.json`. Once the DEV report has been inspected and the
experiment design is accepted, leave all scripts and checkpoint paths
unchanged. The holdout command refuses changed code or checkpoints.

Run the generator **only after deciding to freeze**:

```powershell
python -m experiments.field_boundary_decoder_v5e.generate_holdout `
  --freeze experiments/field_boundary_decoder_v5e/reports/dev_run1/freeze.json
```

It creates an immutable new `data/v5e_holdout.jsonl` with 96 answerable
examples (four length/ending conditions) and 12 unanswerable examples, using
an E-name family and fixed seed 9041. This is a *new* holdout, not any
previous V5C or original holdout. **Do not inspect or edit it.**

## Step C — One-time holdout evaluation

```powershell
python -m experiments.field_boundary_decoder_v5e.compare_decoders `
  --phase holdout `
  --v5 experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt `
  --v5b experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt `
  --v5d experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt `
  --freeze experiments/field_boundary_decoder_v5e/reports/dev_run1/freeze.json `
  --holdout-file experiments/field_boundary_decoder_v5e/data/v5e_holdout.jsonl `
  --output-dir experiments/field_boundary_decoder_v5e/reports/holdout_run1
```

Generated files preserve *both* predictions, the reference answer, exact
correctness flags, improved/regressed/unchanged labels, candidate field scores,
checkpoint/data provenance and breakdowns by name-ending group including
missing information.

### Pre-registered acceptance rule

Accept the alternative *only if*, on the untouched holdout:
1. Its **overall exact-answer count exceeds the baseline** on the same
   checkpoint, and
2. Its **number of regressions is not greater than its improvements**,
   which follows from condition 1 but is audited separately, and
3. On the 12 missing-evidence examples it makes **no additional errors**.

Evaluate each checkpoint independently; do not choose a model or tune a
decoder based on holdout performance. Any later adjustment requires a new,
separately designed untouched evaluation. If the field decoder improves this
synthetic holdout, the conclusion is limited to that grammar.

## Boundaries and status

No training, no checkpoint rewrite, no V5C reuse or final-holdout access.
Each output directory is immutable. The evaluation remains **unrun** until
local tests and experiment commands complete. Do not merge before review.
