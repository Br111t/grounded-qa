# V5G — Question-aware Leader/Location training

This is a **new training experiment on the existing Span Transformer v5 architecture**, with
a distinct model checkpoint, rather than a new architecture version. Decoder is
**complete-field v2**. Prior V5E/V5F predictions, checkpoints and consumed
holdouts remain unchanged.

## Registered objectives and datasets (before training)

- **Primary:** correctly select Leader *or* Location on the same evidence.
- **Training:** 24 independent F-name evidence identities × 2 canonical questions = 48
  examples; 6 missing Leader and 6 missing Location targets. Topic is a
  distractor in evidence but **never a supervised target**.
- **Development:** disjoint G-name identities and the same canonical Leader/Location
  questions, 48 examples. This set can guide training choices.
- **Reserved evaluation:** disjoint H-name identities, 192 examples:
  48 canonical, 96 unfamiliar paraphrases, 24 unrelated requests and
  24 Topic-transfer questions. Do not generate this data until training,
  development review, and code/checkpoint freeze are complete.
- All 3 partitions split by **source identity**, not question row, so no
  evidence passage is shared across partitions.
- Field order reverses for half the records; name families are disjoint.\nRoom identifiers are also unique across all three partitions, preventing\nmissing-Leader passages (which lack a differentiating name) from repeating.
- The same evidence appears in the canonical Leader/Location *pair* within
  a partition, permitting controlled question-only comparisons.
- This is a small synthetic evaluation, **not** evidence of open-domain
  semantic understanding. Topic transfer is exploratory, not a required
  condition for accepting a Leader/Location training improvement.

## Baselines, reporting and acceptance

Report separately:
1. **Question sensitivity:** score deltas when changing only the question;
   general switch rate vs correct switch rate among the 12 pairs where both\n   target fields exist (the other 12 pairs include one missing field)
2. **Question-aware selection:** exact requested-field/value accuracy per
   field and paired both-correct rate (primary metric)
3. **Paraphrases:** unseen paraphrase accuracy by field and accuracy gap vs
   canonical, with consistency inspected per source
4. **Question-conditioned abstention:** rejection of unsupported questions,
   false abstentions on answerable questions, unsupported answer rate and Brier
5. **Topic transfer:** field-selection accuracy and gold-field margins/ranks
6. **Robustness (later test):** field-order invariance and source-value substitution

No single weighted score is used to hide tradeoffs. The desired DEV
improvement is both-correct Leader/Location paired selection, with no
regression in the 12 missing-target development questions. **No holdout
threshold tuning**; any changes suggested by evaluation require a new
separate holdout.

## Training (run from repo root, Windows PowerShell)

```powershell
git fetch origin
git switch experiment/question-aware-field-v5g
$env:PYTHONPATH = "$PWD\src"

python -m unittest tests.experiments.test_question_aware_v5g -v
python -m experiments.question_aware_v5g.data_design --partition train
python -m experiments.question_aware_v5g.data_design --partition dev

python -m experiments.question_aware_v5g.train `
  --steps 1200 `
  --seed 42 `
  --checkpoint experiments/question_aware_v5g/checkpoints/v5g_seed42_steps1200.pt
```

**Important:** do not create the reserved `eval` partition yet.
The trainer reads **only** `data/train.jsonl`. It uses random initialization,
the existing Span Transformer architecture and span training loss, a
deterministic class-balanced sampling schedule, Adam lr=0.001 and 0.5
answerability threshold. It writes a new checkpoint with training-data SHA256.

## Development evaluation

```powershell
python -m experiments.question_aware_v5g.evaluate `
  --phase dev `
  --checkpoint experiments/question_aware_v5g/checkpoints/v5g_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5g/reports/dev_run1
```

The report saves per-question candidate scores, field margins/ranks,
paired both-correct rate, correct switch counts, confusion, false
abstention and Brier score. It also writes a `freeze.json`.

Review the DEV results and **choose one frozen checkpoint** before
touching the reserved evaluation. Do not change the evaluator after
freezing; the later evaluation validates its exact file hash.

## Reserved evaluation (only AFTER freezing)

```powershell
python -m experiments.question_aware_v5g.data_design --partition eval
python -m experiments.question_aware_v5g.evaluate `
  --phase eval `
  --checkpoint experiments/question_aware_v5g/checkpoints/v5g_seed42_steps1200.pt `
  --freeze experiments/question_aware_v5g/reports/dev_run1/freeze.json `
  --output-dir experiments/question_aware_v5g/reports/eval_run1
```

The freeze checks the selected checkpoint hash and evaluator code hash.
The reserved dataset is generated using the preregistered design; once
inspected it is considered **consumed**, and never reused to tune the
same model.

## Version ownership

- Architecture: `span_v5`, unchanged
- Training experiment/checkpoint: `v5g_leader_location_question_selection`
- Decoder: `complete_field_values_joint_logit_v1` (complete-field v2)
- Evaluation design: `V5G`

No results are claimed before local tests, training and evaluation finish.


## Read-only diagnostic: did each training objective improve?

After the original V5G training and DEV evaluation, compare **the same seed-42
model at initialization** with its frozen, final checkpoint using identical
TRAIN and DEV inputs, evaluation mode, and dropout disabled.

```powershell
python -m unittest tests.experiments.test_loss_components_v5g -v

python -m experiments.question_aware_v5g.diagnose_loss_components `
  --checkpoint experiments/question_aware_v5g/checkpoints/v5g_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5g/reports/loss_components_run1
```

Open `reports/loss_components_run1/summary.md` for TRAIN/DEV tables separating:
- Answerability binary cross entropy, including by answerable/unanswerable labels
- Start-position and end-position cross entropy on answerable questions only
- Answerability classification accuracy and start/end head accuracy
- Combined loss averaged across all questions (including zero extraction loss
  on unanswerable examples)
- Splits by Leader/Location, and JSON metrics by evidence structure

This diagnostic **does not retrain, change the existing checkpoint or read
reserved EVAL**. It compares endpoint losses, **not** loss during training.
The original 1,200-step run did not save component-wise trajectories.
To establish whether one objective deteriorated *during* optimization, a
future fresh training run must record whole-split component losses at fixed
checkpoints; endpoints alone cannot establish gradient interference.


## Controlled sampling ablation: uniform exposure to each question

Run a **new** from-scratch experiment using the same V5 Transformer,
same datasets, same loss, same decoder, 1,200 updates, seed 42 and
Adam lr=0.001. The **only intended training change** is how examples
are scheduled.

The original schedule samples each of four (field, answerability)
categories equally; the new schedule shuffles all 48 rows each epoch,
visiting each row once. At 1,200 steps there are 25 complete epochs,
so each row is used exactly 25 times.

| Category | Rows | Original updates | Uniform updates |
| --- | ---: | ---: | ---: |
| Leader answerable | 18 | 300 | 450 |
| Leader unanswerable | 6 | 300 | 150 |
| Location answerable | 18 | 300 | 450 |
| Location unanswerable | 6 | 300 | 150 |

The new run evaluates **whole TRAIN and DEV** (no dropout) at step 0
and every 120 updates, reporting separate answerability BCE, start CE,
end CE, exact accuracy, false abstentions, paired both-correct accuracy
and correct field switches. These diagnostic passes have no gradients
and do not change the weights.

From repo root, on branch `experiment/question-aware-field-v5g`:

```powershell
python -m unittest tests.experiments.test_uniform_sampling_v5g -v

python -m experiments.question_aware_v5g.train_uniform_sampling `
  --steps 1200 `
  --seed 42 `
  --measure-every 120 `
  --checkpoint experiments/question_aware_v5g/checkpoints/v5g_uniform_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5g/reports/uniform_sampling_run1
```

Inspect `reports/uniform_sampling_run1/summary.md` for the timeline
and `metrics.json` for complete measured values. The run refuses
to overwrite its checkpoint or report directory. It neither opens nor
generates reserved EVAL. The original V5G checkpoint and DEV report
remain the baseline; do not use the previous DEV freeze manifest
with a different checkpoint.
