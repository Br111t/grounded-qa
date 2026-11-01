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
