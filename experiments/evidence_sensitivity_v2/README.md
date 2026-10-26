# Evidence sensitivity V2 — fixed question, varying evidence

## Research goal

Test whether the original small Transformer can reproduce supported values from
varying evidence instead of memorizing familiar names. The question is identical
in every example:

> What is the leader of Workshop W0001?

The workshop ID remains W0001. The leader, irrelevant location/topic, ordering
of facts, and presence/absence of the leader are the experimental variables.

## Datasets

Run `src/generate_evidence_data_v2.py` to create all four independently named
files (the V1 generator and existing holdout files are not changed):

| Partition | Path | Examples |
|---|---|---:|
| Training | `data/training/evidence_sensitivity_v2_train.jsonl` | 108 |
| Known-name development evaluation | `data/evaluation/evidence_sensitivity_v2_known.jsonl` | 24 |
| Unseen-name development evaluation | `data/evaluation/evidence_sensitivity_v2_unseen.jsonl` | 24 |
| Missing-leader development evaluation | `data/evaluation/evidence_sensitivity_v2_missing.jsonl` | 12 |

Known-name evaluation has **different context arrangements** from training for
each name. Unseen-name evaluation reserves four entirely unseen leader names.
Missing-leader evaluation contains contexts disjoint from missing-leader training
contexts. There is no duplicate source information across partitions.

The known-name test is a within-vocabulary composition test, the unseen-name
test is an out-of-vocabulary copying test, and the missing-leader test asks
whether the model can abstain on new contexts. These are development evaluation
sets, not the original final holdout.

## Commands (PowerShell, project root)

```powershell
git pull --ff-only origin experiment/compact-prompt-v1

python -m unittest discover -s tests -p "test_evidence_v2.py" -v
python -m unittest discover -s tests -p "test_evidence_training.py" -v

python src/generate_evidence_data_v2.py --seed 42

python src/validate_data.py data/training/evidence_sensitivity_v2_train.jsonl
python src/validate_data.py data/evaluation/evidence_sensitivity_v2_known.jsonl
python src/validate_data.py data/evaluation/evidence_sensitivity_v2_unseen.jsonl
python src/validate_data.py data/evaluation/evidence_sensitivity_v2_missing.jsonl

# Start with 500 updates, evaluate, then decide whether 1000+ is justified.
python src/train.py `
  --experiment-data data/training/evidence_sensitivity_v2_train.jsonl `
  --prompt-version compact_v1 `
  --steps 500 `
  --seed 42 `
  --checkpoint models/evidence_sensitivity_v2_500_steps.pt

python src/evaluate_evidence_v2.py `
  --checkpoint models/evidence_sensitivity_v2_500_steps.pt `
  --group all
```

The `--experiment-data` option accepts only the registered V1 and V2
**training** files; it rejects the evaluation files. The V1 data and
checkpoints remain valid.

## Interpretation

Report exact-match accuracy separately for known-name, unseen-name and
missing-evidence groups. The model might learn familiar name-selection before
learning to reproduce previously unseen names; that distinction is expected
and worth measuring. Training loss does not establish generalization. Do not
read or train on the existing final holdout to tune this experiment.

## V2B: class-aware training schedule (separate from the V2 baseline)

The V2 baseline trains with the original sequential sampler and reports
56/96 correct answerable **training** examples and 0/12 correct unanswerable
training examples at 2,000 updates. Its development accuracy was known 14/24,
unseen 0/24, missing 0/12.

The V2B experiment changes **only the frequency of training updates** for
answerable vs unanswerable examples. It keeps the same 108 unique training
examples, architecture, prompt, optimizer, step budget, and seed. It does not
read evaluation files or change the original sequential sampler by default.

```powershell
git pull --ff-only origin experiment/compact-prompt-v1

python -m unittest discover -s tests -p "test_evidence_sampling.py" -v

python src/train.py `
  --experiment-data data/training/evidence_sensitivity_v2_train.jsonl `
  --prompt-version compact_v1 `
  --sampling balanced `
  --missing-ratio 0.33 `
  --steps 2000 `
  --seed 42 `
  --checkpoint models/evidence_sensitivity_v2b_2000_steps.pt

python src/evaluate_evidence_v2.py `
  --checkpoint models/evidence_sensitivity_v2b_2000_steps.pt `
  --group all
```

For 2,000 steps and 0.33 ratio, the sampler schedules 660 missing-evidence
updates and 1,340 leader-present updates, cycling through the corresponding
examples independently. The schedule is deterministic. The checkpoint name
is new and will not overwrite the V2 baseline checkpoint.

Compare generated-answer accuracy in training as well as all three
development categories; lower teacher-forced loss alone is insufficient.
