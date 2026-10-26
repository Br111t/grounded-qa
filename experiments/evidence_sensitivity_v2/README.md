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
