# V5F — Counterfactual question sensitivity (read-only diagnostic)

## Why

V5E showed that restricting answer spans to complete fields improves boundary
selection, but the models were predominantly trained with a single question:
"What is the leader of Workshop W0001?". High Leader accuracy does not establish
that the question affects which field is selected.

**Hypothesis under investigation:** when evidence stays byte-identical and only
the question changes, the model's start/end logits and preferred complete
field may respond to the requested field. This is exploratory; the checkpoints
were **not trained to answer the Location, Topic, or unrelated questions**.

## Design (registered before inference)

4 evidence passages × 6 fixed questions × V5/V5B/V5D = 72 predictions.
Each passage contains exactly 3 complete fields, with differing field orders
and values. The questions are:

| Variant | Question | Intended behavior |
|---|---|---|
| leader_original | What is the leader of Workshop W0001? | Leader value |
| leader_paraphrase | Who leads Workshop W0001? | Same Leader value |
| leader_reworded | Which person is in charge of Workshop W0001? | Same Leader value |
| location | Where is Workshop W0001 located? | Location value |
| topic | What is the topic of Workshop W0001? | Topic value |
| unrelated | What will the weather be tomorrow? | Abstain |

One control point matters: the *evidence is unchanged*, including byte offsets,
for every question on that passage. Only the question tokens change. The
probe tests field selection, not whether free-text answer generation works.
For each question it records all candidate field start/end logits, joint
scores, selected field, score margin, answerability probability and reference
field. For paired comparisons against the original Leader question it records
per-field score deltas and whether the selection changed.

**Interpretation:** Correct selection under paraphrasing is different from
correctly switching from Leader to Location/Topic. A change in score alone
does not prove semantic understanding. The unrelated question tests whether
the answerability head refuses an unsupported request, but it was trained
primarily around Leader-present/Leader-missing evidence, so failure is not
surprising. This tiny, partly out-of-distribution diagnostic is not a new
generalization benchmark.

## Run

From repository root in PowerShell:

```powershell
git fetch origin
git switch experiment/counterfactual-questions-v5f
$env:PYTHONPATH = "$PWD\src"

python -m unittest tests.experiments.test_counterfactual_questions_v5f -v

python -m experiments.counterfactual_questions_v5f.probe_questions `
  --output-dir experiments/counterfactual_questions_v5f/reports/probe_run1
```

Outputs are immutable and reproducible:
- `all_predictions.json` — all 72 predictions, candidate field scores,
  selected field and answerability probabilities
- `paired_deltas.json` — all 60 question changes compared with the same
  evidence and original Leader question, including signed logit score deltas
- `metrics.json` — checkpoint hashes, experiment and code provenance,
  selected-field confusion counts
- `summary.md` — selected fields, counts by question and target score advantages

No training, checkpoint rewriting, V5E holdout reuse, decoder change, or
new held-out accuracy claim. Keep the original V5E branch and artifacts
untouched. Do not train on the questions until these diagnostics are reviewed.

## Decision after analysis

If the selected field stays Leader regardless of question, that is evidence
of a *fixed-field prior* under these conditions. If logits shift but the
wrong field still wins, the model exhibits question sensitivity without
reliable question-aware selection. If paraphrases produce unstable choices,
that reveals sensitivity to wording. If the model correctly switches between
fields, test with independently designed new evidence and paraphrases before
claiming meaningful question understanding.

Possible subsequent training experiments are separate V5G work and must use
fresh evaluation material. V5F itself intentionally does not modify weights.
