# Grounded-QA

**Small language models from scratch · Evidence-grounded question answering · Controlled AI evaluation**

Grounded-QA is an independent Python/PyTorch research project exploring whether compact models can learn to **select answers from supplied evidence, respond to the question actually asked, and abstain when evidence is missing**. Its broader research direction concerns abstraction, efficient representations, generalization, and architectural changes that might improve capability without relying exclusively on scale.

> **Research question:** Can more capable intelligence emerge through better abstractions, efficient representations, and iterative reasoning rather than simply scaling computation?

Experiments use small synthetic datasets and purpose-built Transformers, copying models, and span extractors. **Results are exploratory, not general-purpose QA benchmark claims.** No pretrained model is required for the recorded experiments.

## Research progression

| Stage | Question explored | Research record |
| --- | --- | --- |
| Foundations–V3 | Can a compact model learn to use evidence rather than memorized patterns? | [History](docs/history/learning_journal.md) |
| V4–V4B | Does source copying improve faithfulness to evidence? | [V4B](experiments/evidence_copy_v4b/README.md) |
| V5–V5C | Where do extractive answer boundaries fail? | [V5C](experiments/span_boundary_probe_v5c/README.md) |
| V5D | Can balanced boundary training and diagnostics isolate span errors? | [V5D](experiments/balanced_boundary_v5d/README.md) |
| V5E | How much does the field-aware decoder change extraction? | [V5E](experiments/field_boundary_decoder_v5e/README.md) |
| V5F | Does a model change its answer when only the question changes? | [V5F](experiments/counterfactual_questions_v5f/README.md) |
| V5G | Can one question-aware model select Leader versus Location and abstain? | [V5G](experiments/question_aware_v5g/README.md) |
| V5H | Does paired field-ranking loss improve correct switching? | [V5H](experiments/question_aware_v5h/README.md) |
| V5I–V5J | Do question-conditioned interactions improve field selection? | [Six-variant comparison](experiments/question_aware_v5i/reports/six_variant_comparison_2026-11-21.md) |
| V5K–V5M | Can frozen-model answerability scoring improve abstention without retraining the encoder? | [Current research summary](CURRENT_EXPERIMENT.md) |

## What the results show

In a controlled, **single-seed exploratory DEV comparison**, the V5J bilinear interaction reached **36/48 exact** and **12/12 correct complete-source question switches**, compared with **24/48** and **0/12** for the listed V5I activation variants. It still made **12/12 unsupported-answer errors** on missing-target DEV questions. [Study and limitations](experiments/question_aware_v5i/reports/six_variant_comparison_2026-11-21.md)

Later TRAIN-fitted answerability experiments using a frozen V5J bilinear model improved missing-target abstention on the *same repeatedly inspected DEV split*. V5K recorded **45/48 exact** and V5M **44/48**, with **0 unsupported-answer errors** in those recorded runs. These findings **do not establish generalization**; the evaluation distribution and sealed EVAL remain important boundaries. [V5K summary](experiments/question_aware_v5i/reports/v5k_joint_a_prime_run1/summary.md) · [V5M summary](experiments/question_aware_v5i/reports/v5m_learned_null_run1/summary.md)

## Explore the project

- [Current experiments, results, and caveats](CURRENT_EXPERIMENT.md)
- [Architecture](docs/architecture.md)
- [Dataset design and split boundaries](docs/dataset_design.md)
- [Training and checkpoint guidance](docs/training_and_checkpoints.md)
- [Original learning journal](docs/history/learning_journal.md)

## Run tests

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

**Local verification reported November 22, 2026:** 174 tests passed, 110 subtests passed, 1 PyTorch nested-tensor warning, in 10.94 seconds on `experiment/question-conditioned-v5i`. This is a reported local result, not an independently executed CI run.

Model checkpoints are kept out of Git. Generated reports and selected datasets are preserved for reproducibility. Do not overwrite historical reports or tune using the reserved EVAL split.

## Grounding contract

- **Answerable:** return the answer supported by evidence.
- **Partially answerable:** answer supported portions; mark unsupported portions `Insufficient information.`
- **Unanswerable:** return `Insufficient information.`
- **No outside inference** as a substitute for supplied evidence.

Future directions—including recurrent memory, iterative refinement, and more compute-efficient abstractions—are **research hypotheses, not demonstrated capabilities**.
