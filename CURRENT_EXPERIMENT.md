# Grounded-QA — Current research status

**Updated:** November 22, 2026  
**Branch:** `experiment/question-conditioned-v5i`  
**Stage:** V5I–V5M exploratory experiments recorded; next controlled study not yet selected

## Research objective

Investigate whether compact, question-aware models can reliably select evidence-supported answers, switch between requested fields, and abstain when the requested evidence is absent. Longer-term ideas include efficient abstractions, structured memory, and iterative refinement; these remain hypotheses.

## Recorded progression and results

| Experiment | Intervention or diagnostic | Recorded exploratory DEV result |
| --- | --- | --- |
| V5H | Paired ranking loss | 24/48 exact; 0/12 correct switches; 12 unsupported-answer errors |
| V5I | Question-conditioned fusion; tanh, GELU, linear controls | Each 24/48 exact; 0/12 correct switches; 12 unsupported-answer errors |
| V5J bilinear | Direct question-dependent bilinear span scoring | 36/48 exact; 12/12 switches; **12 unsupported-answer errors** |
| V5K | TRAIN-fitted joint-score answerability adjustment on frozen V5J | 45/48 exact; 0 unsupported-answer errors; 3 false abstentions |
| V5L | Retrospective paired-question score diagnostic | Descriptive only; **not** a single-question inference approach |
| V5M | 129-parameter learned null head on frozen V5J; single-question inference | 44/48 exact; 0 unsupported-answer errors; 4 false abstentions |

**Interpret counts carefully:** `unsupported_answer_errors` measures incorrect answers to missing-target questions. **Lower is better.** The 12/12 value for V5J is a complete failure to abstain on those examples, not a success.

These results come from small synthetic Leader/Location datasets, one main training seed, and a DEV split used repeatedly for analysis. They are **not sealed-EVAL results or evidence of broad generalization**. V5K and V5M change answerability scoring on a frozen model; they do not demonstrate that a new encoder learned better general reasoning.

## Primary records

- [V5H paired-ranking results](experiments/question_aware_v5h/reports/rank_new_laptop/summary.md)
- [V5I original question-conditioned run](experiments/question_aware_v5i/reports/v5i_seed42/summary.md)
- [V5I/V5J six-variant controlled comparison](experiments/question_aware_v5i/reports/six_variant_comparison_2026-11-21.md)
- [V5K frozen bilinear + joint-score answerability](experiments/question_aware_v5i/reports/v5k_joint_a_prime_run1/summary.md)
- [V5L paired-question diagnostic](experiments/question_aware_v5i/reports/v5l_paired_questions_run1/summary.md)
- [V5M learned single-question null score](experiments/question_aware_v5i/reports/v5m_learned_null_run1/summary.md)

Historical V5C boundary diagnostics remain documented in [V5C](experiments/span_boundary_probe_v5c/README.md); they are not the latest result.

## Next research decision

Prefer **one controlled change at a time**. A separately registered dataset with matched TRAIN/DEV name-initial distributions would help examine whether the observed field-selection and answerability gains survive distribution changes. Compare bilinear and a predeclared control, preserve existing splits and artifacts, and reserve final EVAL for a pre-registered evaluation.

Potential later architecture experiments—recurrent/Hopfield-like memory, offline consolidation, or cooperative agents—should be implemented on new branches with explicit compute-cost and generalization measurements, not described as achieved outcomes.

## Validation and safeguards

The user reported running `python -m pytest tests/ -v` on this branch on November 22, 2026: **174 tests passed, 110 subtests passed, one non-failing nested-tensor warning**. This does not replace empirical model evaluation.

Keep model checkpoints out of Git, preserve recorded reports and frozen comparisons, and do not use reserved EVAL data for model selection.
