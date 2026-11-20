# Current experiment: V5I — Question-conditioned evidence readout

**Status:** Implementation committed to its own branch; local tests and training
not yet executed or validated.
**Working branch:** `experiment/question-conditioned-v5i`
**Parent:** `experiment/question-aware-ranking-v5h`; `develop` is unchanged.

## Motivation

V5H's recorded DEV results were 24/48 exact, 0/24 both-correct pairs,
0/12 correct switches, and 12 missing-target unsupported answers.
The ranking loss altered training but not the encoder or readout.
V5I tests a **model architecture change**, rather than further sampling
or dataset changes.

## Single change

Pool contextual question-byte states, then apply a small learned
question/evidence fusion before start, end and answerability readouts.
Keep encoder, tokenizer, training data, V5H ranking objective, optimizer,
schedule, decoder and evaluation definitions fixed.

- [Complete V5I plan and commands](experiments/question_aware_v5i/README.md)
- [V5I model](src/grounded_qa/models/span_model_v5i.py)
- [V5I training](experiments/question_aware_v5i/train_question_conditioned.py)
- [V5I DEV-only evaluator](experiments/question_aware_v5i/evaluate_dev.py)
- [V5I unit tests](tests/test_v5i_question_conditioning.py)
- [Preserved V5H baseline](experiments/question_aware_v5h/README.md)

## Safeguards

Use **fresh** checkpoint/report paths; never overwrite V5G/V5H results.
Only TRAIN and DEV may be used for this development experiment.
**Do not** inspect or generate reserved EVAL, tune to it, or reuse
old freeze manifests. Do not claim improvement until the new model
has actually been run and evaluated.

## Research criteria

Primary: DEV paired both-correct and correct field switching.
Guardrails: missing-target unsupported answers and false abstentions.
Secondary: start/end/answerability losses, per-field confusion,
parameters, train time, inference cost. Quantify whether adding
a small learned abstraction is worth its compute overhead.

Later investigations—recurrent Hopfield-like refinement, offline
active/sleep consolidation and cooperative agents—remain separate hypotheses.
