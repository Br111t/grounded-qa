# V5H — Question-aware paired-ranking loss ablation

**Status: code added; training and validation results not yet claimed.**

V5H tests whether the current V5 Transformer can learn to *select the
question-requested field* without adding parameters, attention heads, layers,
or recurrent refinement. It is a controlled follow-up to
[V5G's uniform-question sampling ablation](../question_aware_v5g/README.md).
The V5G checkpoints and saved reports are immutable baselines.

## Motivation: completed V5G DEV results

The original V5G and equal-exposure V5G run both obtained 24/48 DEV exact
accuracy and 0/24 paired both-correct. The equal-exposure run made 0/12 correct
Leader/Location switches, falsely abstained 0 times, but returned unsupported
answers for all 12 missing-target questions. On the 12 complete-source pairs,
relative Leader-vs-Location scores shifted in the requested direction on 11
pairs, but Location was preferred under both questions for every pair.

These observations suggest **field-ranking bias**, but they do not prove the
cause. A learning-objective ablation is cheaper and more informative than
immediately scaling the network.

## Preregistered control

Keep fixed relative to the completed *V5G uniform* experiment:

- The same V5 Span Transformer config, tokenizer and **complete-field decoder**.
- The same registered V5G TRAIN and DEV datasets (48 rows each), seed 42.
- The same uniform shuffled-question schedule, 1,200 Adam updates, LR 0.001,
  gradient clipping at 1.0, 0.5 answerability threshold.
- The same BCE for answerability and average start/end CE on answerable rows.
- The same DEV measurements at step 0 and every 120 training updates.
- No changes to reserved EVAL data; do **not** create or read V5G's EVAL
  partition, perform holdout-driven tuning, or reuse an old freeze manifest.

The **single intended change** is the additional ranking term below. No changes
to the model's parameter count, inference path, or decoder.

## Ranking objective

For TRAIN examples with both Leader and Location fields in the evidence:

    field_score = start_logit[field_value_start] + end_logit[field_value_end]
    rank_loss = max(0, margin - (requested_score - competing_score))
    total_loss = BCE + answerable_span_CE + rank_weight * rank_loss

Defaults chosen *before inspecting new V5H results*:

- **margin = 1.0** (raw joint-logit units)
- **rank_weight = 0.5**
- Ranking candidate spans come from the existing *syntactic complete-field
  enumerator* used by the V5G decoder; no gold answer is used to build those
  candidate spans.
- Only TRAIN gold target_field identifies which candidate is correct for the
  ranking loss. No such gold label is used during inference.
- On missing-target questions, or if a rival field is absent, ranking loss is
  exactly zero. Answerability training is unchanged.
- Topic is a distractor, not an eligible supervised target.

**Important distinction:** 24/48 TRAIN questions have both supervised fields,
so with 1,200 uniform steps exactly 600 training updates are ranking-eligible.
A 0/12 DEV switching result after this run would be evidence against this
specific objective/configuration, not against question awareness in general.

## Success criteria, no composite metric

Evaluate separately, using the existing V5G evaluation definitions:

1. **Primary:** DEV paired both-correct > 0/24 and correct switches > 0/12,
   relative to the completed V5G uniform baseline.
2. **Guardrail:** unsupported answers must improve from 12/12 while false
   abstentions do not rise sharply. Ranking does not directly optimize this.
3. **Diagnostics:** DEV Leader and Location accuracy, field confusion,
   Brier score, raw answerability BCE, start/end CE, and field-score margins.
4. **Computational cost:** parameter count, training-step CPU time, and total
   elapsed time including measurement overhead. Existing V5G run did not
   measure time, so do not claim a measured speedup/slowdown without a matched
   rerun. Inference architecture and parameter count are unchanged.

**If primary accuracy fails, do not increase model size or tune the EVAL
holdout.** Consider inspecting the relative gradient scales and loss curves,
or defining a new separately registered experiment. No single metric should
conceal unsupported answers or false abstentions.

## Run locally (Windows PowerShell)

From the repository root, with the same virtual environment used for V5G:

~~~powershell
git fetch origin
git switch experiment/question-aware-ranking-v5h
$env:PYTHONPATH = "$PWD\src"

python -m unittest discover -s tests/experiments -p "test_paired_ranking_v5h.py" -v
python -m unittest discover -s tests/experiments -p "test_uniform_sampling_v5g.py" -v
~~~

Do not proceed until the loss-contract tests pass. Keep your untracked
TRAIN/DEV data and V5G reports: they are used as read-only inputs.
If the path below already exists, change the V5H filename/directory;
existing checkpoint and report artifacts are deliberately never overwritten.

~~~powershell
python -m experiments.question_aware_v5h.train_paired_ranking `
  --steps 1200 `
  --seed 42 `
  --measure-every 120 `
  --margin 1.0 `
  --rank-weight 0.5 `
  --baseline-metrics experiments/question_aware_v5g/reports/uniform_sampling_run1/metrics.json `
  --checkpoint experiments/question_aware_v5h/checkpoints/v5h_rank_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5h/reports/rank_run1
~~~

The optional baseline metrics input is **read-only** and validates the original
TRAIN SHA, steps and seed before comparing the final DEV results.

### Recovery after a completed run fails while formatting the summary

The original V5G uniform baseline lacks a Brier-score field. An earlier
V5H reporting attempt assumed that field existed and raised
`KeyError: 'brier_score'` *after* saving the V5H checkpoint and metrics.json.
This was a **report-formatting error**, not a training failure.

**Do not repeat training** or delete any checkpoint or report directory. The
updated renderer prints `not recorded` for baseline-only metrics absent from
V5G. If `summary.md` is missing but `metrics.json` and the checkpoint exist,
run the report-only repair below. It checks the saved checkpoint SHA256 and
last step, refuses to overwrite an existing summary, and never trains or reads
reserved EVAL.

~~~powershell
git pull --ff-only

python -m unittest discover -s tests/experiments -p "test_report_repair_v5h.py" -v

python -m experiments.question_aware_v5h.train_paired_ranking `
  --repair-summary-only `
  --checkpoint experiments/question_aware_v5h/checkpoints/v5h_rank_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5h/reports/rank_run1
~~~

The recovery code uses the existing `metrics.json`; it does not re-run a
checkpoint evaluation or compute a missing baseline Brier score. V5H's Brier
score, if present, remains reported. For future fresh runs use unused
checkpoint/output paths.

Outputs:

- New checkpoint: experiments/question_aware_v5h/checkpoints/v5h_rank_seed42_steps1200.pt
- New summary: experiments/question_aware_v5h/reports/rank_run1/summary.md
- Machine-readable timeline: experiments/question_aware_v5h/reports/rank_run1/metrics.json

To inspect the complete canonical DEV confusion and question-sensitivity
results with the **unchanged V5G evaluator**, you can run a separate DEV-only
evaluation with a brand-new output directory:

~~~powershell
python -m experiments.question_aware_v5g.evaluate `
  --phase dev `
  --checkpoint experiments/question_aware_v5h/checkpoints/v5h_rank_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5h/reports/rank_dev_predictions_run1
~~~

For compatibility with that existing evaluator, checkpoint metadata retains the
V5G experiment-family identifier and adds experiment_variant =
v5h_paired_ranking. This does **not** mean the V5G weights are modified.

## Reference and computational-thinking link

The experiment follows the comparison and controlled-variable thinking in
[MITx Computational Thinking](https://github.com/Coursework-Archive/math-through-code/tree/main/mitx-computational-thinking):
prefer a small, measurable optimization before adding architectural complexity.
Ranking loss is a new ML objective, **not** code copied from the coursework.

The runtime is CPU-based, one PyTorch thread, in order to match V5G's training
setting. A short smoke test can check wiring, but its results are *not*
comparable to the registered 1,200-step study; it must use separate artifact
paths and not replace the full experiment.
