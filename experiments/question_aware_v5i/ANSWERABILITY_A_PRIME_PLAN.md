# V5K preregistration — answerability residual a′ using end-head evidence

**Status:** hypothesis and diagnostic gate, not a trained result. Do not modify V5J model, checkpoint, DEV data or sealed EVAL.

## Baseline

The completed V5J bilinear model obtained 36/48 DEV exact, 12/12 correct switches on complete-source pairs and **12/12 unsupported-answer errors**. The latter means all missing-target DEV questions were answered incorrectly.

Current model:

`start_i = start_head(H_i) + H_iᵀW_s q`

`end_i = end_head(H_i) + H_iᵀW_e q`

`a = answerability_head(H_BOS)`

Prediction answers if `sigmoid(a) >= 0.5`; otherwise abstains.

## Proposed hypothesis

Does a small learned feature of the existing end head improve the model's decision to abstain when the requested field is absent, **without sacrificing present-field extraction accuracy**?

`a′ = a + λ E(q,H)`

The first preregistered candidate `E` should be a label-free, question-dependent *relative end-score feature* computed from the evidence end logits, not an oracle score at the gold requested field. Candidates under consideration are `max_end - mean_end` and the top-two end score gap. Endpoints used for candidate-field features must come from the existing syntax-only enumerator and must be made available during training and inference in exactly the same way.

Use `λ=0` initially and learn it with answerability BCE on TRAIN. This preserves baseline answerability at initialization. Because scores are uncalibrated, do not hardcode a DEV-derived cutoff; report the unmodified 0.5 probability threshold first.

**Critical caveat:** Raw end logits are not probabilities or proof that the requested field exists. In the present model, end heads score *something* even for missing requested fields. An input-dependent question-to-end feature may still be unable to separate answerable vs unanswerable cases.

## Diagnostic gate before implementing a′

Run `diagnose_bilinear_answerability.py` against the immutable `v5j_interaction_bilinear_seed42_steps1200.pt` checkpoint, using only registered TRAIN and DEV, and inspect:

- answerability logit/probability
- maximum candidate end logit, top-two end-logit gap, end max-minus-mean
- maximum candidate joint logit, joint top-two gap and max-minus-mean
- per-class distributions and exploratory ROC AUC separately on TRAIN and DEV

An AUC near 0.5 means the feature is poorly discriminative. A large TRAIN AUC with poor DEV AUC suggests a dataset-specific relationship, not a robust abstention signal. All comparisons are exploratory; do not optimize feature choice on DEV and then call DEV unbiased performance.

If none of the end-derived features discriminate, **do not add a′ just to make the equation more complex**. Instead diagnose the answerability loss, missing-target supervision and question representation.

## Controlled V5K training (only after assessing diagnostic)

- Preserve V5J bilinear encoder and start/end scoring, original registered F-initial TRAIN/G-initial DEV dataset, seed 42, 1,200 updates, Adam 0.001, uniform question exposure, ranking loss and decoder.
- Compare baseline bilinear to a new bilinear+a′ model, measuring full DEV outcomes, not only answerability.
- Train the new answerability coefficient on TRAIN. New checkpoint/report paths, no overwrites.
- Track answerable accuracy, missing-target abstention, false abstentions, complete-pair correct switches, answerability Brier/BCE, span CE, optimization time and any end-gradient changes.
- The original bilinear run is a reference, but a fully matched from-scratch comparison should also rerun the baseline with the same code path and random seed.
- The subsequent easier mixed-initials TRAIN/DEV experiment is **separate**, so no dataset change can masquerade as a gain from a′.
- Keep EVAL sealed until a full hypothesis, architecture, decoder, checkpoint and acceptance conditions are frozen.

## Commands

From repository root in PowerShell:

```powershell
git switch experiment/question-conditioned-v5i
git pull --ff-only
$env:PYTHONPATH = "$PWD\\src"
python -m experiments.question_aware_v5i.diagnose_bilinear_answerability `
  --checkpoint experiments/question_aware_v5i/checkpoints/v5j_interaction_bilinear_seed42_steps1200.pt `
  --output-dir experiments/question_aware_v5i/reports/bilinear_answerability_separation_run1
```

Then inspect `summary.md` and share the table before deciding which score feature to couple into `a′`.
