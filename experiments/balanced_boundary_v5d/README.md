# V5D — Balanced Boundary Training (design stage)

**Status:** Experiment registered; no new dataset, training run, or checkpoint has been created.

**Branch:** `experiment/v5d-balanced-boundaries` (branched from `experiment/v5c-boundary-analysis` to preserve the read-only diagnostic scripts).

## Why V5D exists

V5C established a systematic end-head generalization failure using 96 locked
**development-only** rows. V5/V5B frequently chose vowel bytes inside other
evidence fields instead of the end of the requested leader name. On matched
four-character names, changing only the terminal vowel to a consonant changed
the independent end-head argmax in 23/24 V5 pairs and 20/24 V5B pairs.
Those observations motivate V5D; they do **not** establish a causal mechanism.

The V5 training set supplied 144 answerable four-character vowel-ending names.
V5B added 72 answerable rows: 24 three-character vowel-ending, 24
five-character `n`-ending, and 24 six-character `a`-ending names. Length
and final character were therefore confounded.

## Preregistered research question

**Does controlling the joint distribution of name length and final-character
class improve extraction of the correct leader-name end boundary on genuinely
new names, while holding model architecture, prompts, optimizer, step count,
and evaluation procedure constant?**

Primary hypothesis: a balanced training design improves independent end-head
accuracy for *unseen consonant-ending names*, including both four and five
characters, compared with the existing V5B checkpoint.

An improvement is not proof that the model learned a general delimiter rule.
Inspect region of predicted end and verify no regression on vowel-ending names.

## Dataset design — proposed, not yet generated

Generate a NEW V5D training addition with a factorial 2 x 2 design.
Prefer equal **unique-name** counts and equal **row** counts per cell, rather
than repeating a small set of names disproportionately.

| Training cell | Distinct names (target) | Rows (target) | Final letters |
|---|---:|---:|---|
| length4_vowel | 8 | 24 | `a`, `i` |
| length4_consonant | 8 | 24 | `n`, `t` |
| length5_vowel | 8 | 24 | `a`, `i` |
| length5_consonant | 8 | 24 | `n`, `t` |
| **Addition** | **32** | **96** | |

Three varied evidence contexts per name, sampling both `Leader`-before-
`Location` and `Location`-before-`Leader` orders; topic presence should
not correlate with any cell. Use fixed seed and explicit count assertions.
This is an **addition to V3 TRAIN**, not a rewrite of V3 or V5B data.

**Before implementing**, independently reserve an untouched V5D development
partition using different complete names and preferably distinct name stems;
ensure no name or identical evidence overlaps with V3 train, V5B train, V5C
probes, or V5D train. Develop/test partitions must never be read by the trainer.
Use the same 2 x 2 cells, equal sizes, and a separately recorded seed.
Do not copy V5C's 96 cases into training: they have already been inspected
repeatedly, making them exploratory rather than unbiased model-selection data.

The final holdout stays untouched. No test-set performance claims until
training and evaluation choices are frozen.

## Controlled comparison plan

1. **Baseline:** Existing V5 and V5B 3,000-step checkpoints, kept read-only.
2. **V5D candidate:** Same `SpanGroundedQA` architecture, byte tokenizer,
   `compact_v1` prompt, end/start cross-entropy objective, threshold 0.5,
   Adam learning rate 0.001, CPU/batch size 1, and declared seed.
3. **Training initialization:** Start from fresh seeded weights if comparing
   training recipes like-for-like. V5/V5B both trained from scratch; do not
   label a warm-start run as directly comparable. Keep checkpoint lineage
   and optimizer-state limitations explicit if later testing warm starts.
4. **Training schedule:** 3,000 updates with a fixed missing-example ratio
   matching V5B (0.33); document how adding 96 answerable rows changes
   sampling frequency and total dataset size. This compares **recipes** and
   does not isolate balance from the larger/additional training set.
5. **To test balance specifically:** Plan a separate matched-data-budget
   control (equal new-row count, answerability balance, step count, and
   context distribution), rather than attributing the change to balancing
   alone. Predefine this control before any outcome-dependent revisions.
6. **Metrics:** Exact extraction; independent start/end-head accuracy;
   correct end in top 5; selected-span accuracy; answerability error;
   predicted end location (name/Location/Topic/punctuation); results per
   length, ending type, and `n` vs `t`; total counts.
7. **Freeze evaluations:** Use an independent V5D development set for the
   planned comparison. V5C remains a historical exploratory probe; never
   train on it or repeatedly optimize against it.

## Safeguards and implementation gates

- Generator must be deterministic and refuse to overwrite different files.
- All row labels must reconstruct the exact evidence bytes.
- Enforce split isolation of names, evidence, and IDs.
- New training code must only accept explicitly registered training paths;
  reject any development or holdout paths before constructing the optimizer.
- Save under a **new** `.pt` filename; never overwrite existing weights.
- Record training dataset paths/hashes, seed, steps, architecture, prompt,
  optimizer/scheduler configuration, initialization and checkpoint lineage.
- Verify save/load predictions and retain regression tests.
- Do not retrain until the dataset generator, validation tests, and comparison
  protocol have been reviewed.

## Immediate next step

Implement **only** a deterministic training/development data builder and
tests for the locked 2 x 2 distribution and isolation rules. Inspect its
summary before enabling any training command.

## Existing evidence and code

- `experiments/span_boundary_probe_v5c/README.md`
- `experiments/span_boundary_probe_v5c/diagnose_end_boundary_distribution_v5c.py`
- `experiments/span_boundary_probe_v5c/diagnose_end_boundary_errors_v5c.py`
- `experiments/span_boundary_probe_v5c/inspect_paired_end_logits_v5c.py`
- `experiments/span_boundary_probe_v5c/inspect_end_target_characters_v5c.py`
- `experiments/variable_span_v5b/train_span_v5b.py`
- `experiments/variable_span_v5b/generate_span_data_v5b.py`
