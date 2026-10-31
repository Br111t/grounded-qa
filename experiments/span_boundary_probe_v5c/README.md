# V5C: Read-only factorial boundary probe

## Motivation

V5B fitted only 20/72 additional training rows, with end-head accuracy:

| Name length | Correct end head | Exact answers |
|---|---:|---:|
| 3 bytes | 12/24 | 13/24 |
| 5 bytes | 0/24 | 0/24 |
| 6 bytes | 7/24 | 7/24 |

The V5B length-5 training names were all consonant-ending (`n`).
That confounds length with final-character identity. Before retraining,
test BOTH saved checkpoints against a locked, matched 2x2 diagnostic.

## Fixed design

**Development only. Do not train on these 96 examples.**

| Group | Name length | Ending class | Rows |
|---|---:|---|---:|
| length4_vowel | 4 | vowel | 24 |
| length4_consonant | 4 | consonant | 24 |
| length5_vowel | 5 | vowel | 24 |
| length5_consonant | 5 | consonant | 24 |

12 base stems appear in all four cells; each stem has two exactly
matched contexts across cells (same location, topic and fact ordering).
Vowel endings rotate between `a` and `i`, consonant endings between
`n` and `t` (6 stems per pair). The 5-character variants insert
the same internal `e` before the final character, and therefore differ
from the 4-character variants only in length and that insertion.

All complete names are disjoint from the original V3 training and
development, AND V5B extra-training and development partitions. Data
is deterministic and locked to seed 42; rerunning the generator is safe
only when files exactly match expected content. These are synthetic
names, not a real-world language benchmark.

Although this is much better balanced than the old length comparison,
some effects of character identity, insertion, and name-stem composition
remain. One seed and 24 cases per cell cannot establish causality.

## Run the read-only diagnostic (PowerShell, repository root)

Use the reorganized branch and confirm local checkpoint paths:

    git status --short
    $env:PYTHONPATH = "$PWD\src"
    Get-ChildItem experiments/span_extractor_v5/checkpoints/*.pt
    Get-ChildItem experiments/variable_span_v5b/checkpoints/*.pt

Run the regression suite:

    python -m unittest discover -s tests -t . -p "test_*.py" -v

All four locked V5C JSONL files should already exist in
`experiments/span_boundary_probe_v5c/data/`.
Do not regenerate or overwrite them during checkpoint evaluation.

Evaluate the original saved V5 and V5B checkpoints on the same locked records:

    python -m experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c --checkpoint experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt --compare-checkpoint experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt

Those checkpoint locations reflect the planned local organization; adjust the
paths to the actual files if necessary. Weights remain ignored by Git.
The evaluator accepts the exact registered historical dataset paths embedded
in older checkpoints, as well as the updated paths. It does not retrain.

The evaluator reports, for each of four cells, full-answer exact match,
independent start/end head accuracy, exact selected spans, truncated/
overextended spans, false abstentions, and a first-versus-second
checkpoint comparison. Aggregated results show length 4 vs 5 and
vowel vs consonant endings separately.

**No new model training, checkpoint creation, model weight changes,
existing V3/V5B data modifications, or final holdout evaluation.**

## Recorded read-only comparison (October 31, 2026)

The 93-test regression suite passed. Both existing local checkpoints loaded
successfully and were compared on the same 96 V5C development examples.

| Diagnostic group | V5 exact | V5B exact |
|---|---:|---:|
| length4_vowel | 22/24 | 20/24 |
| length4_consonant | 0/24 | 0/24 |
| length5_vowel | 3/24 | 9/24 |
| length5_consonant | 0/24 | 1/24 |
| **All four groups** | **25/96** | **30/96** |

V5B improved five-character vowel exact matches, but consonant-ending
results remained poor. Independent start-head accuracy was 84/96 for V5
and 92/96 for V5B, while end-head accuracy was 26/96 for V5 and 28/96
for V5B. These findings suggest an end-boundary generalization weakness;
they do **not** establish causality or final-holdout performance.
No weights or locked dataset records were changed by this comparison.

## Interpretation

- If five-character spans fail for BOTH vowel and consonant endings,
  length/generalized end-boundary selection is the leading concern.
- If consonant endings fail at BOTH lengths, a terminal-character
  preference is more likely involved.
- If both effects appear, test more varied training data or a supervised
  boundary objective, but do not tune repeatedly on these 96 examples.
- Also compare head accuracy to selected-span exact match: the decoder
  jointly chooses a valid start/end span, not independent argmax positions.

Files: `experiments/span_boundary_probe_v5c/generate_span_probe_v5c.py`,
`experiments/span_boundary_probe_v5c/evaluate_span_probe_v5c.py`,
and `tests/experiments/test_span_boundary_probe_v5c.py`.
