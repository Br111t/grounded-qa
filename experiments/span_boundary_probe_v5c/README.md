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

## Steps from repository root (PowerShell)

Pull the new branch after checking for uncommitted work:

    git status --short
    git fetch origin
    git switch --track origin/experiment/span-boundary-probe-v5c

Run regression tests before generating data:

    python -m unittest discover -s tests -p "test_span_boundary_probe_v5c.py" -v

Generate ONLY the four new V5C development files:

    python src/generate_span_probe_v5c.py

Evaluate V5 and V5B with identical locked probe records in one read-only run:

    python src/evaluate_span_probe_v5c.py --checkpoint models/evidence_span_v5_3000_steps.pt --compare-checkpoint models/evidence_span_v5b_3000_steps.pt

The evaluator reports, for each of four cells, full-answer exact match,
independent start/end head accuracy, exact selected spans, truncated/
overextended spans, false abstentions, and a first-versus-second
checkpoint comparison. Aggregated results show length 4 vs 5 and
vowel vs consonant endings separately.

**No new model training, checkpoint creation, model weight changes,
existing V3/V5B data modifications, or final holdout evaluation.**

## Interpretation

- If five-character spans fail for BOTH vowel and consonant endings,
  length/generalized end-boundary selection is the leading concern.
- If consonant endings fail at BOTH lengths, a terminal-character
  preference is more likely involved.
- If both effects appear, test more varied training data or a supervised
  boundary objective, but do not tune repeatedly on these 96 examples.
- Also compare head accuracy to selected-span exact match: the decoder
  jointly chooses a valid start/end span, not independent argmax positions.

Files: `src/generate_span_probe_v5c.py`, `src/evaluate_span_probe_v5c.py`,
and `tests/test_span_boundary_probe_v5c.py`.
