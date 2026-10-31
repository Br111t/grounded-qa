# Current experiment: V5C — Span Boundary Probe

**Status:** Most recent completed read-only diagnostic; next experiment not yet selected.

**Working branch for repository organization:** `refactor/repository-organization` ([PR #2](https://github.com/Br111t/grounded-qa/pull/2)). The original V5C work used historical branch `experiment/span-boundary-probe-v5c`; do not switch branches just to run the reorganized code.

## Research question

Do V5 and V5B span extractors select the correct **end boundary** when leader names vary by length (four or five characters) and final-character class (vowel or consonant)?

V5C is **development-only and read-only**. Its 96 locked examples must **not** be used for training, model selection against a final holdout, or changes to the original V3/V5B datasets.

## Where to work

- [V5C README](experiments/span_boundary_probe_v5c/README.md) — experimental design, procedure, results, and interpretation.
- [Dataset generator](experiments/span_boundary_probe_v5c/generate_span_probe_v5c.py) — deterministic locked probe records.
- [Evaluator](experiments/span_boundary_probe_v5c/evaluate_span_probe_v5c.py) — paired checkpoint comparison.
- [Boundary distribution diagnostic](experiments/span_boundary_probe_v5c/diagnose_end_boundary_distribution_v5c.py) — training-data boundary inspection.
- [Regression tests](tests/experiments/test_span_boundary_probe_v5c.py) — dataset and evaluation checks.
- [V5 checkpoint directory](experiments/span_extractor_v5/) and [V5B checkpoint directory](experiments/variable_span_v5b/) — local weights, not stored in Git.

## Latest verified results — October 31, 2026

| V5C development group | V5 exact | V5B exact |
|---|---:|---:|
| Four characters, vowel-ending | 22/24 | 20/24 |
| Four characters, consonant-ending | 0/24 | 0/24 |
| Five characters, vowel-ending | 3/24 | 9/24 |
| Five characters, consonant-ending | 0/24 | 1/24 |
| **Total** | **25/96** | **30/96** |

Both checkpoints loaded and the read-only comparison completed; the entire regression suite passed **93 tests**. V5B improved five-character vowel exact matches, but end-boundary failures remained severe for consonant endings. This is a diagnostic observation, **not** proof of why the model fails or a final-holdout result.

## Run without retraining

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -t . -p "test_*.py" -v
python -m experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c --checkpoint experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt --compare-checkpoint experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt
```

The original checkpoint files must exist in those local folders. Consult [checkpoint and backup notes](docs/training_and_checkpoints.md). The next modeling decision should be recorded as **a separate experiment**, not silently folded into V5C.
