# V5D failure-pattern hypotheses and logit investigation

**Status:** Read-only descriptive investigation; hypotheses are **not**
confirmed. Model checkpoints and existing datasets are unchanged.

## Evidence recorded from V5D DEV (96 matched examples per checkpoint)

| Model | Exact | Independent start | Independent end |
|---|---:|---:|---:|
| V5 | 7/96 | 19/96 | 34/96 |
| V5B | 32/96 | 84/96 | 38/96 |
| V5D | 21/96 | 51/96 | 22/96 |

Every model has **0/48** independently correct end positions for
consonant-ending names. V5D nevertheless produces **7/48** exact final
answers for consonant-ending names through joint span selection.

The earlier boundary report records:

| Model | Raw end wrong; selected span correct | Raw end correct; selected span wrong |
|---|---:|---:|
| V5 | 0 | 27 |
| V5B | 0 | 6 |
| V5D | 11 | 12 |

Selected individual records from the saved `predictions.csv`:

- **Decoder recovery (V5D):** `v5d_dev_length4_consonant_11_0_q01`,
  expected `Leader: Dblt.`, final answer `Leader: Dblt.`;
  independent raw end delta **-22** but selected end delta **0**.
- **Correct raw end, wrong answer (V5D):**
  `v5d_dev_length4_vowel_03_0_q01`, expected `Leader: Dbdi.`,
  predicted `Leader: di.`; raw end delta **0**, selected end delta
  **0**, raw start delta **+6**. Correct end does not guarantee
  correct answer start.
- **Wrong evidence field (V5D):**
  `v5d_dev_length5_consonant_10_1_q01`, expected
  `Leader: Dbken.`, predicted `Leader: Pine. Leader: D.`;
  independent raw end delta **-22**, selected end delta **-4**.

**Do not infer causality** from any of these descriptive cases.

## Explicit hypotheses to investigate

**H1 — Terminal-character / boundary-cue dependence.**
The raw end head could be relying on learned final-character and punctuation
patterns instead of a transferable entity boundary. Evidence supporting
investigation: 0/48 independently correct consonant endings in each model.
Alternative: the end head could be ranking other source positions for reasons
unrelated to final-character identity.

**H2 — End scores contain useful subordinate candidates.**
The correct end may be ranked below the top raw end while its score is high
enough to win after adding a compatible start score. Evidence: 11 V5D
cases with wrong raw end but correctly selected span. Test: gold-end
rank, score gap to top end, and highest-ranked joint spans.

**H3 — Start/end score interaction causes additional failures.**
Even a correct raw end can be paired with an incorrect start or lose in joint
scoring. Evidence: 12 V5D cases with correct raw end but wrong selected
span. Test: start-head rank, top joint spans and their component scores.

**H4 — Changed training shifts shared representations.**
The V5B-to-V5D degradation in start accuracy could reflect training
composition, initialization, sampling, or gradient interference. This
diagnostic cannot distinguish these. A future controlled training
study would require matched seeds, counts and steps, and potentially
gradient measurements.

**H5 — Synthetic distribution shift contributes.**
V5D training names start with `C`, development names with `D`;
these partitions share other character and context regularities.
Length/ending effects therefore are not fully isolated.

## Read-only logit diagnostic

Run from the repository root in PowerShell after fetching the branch:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest tests.experiments.test_v5d_logits -v
python -m unittest discover -s tests -t . -p "test_*.py" -v

python -m experiments.balanced_boundary_v5d.diagnose_logits `
  --v5 experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt `
  --v5b experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt `
  --v5d experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt `
  --output-dir experiments/balanced_boundary_v5d/reports/logit_probe_run1
```

Outputs are immutable and include `all_cases.json` (all 288
prediction-level records, top five raw start/end candidates, top five
joint spans, gold boundary ranks, and raw/joint comparisons),
`metrics.json` (checkpoint/data SHA-256 identities), and `summary.md`
(overall ranks and deterministic representative examples).

**Reading the output:** The gold-boundary rank measures how well the
correct end scores among permitted evidence positions. Joint spans are
ranked by `start_logit + end_logit` with V5's maximum span byte limit.
These scores are not calibrated probabilities. All positions are inclusive
byte-token offsets (not Python character indices).

**Decision rule:** Review rankings and margins before proposing any new
data, new loss weighting, or architecture change. Preserve V5C and avoid
repeatedly tuning against this DEV set. No final holdout is accessed.


## Primary diagnostic reporting fix (November 2026)

The main `diagnose_logits.py` now calls the same `enrich` and `measure`
functions as `reconcile_logits.py`. Each example has independent booleans
for exact prediction, correct raw end, correct selected span, recovered wrong
raw end, lost correct raw end and abstention. An exact answer can also be a
recovery. The Markdown overview is derived from these enriched records and
also includes end and joint logit margins. `metrics.json` includes the
same overview in its `counts` field, while `all_cases.json` has each
enriched case.

The original `error_category` field remains as a legacy grouping label;
**never count recovery from that exclusive field**.

For new runs, use a new immutable output directory (for example
`reports/logit_probe_run2`). `reconcile_logits.py` remains useful only
for converting prior archived reports. Regression coverage is in
`tests/experiments/test_v5d_logits.py` and
`tests/experiments/test_v5d_reconcile_logits.py`.
