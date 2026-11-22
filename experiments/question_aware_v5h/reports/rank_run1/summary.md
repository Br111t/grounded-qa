# V5H paired-ranking loss — controlled ablation

Same V5G uniform train/dev, shuffled schedule, optimizer, model and decoder.
Only training loss changes: BCE + (start CE + end CE)/2 + 
rank_weight * max(0, margin - (requested joint score - rival joint score)).
Ranking applies only to answerable rows containing BOTH Leader and Location.
Topic is never a ranking candidate. DEV is diagnostic; reserved EVAL unused.

## Configuration

- Seed: 42; updates: 1200; Adam LR: 0.001
- Ranking weight: 0.5; margin: 1.0
- Parameters: 124675; train SHA256: 493d939ab6207799536670e1b90dfa07e67e08265274423fe87d7e96348771b3
- Total elapsed seconds (includes diagnostics): 53.49
- Optimization seconds (excludes diagnostics): 42.17
- Ranking-eligible updates: 600

## Exposures

| Field / answerability | Updates |
| --- | ---: |
| leader/answerable | 450 |
| leader/unanswerable | 150 |
| location/answerable | 450 |
| location/unanswerable | 150 |

## Whole-split DEV/ TRAIN diagnostics (dropout off)

| Step | Split | BCE | Start CE | End CE | Exact | Paired | Switches | False abstain | Unsupported |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | train | 0.663 | 4.350 | 4.714 | 16/48 | 1/24 | 0/12 | 2 | 11 |
| 0 | dev | 0.662 | 4.475 | 4.791 | 16/48 | 2/24 | 0/12 | 3 | 10 |
| 120 | train | 0.563 | 0.888 | 2.399 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 120 | dev | 0.563 | 4.303 | 2.464 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | train | 0.590 | 0.831 | 2.083 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | dev | 0.591 | 4.739 | 2.664 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | train | 0.564 | 0.790 | 1.895 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | dev | 0.563 | 4.785 | 2.182 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | train | 0.563 | 0.747 | 1.981 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | dev | 0.564 | 4.790 | 2.380 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | train | 0.565 | 0.700 | 1.832 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | dev | 0.565 | 5.463 | 2.247 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 720 | train | 0.563 | 0.345 | 1.409 | 28/48 | 4/24 | 4/12 | 0 | 12 |
| 720 | dev | 0.564 | 4.784 | 1.926 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 840 | train | 0.597 | 0.820 | 1.674 | 25/48 | 1/24 | 1/12 | 0 | 12 |
| 840 | dev | 0.597 | 4.362 | 2.335 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 960 | train | 0.584 | 0.057 | 1.279 | 34/48 | 10/24 | 10/12 | 0 | 12 |
| 960 | dev | 0.584 | 5.158 | 2.066 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1080 | train | 0.651 | 0.006 | 1.831 | 32/48 | 8/24 | 8/12 | 0 | 12 |
| 1080 | dev | 0.651 | 4.605 | 2.933 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1200 | train | 0.680 | 0.007 | 0.900 | 35/48 | 11/24 | 11/12 | 0 | 12 |
| 1200 | dev | 0.680 | 5.086 | 2.716 | 24/48 | 0/24 | 0/12 | 0 | 12 |

## Final DEV versus existing V5G uniform baseline

| Metric | V5G uniform | V5H ranking |
| --- | ---: | ---: |
| correct | 24 | 24 |
| leader_correct | 6 | 6 |
| location_correct | 18 | 18 |
| paired_both_correct | 0 | 0 |
| correct_switches | 0 | 0 |
| false_abstentions | 0 | 0 |
| unsupported_answers | 12 | 12 |
| brier_score | not recorded | 0.2141992750547758 |

## Interpretation guardrails

- Compare DEV paired both-correct, correct switches, and BOTH abstention error types.
- Higher field margins alone do not establish accuracy or calibration.
- The checkpoint is new. Existing V5G checkpoints and reports are not changed.
- This does not validate paraphrases or Topic transfer.
- Model parameters are unchanged; training FLOPs may increase modestly.
- Wall time includes machine load and diagnostic cadence; V5G's prior report
  lacks timing, so do not claim measured speed improvements over that run.
- Do not generate reserved EVAL or use any prior freeze for this checkpoint.
