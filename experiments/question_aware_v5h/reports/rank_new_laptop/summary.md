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
- Total elapsed seconds (includes diagnostics): 12.66
- Optimization seconds (excludes diagnostics): 10.54
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
| 120 | train | 0.562 | 0.881 | 2.373 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 120 | dev | 0.562 | 4.188 | 2.406 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | train | 0.567 | 0.856 | 2.117 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | dev | 0.567 | 4.683 | 2.455 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | train | 0.562 | 0.709 | 1.810 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | dev | 0.562 | 3.981 | 2.126 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | train | 0.573 | 0.765 | 1.845 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | dev | 0.573 | 4.623 | 2.144 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | train | 0.568 | 0.667 | 1.646 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | dev | 0.568 | 5.518 | 2.052 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 720 | train | 0.568 | 0.341 | 1.579 | 26/48 | 2/24 | 2/12 | 0 | 12 |
| 720 | dev | 0.568 | 4.297 | 2.298 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 840 | train | 0.605 | 0.545 | 1.660 | 26/48 | 2/24 | 2/12 | 0 | 12 |
| 840 | dev | 0.605 | 4.513 | 2.279 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 960 | train | 0.594 | 0.032 | 1.006 | 35/48 | 11/24 | 11/12 | 0 | 12 |
| 960 | dev | 0.595 | 4.716 | 1.918 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1080 | train | 0.643 | 0.006 | 1.644 | 34/48 | 10/24 | 10/12 | 0 | 12 |
| 1080 | dev | 0.643 | 4.391 | 2.926 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1200 | train | 0.678 | 0.321 | 1.419 | 29/48 | 5/24 | 5/12 | 0 | 12 |
| 1200 | dev | 0.678 | 4.157 | 3.780 | 24/48 | 0/24 | 0/12 | 0 | 12 |

## Interpretation guardrails

- Compare DEV paired both-correct, correct switches, and BOTH abstention error types.
- Higher field margins alone do not establish accuracy or calibration.
- The checkpoint is new. Existing V5G checkpoints and reports are not changed.
- This does not validate paraphrases or Topic transfer.
- Model parameters are unchanged; training FLOPs may increase modestly.
- Wall time includes machine load and diagnostic cadence; V5G's prior report
  lacks timing, so do not claim measured speed improvements over that run.
- Do not generate reserved EVAL or use any prior freeze for this checkpoint.
