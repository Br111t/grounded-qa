# V5I question-conditioned readout — architectural ablation

Same V5G uniform train/dev, V5H ranking loss, schedule, optimizer and decoder.
Loss held fixed relative to V5H: BCE + (start CE + end CE)/2 + 
rank_weight * max(0, margin - (requested joint score - rival joint score)).
Ranking applies only to answerable rows containing BOTH Leader and Location.
Topic is never a ranking candidate. DEV is diagnostic; reserved EVAL unused.

## Configuration

- Seed: 42; updates: 1200; Adam LR: 0.001
- Ranking weight: 0.5; margin: 1.0
- Parameters: 132931; train SHA256: 493d939ab6207799536670e1b90dfa07e67e08265274423fe87d7e96348771b3
- Total elapsed seconds (includes diagnostics): 11.62
- Optimization seconds (excludes diagnostics): 9.53
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
| 0 | train | 0.632 | 4.329 | 4.692 | 17/48 | 0/24 | 0/12 | 0 | 12 |
| 0 | dev | 0.631 | 4.362 | 4.764 | 17/48 | 0/24 | 0/12 | 0 | 12 |
| 120 | train | 0.576 | 0.850 | 2.109 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 120 | dev | 0.576 | 3.586 | 2.326 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | train | 0.579 | 0.928 | 2.297 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 240 | dev | 0.579 | 3.791 | 3.326 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | train | 0.583 | 0.884 | 1.860 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 360 | dev | 0.584 | 4.204 | 2.240 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | train | 0.595 | 0.791 | 1.677 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 480 | dev | 0.596 | 5.164 | 2.313 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | train | 0.579 | 0.670 | 1.693 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 600 | dev | 0.579 | 5.823 | 1.945 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 720 | train | 0.595 | 0.865 | 1.313 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 720 | dev | 0.595 | 5.537 | 2.175 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 840 | train | 0.687 | 0.685 | 2.023 | 25/48 | 1/24 | 1/12 | 0 | 12 |
| 840 | dev | 0.686 | 5.456 | 3.363 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 960 | train | 0.615 | 0.014 | 0.880 | 34/48 | 10/24 | 10/12 | 0 | 12 |
| 960 | dev | 0.614 | 4.978 | 1.604 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1080 | train | 0.529 | 0.117 | 2.503 | 31/48 | 7/24 | 7/12 | 0 | 12 |
| 1080 | dev | 0.527 | 5.780 | 4.353 | 24/48 | 0/24 | 0/12 | 0 | 12 |
| 1200 | train | 0.518 | 0.004 | 1.328 | 35/48 | 11/24 | 11/12 | 0 | 12 |
| 1200 | dev | 0.514 | 4.515 | 2.535 | 24/48 | 0/24 | 0/12 | 0 | 12 |

## Interpretation guardrails

- Compare DEV paired both-correct, correct switches, and BOTH abstention error types.
- Higher field margins alone do not establish accuracy or calibration.
- The checkpoint is new. Existing V5G checkpoints and reports are not changed.
- This does not validate paraphrases or Topic transfer.
- Model adds a learned question-conditioned fusion; compare its parameters and runtime with V5H.
- Wall time includes machine load and diagnostic cadence; V5G's prior report
  lacks timing, so do not claim measured speed improvements over that run.
- Do not generate reserved EVAL or use any prior freeze for this checkpoint.
