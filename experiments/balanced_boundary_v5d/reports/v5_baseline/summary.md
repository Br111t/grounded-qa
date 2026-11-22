# V5D run summary

Checkpoint: `experiments\span_extractor_v5\checkpoints\evidence_span_v5_3000_steps.pt`
Checkpoint SHA256: `d29a7664129364d81cce8fb875db92296338ce1a2d1c77e7098062a9ae8b88a8`
DEV dataset SHA256: `5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd`

## Observations (DEV only)
Exact: 7/96 (7.3%); start head: 19/96; end head: 34/96

| Group | Exact | Start head | End head | Truncated | Overextended | False abstentions |
|---|---:|---:|---:|---:|---:|---:|
| length4_vowel | 6/24 | 6/24 | 24/24 | 18 | 0 | 0 |
| length4_consonant | 0/24 | 6/24 | 0/24 | 12 | 6 | 0 |
| length5_vowel | 1/24 | 4/24 | 10/24 | 16 | 3 | 0 |
| length5_consonant | 0/24 | 3/24 | 0/24 | 12 | 3 | 0 |

## Interpretation
Record observations first. Any causal explanation is a hypothesis, not a proven mechanism.

## Next test
Compare against unchanged V5/V5B checkpoints on the **same V5D DEV records**. Report both regressions and improvements. Do not repeatedly tune on DEV.

No final holdout accessed; no model weights changed.
