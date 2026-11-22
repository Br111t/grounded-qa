# V5D run summary

Checkpoint: `experiments\variable_span_v5b\checkpoints\evidence_span_v5b_3000_steps.pt`
Checkpoint SHA256: `616e4e5e3324655a8a93a3bf4f73c1825e32ebed8b88f18bbbe393d4d828f2c1`
DEV dataset SHA256: `5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd`

## Observations (DEV only)
Exact: 32/96 (33.3%); start head: 84/96; end head: 38/96

| Group | Exact | Start head | End head | Truncated | Overextended | False abstentions |
|---|---:|---:|---:|---:|---:|---:|
| length4_vowel | 21/24 | 21/24 | 24/24 | 3 | 0 | 0 |
| length4_consonant | 0/24 | 21/24 | 0/24 | 16 | 8 | 0 |
| length5_vowel | 11/24 | 21/24 | 14/24 | 11 | 2 | 0 |
| length5_consonant | 0/24 | 21/24 | 0/24 | 22 | 2 | 0 |

## Interpretation
Record observations first. Any causal explanation is a hypothesis, not a proven mechanism.

## Next test
Compare against unchanged V5/V5B checkpoints on the **same V5D DEV records**. Report both regressions and improvements. Do not repeatedly tune on DEV.

No final holdout accessed; no model weights changed.
