# V5D run summary

Checkpoint: `experiments\balanced_boundary_v5d\checkpoints\v5d_balanced_seed42_3000.pt`
Checkpoint SHA256: `2a256d2e29a9d71c72d5ffd0cdc6180089edb96fb2606fd2efcdca6287c7e165`
DEV dataset SHA256: `5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd`

## Observations (DEV only)
Exact: 21/96 (21.9%); start head: 51/96; end head: 22/96

| Group | Exact | Start head | End head | Truncated | Overextended | False abstentions |
|---|---:|---:|---:|---:|---:|---:|
| length4_vowel | 9/24 | 12/24 | 16/24 | 1 | 6 | 0 |
| length4_consonant | 4/24 | 13/24 | 0/24 | 3 | 7 | 0 |
| length5_vowel | 5/24 | 13/24 | 6/24 | 10 | 1 | 2 |
| length5_consonant | 3/24 | 13/24 | 0/24 | 10 | 1 | 2 |

## Interpretation
Record observations first. Any causal explanation is a hypothesis, not a proven mechanism.

## Next test
Compare against unchanged V5/V5B checkpoints on the **same V5D DEV records**. Report both regressions and improvements. Do not repeatedly tune on DEV.

No final holdout accessed; no model weights changed.
