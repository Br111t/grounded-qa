# Dataset design and evaluation boundaries

The project uses synthetic source records, questions, expected answers, and answerability labels. Training examples provide reference answers; evaluation should assess outputs against references without using evaluation labels to update weights.

## Data locations

- [`data/source/`](../data/source/): original shared synthetic examples.
- [`data/training/`](../data/training/): shared training splits.
- [`data/holdout/`](../data/holdout/): held-out source examples. Do not use them to train or repeatedly tune a model.
- `experiments/<experiment>/data/`: experiment-specific training or development data, documented by that experiment's README.

Some early datasets remain at historical locations in the reorganized tree; see each README for the **actual** path rather than assuming a universal location.

## Grounding rule

- If the supplied evidence supports the answer, answer from that evidence.
- For a partly supported question, identify unsupported parts as `Insufficient information.`
- If the answer is not present, answer `Insufficient information.`
- Do not rely on outside knowledge to fill absent facts.

## Evaluation interpretation

- Keep source-related questions together when defining splits to reduce leakage.
- Distinguish **training fit**, **development diagnostics**, and **final holdout** metrics.
- The V3 and V5C synthetic benchmarks use fixed-question leader-value records; passing those tasks is not evidence of general-purpose question answering.
- V5C's 96 locked examples are development-only diagnostics and must not enter training.

Read [the V5C experiment README](../experiments/span_boundary_probe_v5c/README.md) for its matched-group design and measured results.
