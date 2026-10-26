# Evidence sensitivity V3

Goal: Same question, diverse evidence values, and separate development tests.

Fixed question: What is the leader of Workshop W0001?

The existing Transformer, compact_v1 prompt, and balanced sampling are preserved.

Dataset groups (generated using seed 42):
- train: 168 rows, 48 names x 3 contexts plus 24 missing-leader rows
- known: 48 rows using the same leaders in new contexts
- recombined: 24 rows with 12 unseen complete names built from known syllables
- unseen: 24 rows with 12 unseen complete names built from reserved syllables
- missing: 12 rows with absent leaders and new contexts

All generated files use versioned v3 names in data/training or data/evaluation.
No original final holdout is read or changed.

PowerShell from repository root:

1. Fetch and switch branches:

    git status --short
    git fetch origin
    git switch --track origin/experiment/evidence-sensitivity-v3

2. Test and generate:

    python -m unittest discover -s tests -p "test_evidence_v3.py" -v
    python -m unittest discover -s tests -p "test_evidence_sampling.py" -v
    python src/generate_evidence_data_v3.py --seed 42

3. Train from fresh initialization:

    python src/train.py --experiment-data data/training/evidence_sensitivity_v3_train.jsonl --prompt-version compact_v1 --sampling balanced --missing-ratio 0.33 --steps 3000 --loss-every 120 --seed 42 --checkpoint models/evidence_sensitivity_v3_3000_steps.pt

4. Evaluate training fit and development categories separately:

    python src/evaluate_evidence_v3.py --checkpoint models/evidence_sensitivity_v3_3000_steps.pt --group all

The 3000-step budget schedules 2010 present-leader updates and 990
missing-leader updates, matching approximate answerable-example exposure
from V2B despite a larger dataset.

Training only accepts the registered V3 training path, not its evaluation
partitions. The generator refuses to overwrite existing files with different
contents. Existing V1/V2 data and checkpoints remain unchanged.

Read the training and per-category exact-match scores separately; unseen-name
copying is a harder objective than matching familiar evidence values.
