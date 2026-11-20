# V5I — Explicit question-conditioned span readout

**Status:** Architecture and test code committed; tests, training and DEV
measurements have **not** been executed or validated. Reserved EVAL remains
sealed. V5H stays unchanged.

## Research question

Does a compact, learned interaction between question and evidence improve
the model's ability to select different evidence fields when the question
changes, compared with V5H's question-agnostic span readout?

The known V5H *local* result was 24/48 DEV exact, 0/24 paired both-correct,
0/12 correct switches on 12 complete-source pairs, and 12/12 unsupported
answers on missing-target questions. These are comparison points, not V5I
results.

## Single architecture intervention

The unchanged V5 encoder already processes both question and evidence
together, but V5H predicts spans using separate linear projections of each
token's contextual hidden state. V5I explicitly constructs a pooled
question representation from the **question bytes only**, excluding the
compact prompt framing and trailing `Answer:` marker:

```
H = Encoder(embedding(input) + positions)
q = mean(H at question-byte positions)
H' = H + tanh(W_f [H ; broadcast(q)] + b_f)
start = start_head(H')
end = end_head(H')
answerability = answerability_head(H'[BOS])
```

- Encoder layers, attention heads, embeddings, tokenizer and input format
  are unchanged.
- The fusion layer adds **8,256 parameters** (64×128+64);
  the baseline has 124,675 parameters and V5I has **132,931**.
- Candidate masking is unchanged: answers must originate in evidence, not
  the question or framing.
- Training data contain no explicit `requested_field` input to the model.
  The abstract representation is *learned*, not guaranteed.
- The joint complete-field decoder remains unchanged; it receives the
  new start/end logits.

This tests explicit conditioning, not the reversible Hopfield memory,
sleep/consolidation or agentic designs. It also does not establish that
the original end-boundary failure has been corrected.

## Controlled comparison with V5H

Keep fixed: V5G registered TRAIN and DEV datasets (48 each), seed 42,
uniform question schedule, 1,200 updates, Adam LR=0.001, dropout=0.1,
gradient clipping at 1.0, answerability BCE, start/end CE, paired ranking
hinge margin=1.0 and weight=0.5, the existing candidate enumerator,
joint decoder and 0.5 answerability threshold. DEV measured at step 0
and every 120 updates.

The only *intended* experimental variable is the question-conditioned
architecture. The training and evaluation code are independently copied
entry points to prevent V5H files/checkpoints being overwritten.

**Primary:** paired both-correct > 0/24 and correct field switches > 0/12.
**Guardrail:** fewer unsupported answers than 12/12 without excessive
false abstention. **Diagnostics:** field confusion, answerability Brier/BCE,
start and end CE, question sensitivity. **Cost:** parameter count,
optimization time, full report time and matched inference timing when
available. A good gradient or ranking margin alone is not success.

## Run locally on your new Lenovo

From the **repository root** in PowerShell, using the project environment:

```powershell
git fetch origin
git switch --track origin/experiment/question-conditioned-v5i
$env:PYTHONPATH = "$PWD\src"

python -m pytest tests/test_v5i_question_conditioning.py -q
python -m unittest discover -s tests/experiments -p "test_paired_ranking_v5h.py" -v

# Only if the registered TRAIN/DEV files are missing:
python -m experiments.question_aware_v5g.data_design --partition train
python -m experiments.question_aware_v5g.data_design --partition dev

python -m experiments.question_aware_v5i.train_question_conditioned `
  --steps 1200 --seed 42 --measure-every 120 `
  --margin 1.0 --rank-weight 0.5 `
  --checkpoint experiments/question_aware_v5i/checkpoints/v5i_seed42.pt `
  --output-dir experiments/question_aware_v5i/reports/v5i_seed42

python -m experiments.question_aware_v5i.evaluate_dev `
  --checkpoint experiments/question_aware_v5i/checkpoints/v5i_seed42.pt `
  --output-dir experiments/question_aware_v5i/reports/v5i_dev_seed42
```

If the branch already exists locally, use
`git switch experiment/question-conditioned-v5i` instead of
`git switch --track ...`. Review `git status` before switching to
protect local uncommitted work.

The data-generation commands refuse to overwrite mismatching existing
files. Training refuses existing checkpoints and report directories.
For reruns, choose new output paths. Never run on the reserved EVAL
partition or reuse a previous freeze manifest.

## Implementation locations

- `src/grounded_qa/models/span_model_v5i.py`: architecture and dedicated
  V5I-only checkpoint reader/writer
- `experiments/question_aware_v5i/train_question_conditioned.py`:
  V5H objective and schedule, V5I model
- `experiments/question_aware_v5i/evaluate_dev.py`: DEV-only evaluator
- `tests/test_v5i_question_conditioning.py`: mask, gradients, outputs,
  checkpoint isolation

After the run, compare to the *recorded* V5H metrics. A controlled wall-time
comparison needs similar hardware/load, and the 1,200-step run must not be
called generalization until separate held-out experiments are pre-registered.
