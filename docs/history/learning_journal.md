# grounded-qa

A small Python learning project for building a small language model from scratch
to answer questions using only supplied information. We will initialize its
weights locally, train on synthetic data, save our own checkpoint, and later
fine-tune that checkpoint on more synthetic examples.

Pretrained model files are unavailable; Python libraries are allowed. The plan
uses a small neural network built with PyTorch on the current Windows PC's CPU.
The first task will use a narrow set of synthetic record and question formats.
Accuracy is a goal to measure on held-out sources, not a property we can promise
before training and evaluation.

## Answering rule

- **Answerable:** answer every requested part using the supplied information.
- **Partially answerable:** answer the supported part, then identify each missing
  part with `Insufficient information.` Do not guess or use outside knowledge.
- **Unanswerable:** respond with exactly `Insufficient information.`

For example, if the information gives an event's date but no leader, a question
asking for both should receive:

> Date: April 18, 2030. Leader: Insufficient information.

The `answer` field in the dataset shows the desired response. The `answerability`
field is a label for inspecting and evaluating examples; it is not information
the eventual model should receive when answering.

## Files

```text
data/
  source/                   # Unsplit originals used to create the splits
    starter.jsonl           # 12 examples from 3 sources
    generated.jsonl         # 120 generated examples from 12 sources
  training/                 # Examples available for weight updates
    train.jsonl             # 8 starter examples from 2 sources
    generated_train.jsonl   # 90 generated examples from 9 sources
    generated_train.json    # Existing copy, retained (contains JSONL)
    practice_wordings.jsonl # 20 reviewed examples from one training source
    practice_wordings_corrected.jsonl # Same 20 examples; clearer q03 and q03_alt
  holdout/                  # Sources kept out of weight updates
    evaluation.jsonl        # 4 starter examples from 1 source
    generated_evaluation.jsonl # 30 generated examples from 3 sources
    generated_evaluation.json  # Existing copy, retained (contains JSONL)
notebooks/
  explore_examples.ipynb    # Load and inspect the starter examples
src/
  validate_data.py          # Check JSONL structure and label consistency
  split_data.py             # Reproducible split that keeps sources together
  prompting.py              # Shared instructions and input formatting
  preview_prompts.py        # Inspect prompts beside separate reference answers
  tokenizer.py             # Text-to-number mapping built with Python alone
  generate_data.py          # Fictional records, questions, and grounded targets
  expand_practice.py        # Pair existing training questions with alternate wordings
  inspect_tokens.py         # Measure input/answer lengths and check proposed limits
  model.py                  # Tiny encoder-decoder initialized with random weights
  check_model.py            # Check a random or reloaded model without training
  checkpoint.py             # Save weights/settings and rebuild the model on CPU
  train.py                 # Train on 3 examples or complete selected source groups
  predict.py               # Generate answers; review a question or selected sources
  evaluate.py              # Existing empty placeholder
models/                    # Local model storage; ignored by Git
  practice.pt              # Saved 60-update practice model, created locally
tests/
  test_checkpoint.py        # Checks for reloading and preserving existing files
  test_source_selection.py  # Checks for complete source groups and CLI selection
  test_training_data.py     # Checks alternate training files and source membership
  test_multi_source.py      # Checks explicit selection of multiple complete sources
requirements.txt           # pandas, ipykernel, and torch
```

`data/source/` holds the unsplit originals. `data/training/` holds examples used
to adjust model weights, including alternate questions for a training source.
`data/holdout/` holds separate sources for evaluating behavior on information
the model has not trained on. Keep every question about a source in the same
split. The existing `.json` copies also contain JSONL and are retained beside
their corresponding datasets; the scripts read and write the `.jsonl` files.

This folder organization preserves the existing examples and source assignments;
it does not create a new split or run training.

JSONL stores one JSON object per line. Each starter example has:

| Field | Purpose |
| --- | --- |
| `example_id` | Unique identifier for a question and its target answer |
| `source_id` | Identifier shared by all questions about the same source record |
| `information` | The complete fictional source record available to the model |
| `question` | The question to answer |
| `answerability` | `answerable`, `partial`, or `unanswerable` |
| `answer` | The desired response under the answering rule above |

There are three fictional source records, each with two answerable questions,
one partially answerable question, and one unanswerable question. All examples
with a given `source_id` repeat the same `information`.

When we split the data, we assign whole **source groups** to training
or evaluation. Randomly splitting individual questions could put the same source
information in both sets and make evaluation misleading. These 12 examples are
for learning the workflow; they are too few for a meaningful model evaluation.

## Python environment

Run commands in PowerShell from the project folder. The current `.venv` has been
verified with 64-bit Python 3.12.9, pandas 3.1.0, ipykernel 7.4.0, and
PyTorch 2.14.1+cpu. PyTorch is installed as the Python package `torch`.
`requirements.txt` currently lists package names without version pins, so a new
installation may resolve to different versions.

On a new computer, install Python and create a fresh local environment:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The existing environment is already ready to use. Verify it with:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import pandas, ipykernel; print(pandas.__version__, ipykernel.__version__)"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.tensor([2.0]) * 3)"
```

Using the environment's Python directly avoids needing to activate it in
PowerShell. Copy the project files when moving computers, then recreate `.venv`
there rather than copying it. All project data paths are relative.

## Explore in a notebook

Open `notebooks/explore_examples.ipynb` in a notebook-capable editor. In VS Code,
use the Python and Jupyter extensions, choose **Select Kernel**, and select this
project's `.venv\Scripts\python.exe`. Run the cells from top to bottom.

`ipykernel` lets an editor run Python notebook cells. The requirements do not
include a standalone Jupyter browser application. The notebook works with its
working directory set to either the project folder or `notebooks/`.

## Validate the data

After reviewing the examples, run this from the project folder:

```powershell
.\.venv\Scripts\python.exe src/validate_data.py
```

Expected result:

```text
Validation passed: 12 examples across 3 source groups.
answerable: 6, partial: 3, unanswerable: 3
Structure and label wording checked; factual support still requires human review.
```

The validator uses only Python's standard library. It reads the file without
changing it and checks:

- Each line is a JSON object with the required nonempty string fields.
- Each `example_id` is unique and each `answerability` label is allowed.
- A `source_id` always uses the same information, and identical information is
  not assigned to different source IDs.
- Unanswerable responses match `Insufficient information.` exactly.
- Partial responses include that phrase plus additional text; answerable
  responses do not include the fallback phrase.

It reports errors with line numbers and exits with code 1 on failure (0 on
success). Empty files fail validation. To check another file, supply its path:

```powershell
.\.venv\Scripts\python.exe src/validate_data.py data/source/starter.jsonl
```

These are structural checks. The validator cannot establish whether an answer
is factually supported or whether a partial response identifies every missing
part. Your manual review provides that check. It also does not require fixed
example counts, so we can add more examples later.

## Split by source group

After reviewing and validating the examples, run this from the project folder:

```powershell
.\.venv\Scripts\python.exe src/split_data.py
```

The script validates `data/source/starter.jsonl`, sorts its source IDs, and shuffles
those IDs using seed **42**. A seed is the starting value for a random number
generator: the same source IDs and seed reproduce this group selection. The
first shuffled source group goes to evaluation; the remaining groups go to
training. All questions from a source follow that source into the same split.
Rows keep their original order within each output file.

For the current dataset, the result is:

| File | Source groups | Examples | Answerable | Partial | Unanswerable |
| --- | --- | ---: | ---: | ---: | ---: |
| `data/training/train.jsonl` | `ferry_001`, `workshop_001` | 8 | 4 | 2 | 2 |
| `data/holdout/evaluation.jsonl` | `kit_001` | 4 | 2 | 1 | 1 |

Training examples will eventually be used to adjust the model. Evaluation
examples will be held aside to check its behavior on sources it did not train
on. This tiny split is for practicing that workflow, not measuring reliable
model performance.

The script uses Python's standard library and leaves `data/source/starter.jsonl`
unchanged.
It fills missing or empty output files. Rerunning it leaves identical output
files alone; if either output already contains different data, it stops before
writing either file. Input and output paths must be different.

Optional settings are listed by `src/split_data.py --help` when run with Python.
`--evaluation-groups` counts source groups, not individual examples, and defaults
to 1. `--seed` defaults to 42. Changing the set of source IDs can change group
selection even with the same seed. Use `--train-output` and `--evaluation-output`
to choose new files if you want to keep an existing split while experimenting.

Check the generated files with the same validator:

```powershell
.\.venv\Scripts\python.exe src/validate_data.py data/training/train.jsonl
.\.venv\Scripts\python.exe src/validate_data.py data/holdout/evaluation.jsonl
```

## Preview the model's input

A **prompt** is the text we will give the model. Our prompt contains the answering
instructions, the supplied information, and the question. It ends with an empty
`Answer:` section where the model's response will begin.

`src/prompting.py` defines `build_prompt(information, question)`. It accepts only
those two fields, so the answer and answerability label cannot be accidentally
included by passing a whole dataset row. The same instructions cover answerable,
partial, and unanswerable questions; we never tell the model which type it is.

| Dataset content | Role |
| --- | --- |
| `information`, `question` | Included in the prompt |
| `answer` | Desired response for later training; reference for evaluation |
| `answerability` | Used by us to organize and inspect results |
| `example_id`, `source_id` | Used by us to identify examples and keep sources together |

Run this from the project folder to preview one training example of each type:

```powershell
.\.venv\Scripts\python.exe src/preview_prompts.py
```

Each preview displays `MODEL INPUT`, followed by a separate `EXPECTED ANSWER`
section. That expected answer is read from the dataset; it is not generated by
a model and is not part of the prompt. The `Insufficient information.` phrase
does appear in the instructions because it defines the response policy.

To inspect just the partially answerable workshop question:

```powershell
.\.venv\Scripts\python.exe src/preview_prompts.py --example-id workshop_001_q3
```

The prompt contains the workshop information and asks where it takes place and
who leads it. Its separate reference answer is:

```text
Location: Room Cedar. Leader: Insufficient information.
```

You can also preview a different dataset, using the same prompt function:

```powershell
.\.venv\Scripts\python.exe src/preview_prompts.py --data data/holdout/evaluation.jsonl
```

The preview validates its input and reads files without changing them. It uses
only Python's standard library. A model initialized from scratch does not yet
understand these instructions; it must learn from training examples. Before
training, we can shorten the repeated instructions to keep input sequences small
while preserving the supplied information and question. The prompt alone does
not guarantee grounded answers.

## Build our own tokenizer

A tokenizer converts text to numbers that a neural network can process. Our
first tokenizer is a small, fixed mapping based on UTF-8 bytes. It needs no
downloaded vocabulary and does not learn anything from training or evaluation
data. Python supplies the UTF-8 encoding and decoding operations; see the
[Python Unicode guide](https://docs.python.org/3.12/howto/unicode.html).

| Token ID | Meaning |
| --- | --- |
| `0` | PAD: padding for sequences of different lengths |
| `1` | BOS: beginning of a sequence |
| `2` | EOS: end of a sequence |
| `3` through `258` | A UTF-8 byte value plus 3 |

That gives us **259 tokens**. For example, `A` is byte 65, so its token ID is
68. A space is byte 32, so its token ID is 35. Some characters use multiple
UTF-8 bytes and therefore multiple tokens. Token IDs are labels, not meanings;
the neural network will learn its own numerical representations of them.

Run this from the project folder:

```powershell
.\.venv\Scripts\python.exe src/tokenizer.py --text "Room Cedar."
```

The demo adds BOS and EOS, prints the token IDs, and converts them back to the
original text. Expect `Round trip matches: True`. Try the fallback sentence too:

```powershell
.\.venv\Scripts\python.exe src/tokenizer.py --text "Insufficient information."
```

`encode()` leaves sequence markers out unless requested. `decode()` ignores PAD
and BOS and stops at the first EOS, ignoring anything after it. It rejects
invalid token IDs and invalid UTF-8 by default. The optional `errors="replace"`
mode is for displaying malformed future model output, not for changing training
data silently.

This mapping is simple and has a small vocabulary, but it produces long token
sequences compared with tokenizers that combine bytes into words or word parts.
We will measure input lengths before choosing the network's size and limits.
The current tokenizer does not truncate text, train a network, or answer questions.

## Generate a controlled QA dataset

`src/generate_data.py` creates fictional workshops with four possible facts:
date, location, leader, and seat count. Each source shows exactly two facts.
The omitted fields vary: every six sources cover all six possible visible-field
pairs, in a shuffled order. Displayed fact order also varies.

For each source, the generator asks four single-fact questions and all six
two-fact combinations. It chooses one of two phrasings for each question.
Answers are derived only from the visible facts, using the same policy we
reviewed earlier. Hidden values are never passed to the answer-building function.

For example:

```text
Information: Fictional Workshop W0001. Location: Room Cedar. Seats: 12.
Question: What are the date and location of Workshop W0001?
Answer: Location: Room Cedar. Date: Insufficient information.
```

The supported part comes first even when the question asks about a missing part
first. If none of the requested facts are visible, the whole answer is exactly
`Insufficient information.` All questions from a source share the same
`source_id` and identical `information`.

Run from the project folder:

```powershell
.\.venv\Scripts\python.exe src/generate_data.py
.\.venv\Scripts\python.exe src/validate_data.py data/source/generated.jsonl
```

Default counts:

| Scope | Sources | Examples | Answerable | Partial | Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: |
| One source | 1 | 10 | 3 | 4 | 3 |
| Whole file | 12 | 120 | 36 | 48 | 36 |

Review one example of each type using the existing preview tool:

```powershell
.\.venv\Scripts\python.exe src/preview_prompts.py --data data/source/generated.jsonl
```

You can review every question for the first source by selecting individual IDs
from `generated_workshop_0001_q01` through `generated_workshop_0001_q10` with
`--example-id`. Compare the requested facts with the displayed information.

`--records` controls the number of source records, and `--seed` defaults to 42.
The same settings reproduce the same data. Identical existing output is left
alone; differing nonempty output is preserved and produces an error. Use
`--output` to choose another filename when experimenting. Source IDs are local
to one generated dataset: different seeds create alternative datasets with
reused IDs. Generate a single pool and split it by source; do not generate or
merge separate training and evaluation pools with different seeds.

The default output, `data/source/generated.jsonl`, is separate from the reviewed
`data/source/starter.jsonl`, `data/training/train.jsonl`, and
`data/holdout/evaluation.jsonl`. It remains the complete pool for the generated
split described below. The two phrasings and fixed
number of visible facts deliberately keep this first task small. Success here
would measure performance on these controlled templates, not general QA ability.
We can expand the generator after reviewing the examples.

## Split and measure the generated data

We reuse the existing split script with separate output filenames. Run this
from the project folder to reproduce the split:

```powershell
.\.venv\Scripts\python.exe src/split_data.py --input data/source/generated.jsonl --train-output data/training/generated_train.jsonl --evaluation-output data/holdout/generated_evaluation.jsonl --evaluation-groups 3 --seed 42
```

| File | Sources | Examples | Answerable | Partial | Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: |
| `data/training/generated_train.jsonl` | 9 | 90 | 27 | 36 | 27 |
| `data/holdout/generated_evaluation.jsonl` | 3 | 30 | 9 | 12 | 9 |

The evaluation sources are `generated_workshop_0003`,
`generated_workshop_0006`, and `generated_workshop_0008`. Every question about
each of those sources stays in evaluation. The other nine sources are in
training. All 120 original rows are preserved across the two files, with no
shared source IDs. The original starter split still uses
`data/training/train.jsonl` and `data/holdout/evaluation.jsonl`; future model work
will use the generated split explicitly.

Check the new split with the existing validator:

```powershell
.\.venv\Scripts\python.exe src/validate_data.py data/training/generated_train.jsonl
.\.venv\Scripts\python.exe src/validate_data.py data/holdout/generated_evaluation.jsonl
```

Before defining the model, measure how much text it will receive:

```powershell
.\.venv\Scripts\python.exe src/inspect_tokens.py
```

The report validates each file, builds the complete prompt, and counts tokens
using our tokenizer. Input counts include the instructions, information,
question, and formatting. Target counts measure only the expected answer.
Each count includes one EOS token, with no BOS token or padding. These are
UTF-8 byte-token counts, not word counts.

For the current generated data and prompt format:

| Split | Input min / mean / max | Target min / mean / max |
| --- | --- | --- |
| Training | 527 / 545.2 / 560 | 10 / 34.4 / 56 |
| Evaluation | 525 / 540.3 / 554 | 10 / 32.9 / 54 |

The report names one longest example for each measurement (the first if tied).
It also accepts optional limits, for example:

```powershell
.\.venv\Scripts\python.exe src/inspect_tokens.py --max-input-tokens 640 --max-target-tokens 64
```

Those limits accommodate the current examples and are now the defaults in
`ModelConfig`. They are not a measurement of training memory usage. Limits apply to the
counts including EOS. An example exactly at the limit passes; longer examples
are listed and cause exit code 1. Nothing is truncated or rewritten. If the
data or prompt changes, rerun this check before training so needed facts cannot
be silently cut from the input.

You can also supply other dataset paths after `src/inspect_tokens.py`. All
data inspection uses Python's standard library and our existing modules.

## Define and check the tiny model

`src/model.py` defines `TinyGroundedQA` using PyTorch's
[Transformer building block](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Transformer.html).
It creates random weights locally. Installing PyTorch gives us the tools for
building the network; no pretrained model files are involved.

The encoder reads the complete prompt. The decoder uses the encoder's output
and preceding answer tokens to predict the next answer token. Token embeddings
turn IDs into vectors, and learned position embeddings represent their order.
The final linear layer produces one raw score, called a **logit**, for each of
our 259 possible tokens at each answer position.

| Setting | Initial value |
| --- | ---: |
| Vocabulary | 259 tokens |
| Vector size (`d_model`) | 64 numbers per token |
| Attention heads (`nhead`) | 4 |
| Encoder / decoder layers | 2 / 2 |
| Feed-forward hidden size | 128 |
| Dropout | 0.1, disabled during this check |
| Maximum input / target length | 640 / 64 tokens |
| Trainable parameters | 246,147 |

The weights occupy about **0.94 MiB** in float32. This counts only the weights;
PyTorch, intermediate calculations, and eventual training need additional RAM.
We still need to measure training memory and runtime before choosing a longer run.

Run this from the project folder:

```powershell
.\.venv\Scripts\python.exe src/check_model.py
```

The script uses the first row of `data/training/generated_train.jsonl`, one CPU thread,
and seed 42. It prints the parameter count, tensor shapes, and a loss value.
For the current default example, the shapes are:

```text
Source shape: (1, 538) [batch, input tokens]
Decoder shape: (1, 26) [batch, answer tokens]
Logits shape: (1, 26, 259) [batch, answer tokens, vocabulary]
```

The first dimension is 1 because we process one example. The last dimension
of the logits is 259 because the network scores every possible next token.

For this calculation, we use **teacher forcing**: supply the reference answer
shifted by one position, so the decoder receives only the correct preceding
tokens. For a short answer `Hi`, that alignment is:

| Position | Decoder input | Token to predict |
| --- | --- | --- |
| 0 | BOS | H |
| 1 | H | i |
| 2 | i | EOS |

A causal attention mask prevents each position from seeing later decoder
tokens. Padding masks exclude padding from attention; sequences must be padded
on the right. The source prompt ends with EOS. Decoder input (`BOS + answer`)
and target (`answer + EOS`) have equal lengths, so the target limits agree with
our earlier length report. Excess lengths raise an error rather than truncating.

Cross-entropy compares the raw scores with the correct next tokens. A finite
loss shows the calculation works; this loss comes from random weights and does
not measure generated-answer accuracy. The script disables gradient tracking,
does not update weights, and saves no checkpoint. During future answer
generation, the decoder will use its own preceding output tokens; it will not
receive the reference answer.

Use `--example-id` to check another row, or `--data` to select a different file.
Without `--checkpoint`, each run creates a new random model. To check saved
weights instead, supply a checkpoint as described below.

## Try the first training loop

Without `--source-id` or `--source-ids`, `src/train.py` runs the original small learning experiment.
It selects the first
answerable, partial, and unanswerable example from `data/training/generated_train.jsonl`.
For the current file, those are `generated_workshop_0001_q02`,
`generated_workshop_0001_q05`, and `generated_workshop_0001_q01`. All three concern
the same workshop. This deliberately simple practice task checks that the
network can begin fitting a few familiar examples.

With `.venv` active, run from the project folder:

```powershell
python src/train.py
```

Without activation, use `.\.venv\Scripts\python.exe src/train.py` instead.
The defaults are 60 updates, learning rate 0.001, seed 42, one CPU thread,
float32 weights, and a batch size of one. **One step is one weight update**:
the script cycles through the three examples, so each receives 20 updates.
The learning rate controls the scale of the optimizer's adjustments.

Read the four numbered comments inside the loop in `src/train.py`:

1. `optimizer.zero_grad(...)` clears the previous update's gradients.
2. The model predicts token scores, and cross-entropy calculates the loss.
3. `loss.backward()` calculates gradients: how changing each weight affects
   the loss. Gradient clipping limits their combined size, and nonfinite loss
   or gradients stop the experiment.
4. `optimizer.step()` uses the Adam optimizer to adjust the weights.

This follows the process in PyTorch's
[training-loop tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
Teacher forcing and the causal mask work as described in the model check.

Before training and every 15 updates, the script measures loss across all
selected practice examples (three by default). It adds their token losses and divides by the number of
target tokens, including EOS. Longer answers therefore contribute more tokens.
These measurements use `model.eval()` and `torch.no_grad()` to disable dropout
and gradient tracking. **`model.eval()` changes network behavior; it does not
load the evaluation dataset.** Training mode is restored for the next update.

Our checked default run produced:

```text
Before training: 5.8311
After  15 updates: 3.5946
After  30 updates: 2.2740
After  45 updates: 1.4076
After  60 updates: 0.9257
```

The initial value differs from `check_model.py` because this report combines
three examples, while that check uses one. The experiment took 22.3 seconds in
the checked environment, excluding package imports and setup. Runtime and
exact numerical results can differ between environments. Loss need not fall
at every measurement. Lower loss here means better prediction of reference
tokens when given correct preceding tokens; it does not establish generated
answer accuracy or performance on unseen sources.

For a shorter run, use `python src/train.py --steps 15`. `--learning-rate` and
`--seed` are also available; `--help` lists the options. The default input file
is the generated training split. `--data` can select a reviewed practice file
whose sources and facts match that split, as described below. The training
script does not automatically read the holdout and does not change any dataset.

**Each training run starts from random weights.** Without `--checkpoint`,
learned weights live only in memory during the run. To keep them, supply a new
checkpoint filename as described below. Running `check_model.py` without
`--checkpoint` still checks a fresh random model.

## Save weights and reload them

A **checkpoint** is a file containing the learned numbers and the information
needed to rebuild the model that uses them. `src/checkpoint.py` provides the
shared saving and loading functions. This follows PyTorch's
[state-dictionary saving and loading pattern](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

To train the same three examples and keep the result, run:

```powershell
python src/train.py --checkpoint models/practice.pt
```

The checkpoint is saved only after training finishes. The script immediately
loads a second model from that file and compares its scores with the trained
model on all three practice examples, with dropout disabled. Expect:

```text
Checkpoint saved: models\practice.pt
Reload check passed: identical scores on all 3 practice examples.
```

An existing filename is refused **before training starts**, including an empty
file. If `models/practice.pt` already exists, use it for the reload command below.
For a separate training run, choose a new name such as:

```powershell
python src/train.py --checkpoint models/practice_run2.pt
```

Saving also uses exclusive file creation, so it cannot overwrite a file that
appears during training. New parent folders are created when saving. A failed
write attempts to remove only the incomplete file it just created. Checkpoint
filenames must end in `.pt`; these files and `models/` are already ignored by Git.

The checkpoint contains:

| Saved item | Why we need it |
| --- | --- |
| `model_state_dict` | The learned tensors, identified by their layer names |
| `model_config` | Dimensions and limits used to construct the same network |
| `tokenizer` | The byte mapping and special token IDs |
| `prompt_template` | The instructions and formatting used for inputs |
| `training` | Update count, learning rate, seed, selection mode/count, selected IDs, and recorded losses |
| Format and PyTorch versions | Information identifying how the file was created |

To reload in a separate Python process and check one example:

```powershell
python src/check_model.py --checkpoint models/practice.pt
```

This command reads saved tensors onto the CPU with `weights_only=True`, builds
the network using its saved configuration, and installs the tensors using
`load_state_dict(..., strict=True)`. It then calls `model.eval()` to disable
dropout. The loader rejects incompatible prompt/tokenizer settings, incomplete
configurations, invalid weights, and mismatched layer names or shapes.

Look for `Loaded checkpoint`, `Recorded training updates: 60`, and
`Teacher-forced loss from reloaded weights`. That loss covers the single
example shown by `check_model.py`. It need not equal the three-example mean
reported by `train.py`. `--seed` affects only the random-model check; it does
not change reloaded weights. The check performs no further training.

The local `models/practice.pt` was created by the verified 60-update run.
The reload comparison checks preservation of the scores; it does not establish
answer accuracy. `src/predict.py` separately checks generated answers, as
described below.

This checkpoint stores model weights and run information. Optimizer history
and random-generator state are not saved, and `train.py` does not yet resume
from a checkpoint. Later fine-tuning can start from these saved model weights.

Older checkpoints may record a dataset path from before the folder move in
their `training_file` metadata. That path is historical run information; the
checkpoints still load, and their saved metadata and weights are unchanged.

When moving computers, copy the checkpoint with the project, recreate `.venv`,
and use the relative path in the reload command.

Run the focused checkpoint checks with:

```powershell
python -m unittest discover -s tests -p "test_checkpoint.py"
```

## Train and review every question for one workshop

The 600-update experiment fit the three selected questions, but answered the
unpracticed leader question with `Insufficient information.` Its reference
answer, `Leader: Mira.`, was correct. Having a question in the training file
does not mean it was selected for weight updates.

The next experiment expands selection to **all ten questions** for
`generated_workshop_0001`. These include three answerable, four partial, and
three unanswerable examples, including the leader question. The source stays
entirely within `data/training/generated_train.jsonl`; no dataset is regenerated or split
again, and training does not read the holdout.

With `.venv` active, run from the project folder:

```powershell
python src/train.py --source-id generated_workshop_0001 --steps 2000 --checkpoint models/workshop_0001_2000_steps.pt
```

`--source-id` selects **every row with that source ID**, preserving file order.
The source must be present in the training file; a missing source stops the run
before the model is constructed. Without either source selection option, the
original selection of one example per answerability type is preserved.

At startup, check for `Selected 10 of 90 training examples.` The script lists
all selected IDs, including `generated_workshop_0001_q03`, and reports the
number of complete passes through them. With batch size one, 2,000 updates on
ten examples gives each example 200 updates. That is the same number of updates
per example as the earlier 600-update run on three examples. It is a starting
experiment; the larger set need not reach perfect scores at that update count.
This run will take longer than the three-example experiment.

Each run starts from random weights. It does not continue from
`practice_600_steps.pt`. Use a new checkpoint filename if the destination
already exists. The saved metadata records the actual selected IDs, source IDs,
selection mode, and example count. The reload check compares all ten examples.

After training, review **all ten generated answers** with the same source ID:

```powershell
python src/predict.py --checkpoint models/workshop_0001_2000_steps.pt --source-id generated_workshop_0001
```

Without a source or example selection option, prediction previews only the first
example of each answerability type. `--source-id`, `--source-ids`, and
`--example-id` are mutually exclusive: choose one source, several sources,
or one question. Dataset selection also cannot be combined with custom
`--information`/`--question` inputs.

Prediction starts the decoder with BOS and feeds back its own generated tokens.
It chooses the highest-scoring allowed token, excludes PAD/BOS from output,
and stops at EOS or the token budget. No reference answer is supplied to the
generation function. References are displayed afterward for comparison.

Review `Generated answer` and `Exact completed match` for each question.
An exact completed match requires identical text, valid UTF-8, and a predicted
EOS. Quotation marks in the printed answer delimit the displayed text.
`Stop` and `Valid UTF-8` describe output mechanics; they do not establish factual
correctness. No dataset or checkpoint is changed during prediction.

The immediate practice target is 10/10. The leader question is now included in
training, so a correct answer to it would measure fitting a practiced question.
Even 10/10 would not establish performance on new workshops or changed facts.
Broader training and separate-source evaluation remain later steps.

Run the checkpoint and selection checks with:

```powershell
python -m unittest discover -s tests
```

## Expand question wordings before training

The ten-question workshop run scored 7/10 after 2,000 updates and 9/10 after
4,000 updates. The remaining question, q05, asks for date and location. Its
correct target is `Location: Room Maple. Date: Insufficient information.`
Rephrasing that question changed the generated answer, but it still incorrectly
named seats as the missing field. These are practice results on one source.

The generator already defines two equivalent question templates. Its shared
`make_questions(name, requested)` function returns both; ordinary generation
still randomly chooses one, preserving the existing seeded output. The new
`src/expand_practice.py` uses both wordings from the existing training records.
It copies the reviewed reference answers; it does not use generated model answers.

With `.venv` active, run from the project folder:

```powershell
python src/expand_practice.py --source-id generated_workshop_0001
python src/validate_data.py data/training/practice_wordings.jsonl
```

The new file contains **20 examples from the same source**: 6 answerable,
8 partial, and 6 unanswerable. Each original row is immediately followed by
its alternate. For each pair:

- `example_id` gets an `_alt` suffix on the alternate row, so IDs stay unique.
- `question` uses the other wording while requesting the same facts in the same order.
- `source_id`, `information`, `answerability`, and `answer` remain identical.

For example, the two questions for q05 are:

```text
What are the date and location of Workshop W0001?
For Workshop W0001, give the date and location.
```

Both references must be exactly:

```text
Location: Room Maple. Date: Insufficient information.
```

Open `data/training/practice_wordings.jsonl` and review all ten pairs. To display q05's
two model inputs and their separate expected answers in the terminal:

```powershell
python src/preview_prompts.py --data data/training/practice_wordings.jsonl --example-id generated_workshop_0001_q05
python src/preview_prompts.py --data data/training/practice_wordings.jsonl --example-id generated_workshop_0001_q05_alt
```

Check that each question pair requests the same fields and has the same correct
answer. For the partial cases, check the missing field's name as well as the
supported fact. In particular, q05 asks for date, whereas q09 asks for seats.
The validator checks structure and label wording; your review checks meaning.

The expansion reads only `data/training/generated_train.jsonl`. It supports the existing
workshop templates and stops on unknown question formats instead of guessing.
An identical rerun leaves the output alone; different existing output is refused.
Use `--output` to choose another new file if needed. The original datasets and
checkpoints are preserved. No additional split is needed: this source already
belongs to training, and all its variants retain that source ID.

Review the new file before training. Creating it does not automatically select
it for training: use the explicit `--data` option in the next section. The
expansion itself does not train a model or establish that alternate wording
will improve accuracy.

## Train on the reviewed alternate wordings

All ten pairs have now been reviewed. `src/train.py` accepts `--data` to choose
the expanded practice file. For this stage, it checks every source ID in that
file against `data/training/generated_train.jsonl` and requires identical information
for each source. Unknown sources or changed source facts stop the run before
the model and optimizer are created. Question wording and example IDs may
differ; reference-answer correctness still depends on the earlier review.

Run this from the project folder with `.venv` active:

```powershell
python src/train.py --data data/training/practice_wordings.jsonl --source-id generated_workshop_0001 --steps 4000 --checkpoint models/workshop_0001_wordings_4000_steps.pt
```

Check the startup output for the new file path and `Selected 20 of 20 training
examples.` Both `generated_workshop_0001_q05` and its `_alt` row should appear.
The loop visits each row once per pass, so 4,000 updates give 200 passes and
200 updates per wording. The earlier 4,000-update run gave each of ten original
questions 400 updates. This experiment keeps the same total update budget
while distributing it across both wordings; improvement is not guaranteed.

This starts from fresh random weights with the same default seed and learning
rate. It does not continue the old checkpoint. Use the new checkpoint name to
preserve earlier runs. The saved metadata records the selected dataset path
and all twenty example IDs; the reload comparison checks all twenty examples.

After training, select the expanded file for prediction as well:

```powershell
python src/predict.py --checkpoint models/workshop_0001_wordings_4000_steps.pt --data data/training/practice_wordings.jsonl --source-id generated_workshop_0001
```

Check for `Previewing 20 examples` and review each generated answer. In
particular, compare q05 with q05_alt: both should answer the location and label
the date as missing. Report the final exact-match count and any failures.
The practice target is 20/20; even that would only establish fitting these
twenty examples from one familiar source.

For a direct comparison with the earlier 9/10 result, preview the original ten
questions using the new checkpoint and the default dataset:

```powershell
python src/predict.py --checkpoint models/workshop_0001_wordings_4000_steps.pt --source-id generated_workshop_0001
```

The original dataset and old checkpoint remain available. Without `--data`,
training and prediction still use `data/training/generated_train.jsonl`. Without
a source selection option, the default selection is still just one example per
answerability type; include `--data` and `--source-id` to practice all twenty
expanded rows from that one source.

The completed 4,000-update wording run reduced mean training token loss from
5.8382 to 0.0818 and took 2,937.2 seconds (about 49 minutes). Saving and reloading
preserved its scores. Its independent generated-answer preview nevertheless
matched only **2/20** references:

| Questions | Exact completed matches |
| --- | ---: |
| Original ten | 1/10 |
| Alternate ten | 1/10 |
| All twenty | 2/20 |

Every generated answer was `Location: Room Maple. Leader: Mira.` Only q08 and
q08_alt requested both facts. This is a regression from the earlier checkpoint's
9/10 on the original questions. Low teacher-forced token loss can coexist with
incorrect generated sequences; the two scores measure different behavior.
Both checkpoints are preserved for comparison. The holdout remains untested.

## Return to the single-source wording experiment

The alternate-wording experiment is still unresolved. The two-source pilot
also produced the same misspelled `Insuficient information.` for all twenty
questions, scoring 0/20. Keep those results as experiment records; the current
focus returns to the twenty wordings for Workshop W0001 before broader training.

`data/training/practice_wordings_corrected.jsonl` is a separate copy of the
reviewed twenty-row file. Only the questions for q03 and q03_alt change:

```text
Who is the leader of Workshop W0001?
For Workshop W0001, give the leader's name.
```

Both references remain `Leader: Mira.` All IDs, source information, targets,
labels, row counts, and row order are preserved. Choose this file explicitly
with `--data`; do not combine it with the old file, because they share IDs.
Earlier datasets and checkpoints are retained for reproducing earlier results.
The generator and expander still use the earlier templates; they do not recreate
this manually corrected copy. The holdout does not need changing for this step.

Preview the corrected question and its alternate without loading a model:

```powershell
python src/preview_prompts.py --data data/training/practice_wordings_corrected.jsonl --example-id generated_workshop_0001_q03
python src/preview_prompts.py --data data/training/practice_wordings_corrected.jsonl --example-id generated_workshop_0001_q03_alt
```

Sample count and training updates are separate choices: this file has twenty
examples, and 4,000 updates would visit each one 200 times. We have not established
a sufficient sample size or update budget for reliable alternate-wording answers.
Fixing the grammar alone does not establish that the model will answer correctly.
If generated answers remain poor, record the failures and diagnose the small
experiment before adding more sources or committing to another long run.
Review original and alternate answers separately; fitting all twenty would be
a useful practice milestone, but would not establish accuracy on unseen sources.

## Select multiple independent training sources

The implemented multi-source option is available for later experiments.
`--source-ids` selects
all questions from each named source in file order. Every requested ID must be
present in the chosen dataset; empty, duplicate, missing, or conflicting source
selections stop before model creation. Existing single-source commands and the
default three-example selection still work.

Start with these two sources from `data/training/generated_train.jsonl`:

| Source | Supplied facts |
| --- | --- |
| `generated_workshop_0001` | Leader: Mira. Location: Room Maple. |
| `generated_workshop_0002` | Leader: Tavi. Date: 2030-04-23. |

That selects **20 original examples across two sources**, rather than both
wordings of one source. It lets us examine whether responses change with the
supplied values and available fields. The existing source split is preserved.

Use a short pilot to check the selection and runtime before choosing a longer
training budget:

```powershell
python src/train.py --source-ids generated_workshop_0001 generated_workshop_0002 --steps 200 --checkpoint models/two_sources_fresh_200_steps.pt
```

Expect `Selected 20 of 90 training examples.` and `Source groups: 2`. Two hundred
updates give each example ten updates. This is a workflow and timing check;
accurate generated answers are not expected from so little practice. The
checkpoint records the actual example IDs, source IDs, and source count.

Review the same selected questions afterward:

```powershell
python src/predict.py --checkpoint models/two_sources_fresh_200_steps.pt --source-ids generated_workshop_0001 generated_workshop_0002
```

Include the plural `--source-ids` option in both commands. Do not combine it with
`--source-id` or prediction's `--example-id`. The training data checks still
restrict alternate practice files to sources and facts from the training split.

Fresh training remains the default. Checkpoint initialization for fine-tuning
is still planned and will be an explicit choice, so fresh training and
fine-tuning can be compared on the same selected examples. There is no automatic
carry-forward, optimizer restoration, or architecture change in this step.
Learning rate, file-order cycling, and the existing loss-report interval are
unchanged. Any future comparison should report the starting checkpoint and
update budget, as well as generated-answer scores.

## Plan for a model we can fine-tune

We will build and check each part separately:

1. Convert text to token IDs with our tokenizer (implemented).
2. Generate more synthetic records and correct answers, with deliberate missing
   facts, changed values, and varied questions (first generator implemented).
   Keep sources together in splits and keep evaluation examples out of training
   (generated split and token-length inspection implemented).
3. Define a tiny neural network with randomly initialized weights (implemented
   and checked on the CPU). Measure training memory use and runtime before a
   longer run.
4. Train it to predict answer tokens from information and question tokens
   (practice selection, checkpoint saving/reloading, and answer generation
   implemented). We can now train and review one or more complete source groups. Later,
   evaluate fully supported, partial, and
   unsupported questions separately on held-out sources.
5. Fine-tune our saved checkpoint on additional synthetic training examples and
   compare performance with the saved original, including whether earlier
   behavior is retained.

Initial training creates the first learned weights. Fine-tuning starts from
those saved weights and adjusts them further. Our current 12 examples illustrate
the format; they cannot teach broad language skills or establish reliable QA.
The synthetic data generator will create training targets; the neural network
must learn to produce answers at inference time. The tokenizer only encodes text.

## Current stopping point

The folders, environment, Git exclusions, starter examples, and exploration
notebook are ready. The starter examples have been manually reviewed, and the
data validator, source-group split, prompt preview, tokenizer, and initial
synthetic generator are ready. The generated data now has its own training and
evaluation split, and token lengths have been measured. CPU PyTorch is installed;
the tiny network and its calculation check are ready. The generated split has
been reviewed, and a short CPU training experiment has reduced loss on three
training examples. Its weights have been saved and reloaded successfully.
At 600 updates, all three practiced answers matched their references; the
unpracticed leader question did not. Training on all ten questions for one
source subsequently reached 7/10 at 2,000 updates and 9/10 at 4,000 updates.
The alternate-wording run subsequently scored 2/20, including 1/10 on the
original questions. The 200-update two-source pilot scored 0/20; no long
multi-source training run has been performed. A separate corrected copy of the
single-source wordings is ready, and the next focus is that smaller experiment.
The folder is not yet a Git repository; `.gitignore` is ready for when Git is initialized.
It excludes `.venv`, Python and notebook caches,
local models, and common model checkpoint files.

The plan is to build our own small language model and run it locally on the
current 8 GB Windows PC's CPU. Practice training has been tested on three, ten,
and twenty examples from one source, with the saved checkpoints kept for comparison.
No pretrained models have been downloaded. Generated answers have been checked
on the practice questions; holdout answer accuracy has not been measured.

**Next small step:** review q03 and q03_alt in
`data/training/practice_wordings_corrected.jsonl`, then diagnose the unresolved
single-source wording results before choosing another training experiment.
