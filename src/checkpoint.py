"""Save and reload this project's model weights, settings, and run information."""

from contextlib import suppress
from dataclasses import asdict, fields
import json
from pathlib import Path
import pickle

import torch

from model import ModelConfig, TinyGroundedQA
from prompting import build_prompt
from tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID, VOCAB_SIZE


FORMAT_VERSION = 1
TOKENIZER_SPEC = {
    "encoding": "utf-8", "byte_offset": BYTE_OFFSET, "vocab_size": VOCAB_SIZE,
    "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID,
}
# Capture the instructions AND formatting, without storing an example's answer.
PROMPT_TEMPLATE = build_prompt("{information}", "{question}")


def check_destination(path: Path) -> None:
    """Refuse an existing filename before spending time on training."""
    if path.suffix.lower() != ".pt":
        raise ValueError("Use a checkpoint filename ending in .pt.")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Checkpoint already exists: {path}. Choose a different --checkpoint filename.")


def _check_training_info(training: object) -> None:
    if not isinstance(training, dict):
        raise ValueError("Checkpoint training information must be a dictionary.")
    steps = training.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("Checkpoint must record a positive number of training steps.")


def _check_weights(state: object) -> None:
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state dictionary.")
    for name, tensor in state.items():
        if (
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided or tensor.dtype != torch.float32
        ):
            raise ValueError("Checkpoint weights must be named float32 tensors.")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Checkpoint weight {name!r} contains non-finite values.")


def save_checkpoint(path: Path, model: TinyGroundedQA, training: dict) -> None:
    """Save a new file without overwriting any existing checkpoint.

    state_dict maps layer names to their numerical weights. We store that data,
    not the Python model object. ModelConfig is stored as a plain dictionary.
    """
    check_destination(path)
    # Keep metadata to portable strings, numbers, lists, and dictionaries.
    training = json.loads(json.dumps(training, allow_nan=False))
    _check_training_info(training)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    _check_weights(state)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": state,
        "tokenizer": TOKENIZER_SPEC,
        "prompt_template": PROMPT_TEMPLATE,
        "pytorch_version": str(torch.__version__),
        "training": training,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        # 'x' creates a new file exclusively: even a file created after our
        # preflight check cannot be overwritten here.
        with path.open("xb") as checkpoint_file:
            created = True
            torch.save(payload, checkpoint_file)
    except BaseException:
        # Close the file first (required on Windows), then remove only the
        # incomplete file created by this call. Existing files are never removed.
        if created:
            with suppress(OSError):
                path.unlink()
        raise


def load_checkpoint(path: Path) -> tuple[TinyGroundedQA, dict]:
    """Rebuild a CPU model, restore its weights, and return it in evaluation mode."""
    try:
        # Our format uses tensors and plain metadata, so unrestricted object
        # loading is unnecessary. There is no fallback to weights_only=False.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError(f"Could not load checkpoint {path}: {error}") from error

    if not isinstance(payload, dict) or type(payload.get("format_version")) is not int:
        raise ValueError("This is not a grounded-qa checkpoint.")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Checkpoint format version is not supported.")
    if payload.get("tokenizer") != TOKENIZER_SPEC:
        raise ValueError("Checkpoint tokenizer does not match the current tokenizer.")
    if payload.get("prompt_template") != PROMPT_TEMPLATE:
        raise ValueError("Checkpoint prompt format does not match the current prompt format.")
    config_data = payload.get("model_config")
    if not isinstance(config_data, dict) or set(config_data) != {field.name for field in fields(ModelConfig)}:
        raise ValueError("Checkpoint must contain all model configuration fields.")
    try:
        config = ModelConfig(**config_data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint model configuration: {error}") from error
    _check_training_info(payload.get("training"))
    state = payload.get("model_state_dict")
    _check_weights(state)

    model = TinyGroundedQA(config).to(device="cpu", dtype=torch.float32)
    # Strict loading rejects missing/extra weights and incompatible shapes.
    # We checked dtype above because load_state_dict can otherwise cast silently.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload["training"]
