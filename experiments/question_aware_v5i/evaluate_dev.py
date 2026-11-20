"""V5I read-only DEV evaluation. Reserved EVAL cannot be selected here."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.evaluate import enriched, predict, summarize
from grounded_qa.models.span_model_v5i import load_v5i_checkpoint


def run(checkpoint: Path, output_dir: Path):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Use a new, unused DEV report directory")
    model, meta = load_v5i_checkpoint(checkpoint)
    if meta.get("experiment_variant") != "v5i_question_conditioned":
        raise ValueError("Expected a V5I training checkpoint")
    if meta.get("training_sha256") != hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest():
        raise ValueError("Checkpoint does not match registered TRAIN data")
    rows = read_partition("dev")
    model.eval()
    outcomes = [enriched(row, predict(model, row)) for row in rows]
    metrics = summarize(outcomes)
    payload = {
        "phase": "dev",
        "experiment_variant": meta["experiment_variant"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_sha256": meta["training_sha256"],
        "dev_sha256": hashlib.sha256((DATA / "dev.jsonl").read_bytes()).hexdigest(),
        "metrics": metrics,
        "reserved_eval_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions.json").write_text(json.dumps(outcomes, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Reserved EVAL was not accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
