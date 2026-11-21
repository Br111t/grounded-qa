"""V5K read-only probability diagnostic, threshold sensitivity and false abstentions.

Reloads immutable V5J bilinear checkpoint and TRAIN-fitted V5K calibration.
The threshold sweep is descriptive; never tune or choose a threshold on DEV.
Reserved EVAL is not touched. No training or overwrites.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5i.diagnose_bilinear_answerability import (
    load_bilinear_checkpoint, measure_row,
)

THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def calibrated_probability(record, params):
    z = (record["max_joint"] - params["train_mean"]) / params["train_std"]
    logit = record["answerability_logit"] + params["lambda"] * z + params["bias_delta"]
    return float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64))), float(logit)


def load_calibration(path, checkpoint_path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if payload.get("checkpoint_sha256") != checksum:
        raise ValueError("V5K calibration does not match this V5J checkpoint")
    train_checksum = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    if payload.get("train_sha256") != train_checksum:
        raise ValueError("V5K calibration does not match registered TRAIN data")
    if payload.get("feature") != "max_joint":
        raise ValueError("Expected frozen V5K max_joint feature")
    params = payload["calibration"]
    if (params["train_std"] <= 0
            or not all(isinstance(params[k], (int, float))
                       for k in ("lambda", "bias_delta", "train_mean", "train_std"))):
        raise ValueError("Invalid V5K calibration values")
    return params, checksum


def diagnostics(rows, params, model):
    results = []
    for row in rows:
        record = measure_row(model, row)
        p, a_prime = calibrated_probability(record, params)
        top_field = record["candidates"][0]["field"]
        results.append({
            "example_id": row["example_id"],
            "source_id": row["source_id"],
            "question": row["question"],
            "requested_field": row["target_field"],  # audit label, not inference input
            "answerable": row["expected_value"] is not None,
            "top_candidate_field": top_field,
            "candidate_count": record["candidate_count"],
            "max_joint": record["max_joint"],
            "max_end": record["max_end"],
            "joint_gap_top2": record["joint_gap_top2"],
            "original_logit": record["answerability_logit"],
            "original_probability": record["answer_probability"],
            "calibrated_logit": a_prime,
            "calibrated_probability": p,
            "correct_top_field_if_answered": (
                row["expected_value"] is not None and top_field == row["target_field"]
            ),
            "false_abstention_at_0_5": (
                row["expected_value"] is not None and p < 0.5
            ),
            "unsupported_answer_at_0_5": (
                row["expected_value"] is None and p >= 0.5
            ),
        })
    return results


def threshold_metrics(rows, threshold):
    false_abstentions = sum(r["answerable"] and r["calibrated_probability"] < threshold for r in rows)
    unsupported = sum(not r["answerable"] and r["calibrated_probability"] >= threshold for r in rows)
    correct = sum(
        (r["answerable"] and r["calibrated_probability"] >= threshold
         and r["correct_top_field_if_answered"])
        or (not r["answerable"] and r["calibrated_probability"] < threshold)
        for r in rows
    )
    return {
        "threshold": threshold, "correct": correct, "total": len(rows),
        "false_abstentions": false_abstentions,
        "unsupported_answers": unsupported,
        "answerable_returned": sum(
            r["answerable"] and r["calibrated_probability"] >= threshold for r in rows
        ),
        "missing_rejected": sum(
            not r["answerable"] and r["calibrated_probability"] < threshold for r in rows
        ),
    }


def run(checkpoint, calibration, output_dir):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Report directory must be unused")
    torch.set_num_threads(1)
    model, _ = load_bilinear_checkpoint(checkpoint)
    params, sha = load_calibration(calibration, checkpoint)
    splits = {
        split: diagnostics(read_partition(split), params, model)
        for split in ("train", "dev")
    }
    baseline = threshold_metrics(splits["dev"], 0.5)
    if (baseline["correct"] != 45 or baseline["false_abstentions"] != 3
            or baseline["unsupported_answers"] != 0):
        raise ValueError("V5K DEV metrics do not match saved result")
    sweeps = {
        split: [threshold_metrics(rows, value) for value in THRESHOLDS]
        for split, rows in splits.items()
    }
    mistakes = [r for r in splits["dev"]
                if r["false_abstention_at_0_5"] or r["unsupported_answer_at_0_5"]]
    answerable = [r["calibrated_probability"] for r in splits["dev"] if r["answerable"]]
    missing = [r["calibrated_probability"] for r in splits["dev"] if not r["answerable"]]
    report = {
        "analysis": "v5k_read_only_probability_and_threshold_diagnostic",
        "checkpoint_sha256": sha,
        "calibration": params, "thresholds": sweeps,
        "dev_errors_at_0_5": mistakes,
        "dev_answerable_probability_range": [min(answerable), max(answerable)],
        "dev_missing_probability_range": [min(missing), max(missing)],
        "records": splits,
        "no_training": True, "no_dev_threshold_selection": True,
        "reserved_eval_accessed": False,
    }
    lines = [
        "# V5K calibrated probability diagnostic", "",
        "Checkpoint and calibration are read-only. Threshold 0.5 is the registered",
        "decision rule. Other thresholds are *descriptive only* and must not be",
        "selected using this repeatedly inspected DEV split.", "",
        "## DEV misclassified at threshold 0.50", "",
        "| Example | Requested | Top-scoring field | Original p | Calibrated p | Max joint |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in mistakes:
        lines.append(
            f"| {r['example_id']} | {r['requested_field']} | {r['top_candidate_field']} | "
            f"{r['original_probability']:.5f} | {r['calibrated_probability']:.5f} | {r['max_joint']:.3f} |"
        )
    lines += [
        "", f"DEV answerable probability range: {min(answerable):.5f}–{max(answerable):.5f}",
        f"DEV missing-target probability range: {min(missing):.5f}–{max(missing):.5f}",
        "", "## Probability threshold sensitivity (descriptive only)", "",
        "| Threshold | TRAIN correct | TRAIN false abstain | TRAIN unsupported | DEV correct | DEV false abstain | DEV unsupported |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a, b in zip(sweeps["train"], sweeps["dev"]):
        lines.append(
            f"| {a['threshold']:.2f} | {a['correct']}/48 | {a['false_abstentions']} | "
            f"{a['unsupported_answers']} | {b['correct']}/48 | "
            f"{b['false_abstentions']} | {b['unsupported_answers']} |"
        )
    lines += [
        "", "## Interpretation safeguards", "",
        "- A false abstention is an answerable example with calibrated p below threshold.",
        "- An unsupported answer is a missing-target example with p at or above threshold.",
        "- Changing the threshold does not change candidate field rankings or extracted spans.",
        "- Candidate counts and question/evidence structure may confound score calibration.",
        "- Review TRAIN and DEV ranges without picking an optimal DEV threshold.",
        "- The original V5J model and V5K calibration were not modified; sealed EVAL unused.", "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--calibration-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    run(args.checkpoint, args.calibration_path, args.output_dir)


if __name__ == "__main__":
    main()
