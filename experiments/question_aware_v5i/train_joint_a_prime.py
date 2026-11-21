"""V5K: frozen V5J bilinear model + TRAIN-only joint-score answerability calibration.

a_prime = a + lambda * ((max_joint - train_mean) / train_std) + bias_delta.
No encoder/span-head training, DEV fitting, requested-field inference oracle,
or sealed EVAL. New calibration and report paths only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.evaluate import enriched, summarize
from experiments.question_aware_v5i.diagnose_bilinear_answerability import (
    load_bilinear_checkpoint, measure_row,
)


def standardize(values, mean, std):
    if std <= 0:
        raise ValueError("Standard deviation must be positive")
    return (values - mean) / std


def fit_calibrator(records, *, steps=1000, lr=0.05):
    if steps < 1 or lr <= 0:
        raise ValueError("Invalid calibration configuration")
    feature = torch.tensor([r["max_joint"] for r in records], dtype=torch.float64)
    base = torch.tensor([r["answerability_logit"] for r in records], dtype=torch.float64)
    labels = torch.tensor([float(r["answerable"]) for r in records], dtype=torch.float64)
    if len(feature) != 48 or int(labels.sum()) != 36:
        raise ValueError("Unexpected TRAIN calibration labels")
    mean = float(feature.mean())
    std = float(feature.std(unbiased=False))
    if std <= 1e-12:
        raise ValueError("TRAIN feature is constant")
    z = standardize(feature, mean, std)
    lambda_weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias_delta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.Adam([lambda_weight, bias_delta], lr=lr)
    trajectory = []
    for step in range(steps + 1):
        with torch.no_grad():
            logits = base + lambda_weight * z + bias_delta
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if step in (0, steps):
                trajectory.append({"step": step, "train_bce": float(loss)})
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            base + lambda_weight * z + bias_delta, labels
        )
        loss.backward()
        optimizer.step()
        if not (torch.isfinite(lambda_weight) and torch.isfinite(bias_delta)):
            raise ValueError("Nonfinite calibration parameters")
    return {
        "lambda": float(lambda_weight.detach()),
        "bias_delta": float(bias_delta.detach()),
        "train_mean": mean,
        "train_std": std,
        "steps": steps,
        "lr": lr,
        "train_bce": trajectory,
    }


def classify(rows, record_by_id, calibrator=None):
    outputs = []
    for row in rows:
        record = record_by_id[row["example_id"]]
        base = record["answerability_logit"]
        adjusted = base
        if calibrator is not None:
            adjusted += (calibrator["lambda"] *
                         (record["max_joint"] - calibrator["train_mean"]) /
                         calibrator["train_std"] + calibrator["bias_delta"])
        p = float(torch.sigmoid(torch.tensor(adjusted, dtype=torch.float64)))
        chosen = record["candidates"][0] if p >= 0.5 else None
        candidate_scores = record["candidates"]
        candidate_lookup = {c["field"]: c["joint_score"] for c in candidate_scores}
        prediction = {
            "predicted_field": chosen["field"] if chosen else None,
            "predicted_value": None,  # filled from syntax only below
            "answerability_probability": p,
            "candidate_fields": candidate_scores,
            "field_scores": candidate_lookup,
        }
        # Determine the selected value using the same evidence-only parser, not
        # gold labels or target_field, at inference.
        if chosen is not None:
            from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
            field_candidates = complete_field_spans(
                row["information"], 16
            )
            matching = [c for c in field_candidates
                        if c["field"].lower() == chosen["field"]]
            if len(matching) != 1:
                raise ValueError("Candidate field mapping invalid")
            prediction["predicted_value"] = matching[0]["value"]
        outputs.append(enriched(row, prediction))
    return outputs


def scorecard(outcomes):
    s = summarize(outcomes)
    canon = s["canonical"]
    return {
        "dev_exact": canon["correct"],
        "total": canon["count"],
        "answerable_correct": sum(
            x["correct"] for x in outcomes if x["expected_value"] is not None
        ),
        "unanswerable_correct": sum(
            x["correct"] for x in outcomes if x["expected_value"] is None
        ),
        "unsupported_answer_errors": canon["unsupported_answers"],
        "false_abstentions": canon["false_abstentions"],
        "paired_both_correct": s["paired"]["both_correct"],
        "correct_switches": s["paired"]["correct_switches"],
        "brier_score": canon["brier_score"],
        "field_confusion": canon["field_confusion"],
    }


def run(checkpoint, output_dir, calibration_path, *, steps=1000, lr=0.05):
    for path in (output_dir, calibration_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output must be unused: {path}")
    torch.set_num_threads(1)
    model, metadata = load_bilinear_checkpoint(checkpoint)
    train_rows, dev_rows = read_partition("train"), read_partition("dev")
    train_records = [measure_row(model, row) for row in train_rows]
    dev_records = [measure_row(model, row) for row in dev_rows]
    fitted = fit_calibrator(train_records, steps=steps, lr=lr)
    train_lookup = {r["example_id"]: r for r in train_records}
    dev_lookup = {r["example_id"]: r for r in dev_records}
    baseline_train = scorecard(classify(train_rows, train_lookup))
    adjusted_train = scorecard(classify(train_rows, train_lookup, fitted))
    baseline_dev = scorecard(classify(dev_rows, dev_lookup))
    adjusted_dev = scorecard(classify(dev_rows, dev_lookup, fitted))
    report = {
        "experiment": "v5k_frozen_bilinear_joint_score_a_prime",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((DATA / "dev.jsonl").read_bytes()).hexdigest(),
        "calibration": fitted,
        "train": {"baseline": baseline_train, "adjusted": adjusted_train},
        "dev": {"baseline": baseline_dev, "adjusted": adjusted_dev},
        "baseline_checkpoint_untouched": True,
        "span_heads_frozen": True,
        "no_dev_training": True,
        "reserved_eval_accessed": False,
    }
    # Contract check: original baseline should be reproduced before comparing a'.
    if (baseline_dev["dev_exact"] != 36 or
            baseline_dev["correct_switches"] != 12 or
            baseline_dev["unsupported_answer_errors"] != 12):
        raise ValueError("Frozen baseline DEV results disagree with recorded V5J; refusing report")
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("x", encoding="utf-8") as f:
        json.dump({
            "checkpoint_sha256": report["checkpoint_sha256"],
            "train_sha256": report["train_sha256"],
            "feature": "max_joint",
            "calibration": fitted,
        }, f, indent=2, allow_nan=False)
        f.write("\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# V5K frozen bilinear + joint-score a′", "",
        "Frozen V5J encoder and heads; only lambda and bias_delta learned using TRAIN BCE.",
        "Joint-score feature is computed without gold requested-field information.",
        "Original 0.5 answerability threshold. DEV exploratory. Reserved EVAL untouched.",
        "",
        f"TRAIN-only feature mean={fitted['train_mean']:.6f}; std={fitted['train_std']:.6f}",
        f"lambda={fitted['lambda']:.6f}; bias_delta={fitted['bias_delta']:.6f}",
        "", "| Metric | Baseline DEV | a′ DEV |", "| --- | ---: | ---: |",
    ]
    for key in ("dev_exact", "answerable_correct", "unanswerable_correct",
                "unsupported_answer_errors", "false_abstentions",
                "paired_both_correct", "correct_switches", "brier_score"):
        lines.append(f"| {key} | {baseline_dev[key]} | {adjusted_dev[key]} |")
    lines += [
        "", "## Success interpretation", "",
        "- Aim: improve missing-target abstention while preserving answerable extraction and 12/12 field switches.",
        "- Unsupported-answer errors count mistakes; lower is better.",
        "- This is a TRAIN-fitted post-hoc calibration of a frozen model, not end-to-end training.",
        "- The diagnostic examined DEV before this experiment; these findings remain exploratory.",
        "- Scores may exploit syntactic differences in field availability; test a separately registered distribution later.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--calibration-path", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    run(args.checkpoint, args.output_dir, args.calibration_path,
        steps=args.steps, lr=args.lr)


if __name__ == "__main__":
    main()
