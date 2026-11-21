"""Read-only V5J bilinear answerability / span-score separation diagnostic.

The candidate scores are computed without using the target label to select a
candidate. Labels are used only for retrospective TRAIN/DEV grouping.
Reserved EVAL stays sealed; no checkpoint mutation or fitting on DEV.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5i.compare_interactions import InteractionSpanQA, EXPERIMENT
from grounded_qa.models.span_model_v5 import prepare_span_input


def load_bilinear_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("experiment") != EXPERIMENT
            or payload.get("interaction") != "bilinear"):
        raise ValueError("Expected V5J bilinear checkpoint")
    meta = payload["metadata"]
    if meta.get("experiment_variant") != EXPERIMENT or meta.get("interaction") != "bilinear":
        raise ValueError("Invalid checkpoint metadata")
    train_sha = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    if meta.get("training_sha256") != train_sha:
        raise ValueError("TRAIN dataset checksum mismatch")
    dev_sha = hashlib.sha256((DATA / "dev.jsonl").read_bytes()).hexdigest()
    if meta.get("dev_sha256") != dev_sha:
        raise ValueError("DEV dataset checksum mismatch")
    model = InteractionSpanQA("bilinear")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, meta


def rank_auc(pos, neg):
    """Empirical AUC: P(score_positive > score_negative), 0.5 for ties."""
    if not pos or not neg:
        raise ValueError("Both classes required")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def summary(rows, key):
    positive = [r[key] for r in rows if r["answerable"]]
    negative = [r[key] for r in rows if not r["answerable"]]
    return {
        "answerable_count": len(positive), "missing_count": len(negative),
        "answerable_mean": sum(positive) / len(positive),
        "missing_mean": sum(negative) / len(negative),
        "answerable_min": min(positive), "answerable_max": max(positive),
        "missing_min": min(negative), "missing_max": max(negative),
        "auc_high_implies_answerable": rank_auc(positive, negative),
    }


@torch.no_grad()
def measure_row(model, row):
    ids, mask = prepare_span_input(row["information"], row["question"])
    start, end, a = model(ids, mask)
    candidates = complete_field_spans(row["information"], model.config.max_span_bytes)
    if not candidates:
        raise ValueError("Expected at least one evidence candidate")
    scores = []
    for candidate in candidates:
        s, e = candidate["start"], candidate["end"]
        scores.append({
            "field": candidate["field"].lower(),
            "start_score": float(start[0, s]),
            "end_score": float(end[0, e]),
            "joint_score": float(start[0, s] + end[0, e]),
        })
    scores.sort(key=lambda c: -c["joint_score"])
    ends = sorted((c["end_score"] for c in scores), reverse=True)
    joints = [c["joint_score"] for c in scores]
    return {
        "example_id": row["example_id"],
        "source_id": row["source_id"],
        "question_field": row["target_field"],  # evaluation label; not inference feature
        "answerable": row["expected_value"] is not None,
        "answerability_logit": float(a[0]),
        "answer_probability": float(torch.sigmoid(a[0])),
        "max_end": ends[0],
        "end_gap_top2": ends[0] - ends[1] if len(ends) >= 2 else 0.0,
        "max_joint": max(joints),
        "joint_gap_top2": joints[0] - joints[1] if len(joints) >= 2 else 0.0,
        "max_end_minus_mean": ends[0] - sum(ends) / len(ends),
        "max_joint_minus_mean": max(joints) - sum(joints) / len(joints),
        "candidate_count": len(scores),
        "candidates": scores,
    }


def run(checkpoint, output_dir):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Use a new output directory")
    torch.set_num_threads(1)
    model, meta = load_bilinear_checkpoint(checkpoint)
    results = {}
    features = (
        "answerability_logit", "answer_probability", "max_end",
        "end_gap_top2", "max_joint", "joint_gap_top2",
        "max_end_minus_mean", "max_joint_minus_mean",
    )
    for split in ("train", "dev"):
        rows = [measure_row(model, r) for r in read_partition(split)]
        results[split] = {
            "records": rows,
            "summary": {name: summary(rows, name) for name in features},
        }
    payload = {
        "analysis": "v5j_bilinear_answerability_score_separation",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata": meta, "features": features, "results": results,
        "no_training": True, "reserved_eval_accessed": False,
    }
    lines = [
        "# Read-only bilinear: present vs missing field-score separation", "",
        "AUC = 0.5 indicates no rank separation; 1.0 indicates perfect rank",
        "separation with larger scores for answerable questions. AUC is descriptive,",
        "computed independently on TRAIN and DEV; no threshold was optimized.",
        "The requested-field label is NEVER passed to the model or feature summary.", "",
        "| Feature | TRAIN AUC | DEV AUC | TRAIN present mean | TRAIN missing mean | DEV present mean | DEV missing mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature in features:
        t, d = (results[p]["summary"][feature] for p in ("train", "dev"))
        lines.append(
            f"| {feature} | {t['auc_high_implies_answerable']:.3f} | "
            f"{d['auc_high_implies_answerable']:.3f} | "
            f"{t['answerable_mean']:.3f} | {t['missing_mean']:.3f} | "
            f"{d['answerable_mean']:.3f} | {d['missing_mean']:.3f} |"
        )
    lines += [
        "", "## Caveats", "",
        "- Maximum end and joint scores are confidence proxies, NOT direct evidence that the requested field exists.",
        "- Top-two gaps are zero when fewer than two candidate fields exist.",
        "- Candidate count and field presence structure may confound group comparisons.",
        "- Each source contributes two correlated questions; this exploratory AUC is not an independent-sample confidence interval.",
        "- If features overlap heavily, adding them to answerability may not fix abstention.",
        "- The reserved EVAL set was not accessed.", "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    run(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
