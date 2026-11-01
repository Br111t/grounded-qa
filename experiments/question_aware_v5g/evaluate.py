"""V5G deterministic scorecard; one evaluation phase per output directory.

DEV may guide changes. EVAL is a reserved, single-use diagnostic; run only
after freezing architecture, decoder, checkpoint and acceptance conditions.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from grounded_qa.models.span_model_v5 import load_span_checkpoint, prepare_span_input
from grounded_qa.tokenization.tokenizer import decode


def predict(model, row):
    with torch.no_grad():
        ids, mask = prepare_span_input(row["information"], row["question"])
        start, end, answer_logit = model(ids, mask)
        p = float(torch.sigmoid(answer_logit[0]).item())
        fields = complete_field_spans(row["information"], model.config.max_span_bytes)
        candidates = []
        for c in fields:
            s, e = c["start"], c["end"]
            if not mask[0, s:e + 1].all().item() or decode(ids[0, s:e+1].tolist()) != c["value"]:
                raise ValueError("Invalid grounded candidate")
            candidates.append({**c, "joint_score": float((start[0, s] + end[0, e]).item())})
        candidates.sort(key=lambda c: (-c["joint_score"], c["start"]))
        chosen = candidates[0] if candidates and p >= 0.5 else None
        return {
            "predicted_field": chosen["field"].lower() if chosen else None,
            "predicted_value": chosen["value"] if chosen else None,
            "answerability_probability": p,
            "candidate_fields": candidates,
            "field_scores": {c["field"].lower(): c["joint_score"] for c in candidates},
        }


def enriched(row, prediction):
    target = row["target_field"]
    expected = row["expected_value"]
    requested_exists = expected is not None
    selected = prediction["predicted_field"]
    correct = (selected == target and prediction["predicted_value"] == expected
               if requested_exists else selected is None)
    scores = prediction["field_scores"]
    if requested_exists and target in scores:
        rivals = [score for field, score in scores.items() if field != target]
        gold_margin = scores[target] - max(rivals) if rivals else None
        gold_rank = 1 + sum(score > scores[target] for score in rivals)
    else:
        gold_margin, gold_rank = None, None
    return {
        **row, **prediction,
        "correct": correct,
        "gold_field_rank": gold_rank,
        "gold_field_margin": gold_margin,
        "false_abstention": requested_exists and selected is None,
        "unsupported_answer": not requested_exists and selected is not None,
        "brier_component": (prediction["answerability_probability"] -
                            (1.0 if requested_exists else 0.0)) ** 2,
    }


def summarize(rows):
    by_group = defaultdict(list)
    for row in rows:
        if row["target_field"] == "unrelated":
            group = "unsupported"
        elif row["target_field"] == "topic":
            group = "topic_transfer"
        elif row["question_variant"].startswith("paraphrase"):
            group = "paraphrase"
        else:
            group = "canonical"
        by_group[group].append(row)
    report = {}
    for group, entries in by_group.items():
        n = len(entries)
        report[group] = {
            "count": n,
            "correct": sum(r["correct"] for r in entries),
            "accuracy": sum(r["correct"] for r in entries) / n,
            "false_abstentions": sum(r["false_abstention"] for r in entries),
            "unsupported_answers": sum(r["unsupported_answer"] for r in entries),
            "brier_score": sum(r["brier_component"] for r in entries) / n,
            "field_confusion": dict(Counter(
                f"{r['target_field']} -> {r['predicted_field']}" for r in entries
            )),
        }
    pairs = defaultdict(dict)
    for row in rows:
        if row["question_variant"] == "canonical" and row["target_field"] in ("leader", "location"):
            pairs[row["source_id"]][row["target_field"]] = row
    paired = [p for p in pairs.values() if set(p) == {"leader", "location"}]
    correct_pairs = sum(p["leader"]["correct"] and p["location"]["correct"] for p in paired)
    switches = sum(p["leader"]["predicted_field"] != p["location"]["predicted_field"]
                   for p in paired)
    correct_switches = sum(
        p["leader"]["predicted_field"] == "leader" and
        p["location"]["predicted_field"] == "location" for p in paired
    )
    # Paired field score change when question changes; evidence held constant.
    sensitivity = []
    for p in paired:
        l, loc = p["leader"], p["location"]
        if "leader" in l["field_scores"] and "location" in l["field_scores"]:
            sensitivity.append({
                "source_id": l["source_id"],
                "leader_advantage_leader_question": (
                    l["field_scores"]["leader"] - l["field_scores"]["location"]),
                "location_advantage_location_question": (
                    loc["field_scores"]["location"] - loc["field_scores"]["leader"]),
                "location_score_delta": (
                    loc["field_scores"]["location"] - l["field_scores"]["location"]),
            })
    report["paired"] = {
        "count": len(paired), "both_correct": correct_pairs,
        "selection_switches": switches, "correct_switches": correct_switches,
        "question_sensitivity": sensitivity,
    }
    for field in ("leader", "location"):
        subset = [r for r in rows if r["target_field"] == field and
                  r["question_variant"] == "canonical"]
        if subset:
            report[f"{field}_canonical"] = {
                "total": len(subset), "correct": sum(r["correct"] for r in subset),
            }
    if "paraphrase" in by_group:
        variants = defaultdict(list)
        for r in by_group["paraphrase"]:
            variants[r["target_field"]].append(r)
        report["paraphrase_by_field"] = {
            field: {"total": len(v), "correct": sum(r["correct"] for r in v)}
            for field, v in variants.items()
        }
    return report


def run(checkpoint, partition, output, freeze=None):
    if output.exists():
        raise FileExistsError("Immutable output already exists")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if partition == "eval":
        if freeze is None:
            raise ValueError("Reserved evaluation requires DEV freeze")
        locked = json.loads(freeze.read_text(encoding="utf-8"))
        if (locked.get("phase") != "dev" or
            locked.get("checkpoint_sha256") != checkpoint_hash or
            locked.get("evaluation_code_sha256") != code_hash):
            raise ValueError("Checkpoint or evaluation code drifted after DEV freeze")
    elif freeze is not None:
        raise ValueError("DEV cannot consume a freeze manifest")
    rows = read_partition(partition)
    model, metadata = load_span_checkpoint(checkpoint)
    if metadata.get("experiment") != "v5g_leader_location_question_selection":
        raise ValueError("Expected V5G checkpoint")
    if metadata.get("training_sha256") != hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest():
        raise ValueError("Training dataset does not match checkpoint provenance")
    model.eval()
    predictions = [enriched(row, predict(model, row)) for row in rows]
    metrics = summarize(predictions)
    manifest = {
        "phase": partition, "checkpoint_sha256": checkpoint_hash,
        "evaluation_code_sha256": code_hash,
        "dataset_sha256": hashlib.sha256((DATA / f"{partition}.jsonl").read_bytes()).hexdigest(),
        "training_data_sha256": metadata["training_sha256"],
        "metrics": metrics,
        "evaluation_is_exploratory": partition == "dev",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.json").write_text(
        json.dumps(predictions, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if partition == "dev":
        (output / "freeze.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--phase", choices=("dev", "eval"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--freeze", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run(args.checkpoint, args.phase, args.output_dir, args.freeze)


if __name__ == "__main__":
    main()
