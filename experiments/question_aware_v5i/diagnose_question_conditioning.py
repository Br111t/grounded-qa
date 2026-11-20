"""V5I read-only TRAIN/DEV question-conditioning diagnostic.

For matched Leader/Location questions over identical evidence:
  * compares pooled question vectors;
  * compares actual field scores and chosen field;
  * swaps ONLY the pooled question vector into a fixed encoded evidence
    representation, isolating the direct V5I fusion/readout pathway.
This is a diagnostic intervention, not a prediction or a training step.

Never loads or generates the reserved EVAL split.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from grounded_qa.models.span_model_v5 import prepare_span_input
from grounded_qa.models.span_model_v5i import (
    QuestionConditionedSpanQA, load_v5i_checkpoint,
)

FIELDS = ("leader", "location")


@torch.no_grad()
def forward_parts(model, row):
    """Capture encoder hidden states without altering forward behavior."""
    ids, mask = prepare_span_input(row["information"], row["question"])
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = model.encoder.register_forward_hook(capture)
    try:
        starts, ends, answer_logit = model(ids, mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise AssertionError("Expected exactly one encoder invocation")
    hidden = captured[0]
    qmask = model.question_mask(ids, mask)
    q = (hidden * qmask.unsqueeze(-1)).sum(dim=1) / qmask.sum(dim=1, keepdim=True)
    return {
        "ids": ids, "mask": mask, "hidden": hidden, "question": q,
        "starts": starts, "ends": ends,
        "answer_probability": float(torch.sigmoid(answer_logit)[0].item()),
    }


@torch.no_grad()
def scores_for_question(model, parts, question_vector, candidates):
    """Fix encoder states; change only q before fusion/readout."""
    hidden = parts["hidden"]
    if question_vector.shape != (hidden.shape[0], model.config.d_model):
        raise ValueError("Invalid question vector shape")
    combined = torch.cat(
        (hidden, question_vector.unsqueeze(1).expand_as(hidden)), dim=-1
    )
    fused = hidden + torch.tanh(model.question_fusion(combined))
    starts = model.start_head(fused).squeeze(-1).masked_fill(
        ~parts["mask"], -1e9
    )
    ends = model.end_head(fused).squeeze(-1).masked_fill(
        ~parts["mask"], -1e9
    )
    scores = {}
    for c in candidates:
        field = c["field"].lower()
        if field not in FIELDS:
            continue
        if field in scores:
            raise ValueError("Duplicate field span")
        s, e = c["start"], c["end"]
        if not parts["mask"][0, s:e + 1].all().item():
            raise ValueError("Candidate outside evidence")
        scores[field] = float((starts[0, s] + ends[0, e]).item())
    return scores


def select(scores):
    # The full evaluation decoder also considers Topic candidates and
    # answerability. This returns only the relative supervised-field winner.
    return max(scores, key=scores.get) if scores else None


def margin(scores, requested):
    rival = "location" if requested == "leader" else "leader"
    if requested not in scores or rival not in scores:
        return None
    return scores[requested] - scores[rival]


@torch.no_grad()
def analyze_pair(model, rows):
    if set(rows) != set(FIELDS):
        raise ValueError("Expected a canonical Leader/Location pair")
    a, b = rows["leader"], rows["location"]
    if a["information"] != b["information"] or a["source_id"] != b["source_id"]:
        raise ValueError("Pair has different evidence")
    candidates = complete_field_spans(a["information"], model.config.max_span_bytes)
    leader = forward_parts(model, a)
    location = forward_parts(model, b)
    # Independent question-aware score paths; should reproduce the model's
    # actual start/end logits when used with the original question vector.
    ll = scores_for_question(model, leader, leader["question"], candidates)
    lo = scores_for_question(model, leader, location["question"], candidates)
    oo = scores_for_question(model, location, location["question"], candidates)
    ol = scores_for_question(model, location, leader["question"], candidates)
    for parts, scores in [(leader, ll), (location, oo)]:
        for c in candidates:
            if c["field"].lower() not in FIELDS:
                continue
            direct = float((
                parts["starts"][0, c["start"]] + parts["ends"][0, c["end"]]
            ).item())
            if abs(scores[c["field"].lower()] - direct) > 1e-4:
                raise AssertionError("Isolated score differs from actual forward")
    ql, qo = leader["question"][0], location["question"][0]
    qdistance = float(torch.linalg.vector_norm(ql - qo).item())
    qcosine = float(torch.nn.functional.cosine_similarity(
        ql.unsqueeze(0), qo.unsqueeze(0), dim=1
    )[0].item())
    # Counterfactual: hold encoded states fixed, change only question pooling.
    def max_delta(first, second):
        shared = set(first) & set(second)
        return max((abs(first[k] - second[k]) for k in shared), default=None)
    return {
        "source_id": a["source_id"],
        "both_fields_present": set(FIELDS).issubset(ll) and set(FIELDS).issubset(oo),
        "question_vector_l2": qdistance,
        "question_vector_cosine": qcosine,
        "leader_question": {
            "observed_scores": ll,
            "observed_margin": margin(ll, "leader"),
            "observed_winner": select(ll),
            "swapped_question_scores": lo,
            "fixed_encoder_max_score_delta": max_delta(ll, lo),
            "answer_probability": leader["answer_probability"],
        },
        "location_question": {
            "observed_scores": oo,
            "observed_margin": margin(oo, "location"),
            "observed_winner": select(oo),
            "swapped_question_scores": ol,
            "fixed_encoder_max_score_delta": max_delta(oo, ol),
            "answer_probability": location["answer_probability"],
        },
        "actual_max_score_delta": max_delta(ll, oo),
        "actual_correct_switch": (
            set(FIELDS).issubset(ll)
            and select(ll) == "leader" and select(oo) == "location"
        ),
    }


def avg(entries, key):
    values = [r[key] for r in entries if r[key] is not None]
    return sum(values) / len(values) if values else None


def summarize(records):
    both = [r for r in records if r["both_fields_present"]]
    return {
        "source_pairs": len(records),
        "both_fields_present": len(both),
        "correct_switches_among_both_present": sum(
            r["actual_correct_switch"] for r in both
        ),
        "mean_question_vector_l2": avg(records, "question_vector_l2"),
        "mean_question_vector_cosine": avg(records, "question_vector_cosine"),
        "mean_actual_max_field_score_delta": avg(
            both, "actual_max_score_delta"
        ),
        "mean_fixed_encoder_leader_question_delta": avg(
            [r["leader_question"] for r in both],
            "fixed_encoder_max_score_delta",
        ),
        "mean_fixed_encoder_location_question_delta": avg(
            [r["location_question"] for r in both],
            "fixed_encoder_max_score_delta",
        ),
        "mean_leader_question_margin": avg(
            [r["leader_question"] for r in both], "observed_margin"
        ),
        "mean_location_question_margin": avg(
            [r["location_question"] for r in both], "observed_margin"
        ),
    }


def run(checkpoint: Path, output_dir: Path):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Use a fresh diagnostic output directory")
    model, info = load_v5i_checkpoint(checkpoint)
    if info.get("experiment_variant") != "v5i_question_conditioned":
        raise ValueError("Expected the V5I experiment variant")
    expected_train = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    if info.get("training_sha256") != expected_train:
        raise ValueError("TRAIN data do not match checkpoint")
    model.eval()
    results = {}
    for partition in ("train", "dev"):
        rows = read_partition(partition)
        pairs = defaultdict(dict)
        for row in rows:
            if row["question_variant"] != "canonical" or row["target_field"] not in FIELDS:
                raise ValueError("Unexpected question in registered TRAIN/DEV")
            field = row["target_field"]
            if field in pairs[row["source_id"]]:
                raise ValueError("Duplicate question field for source")
            pairs[row["source_id"]][field] = row
        records = [
            analyze_pair(model, pairs[source])
            for source in sorted(pairs)
        ]
        results[partition] = {
            "dataset_sha256": hashlib.sha256(
                (DATA / f"{partition}.jsonl").read_bytes()
            ).hexdigest(),
            "summary": summarize(records),
            "pairs": records,
        }
    report = {
        "diagnostic": "v5i_question_vector_intervention",
        "model_checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "no_training": True,
        "reserved_eval_accessed": False,
        "note": (
            "Fixed-encoder question-vector swap isolates the DIRECT fusion path. "
            "Actual observed differences also include encoder changes when "
            "question tokens change. These comparisons are diagnostic, not "
            "causal proof of learned abstraction or generalization."
        ),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "question_sensitivity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    lines = [
        "# V5I TRAIN/DEV question-conditioning diagnostic",
        "",
        "Read-only checkpoint, no training, no reserved EVAL access.",
        "",
        "| Metric | TRAIN | DEV |",
        "| --- | ---: | ---: |",
    ]
    metrics = (
        "source_pairs", "both_fields_present",
        "correct_switches_among_both_present",
        "mean_question_vector_l2", "mean_question_vector_cosine",
        "mean_actual_max_field_score_delta",
        "mean_fixed_encoder_leader_question_delta",
        "mean_fixed_encoder_location_question_delta",
        "mean_leader_question_margin", "mean_location_question_margin",
    )
    for metric in metrics:
        values = []
        for partition in ("train", "dev"):
            value = results[partition]["summary"][metric]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append(f"| {metric} | {values[0]} | {values[1]} |")
    lines.extend([
        "", "## Interpret carefully",
        "",
        "- Direct fusion sensitivity is measured by swapping the question vector",
        "  while **holding encoder evidence states fixed**.",
        "- Observed field-score differences additionally include changes in",
        "  encoder hidden states when the question text changes.",
        "- Nonzero sensitivity does not imply the correct field is selected.",
        "- Both-field winners here exclude Topic and answerability; use the",
        "  established V5I DEV metrics for actual final accuracy.",
        "- Compare TRAIN versus DEV, then inspect per-source JSON for exceptions.",
        "- A finding on this registered DEV set is exploratory, not holdout proof.",
        "",
    ])
    (output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
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
