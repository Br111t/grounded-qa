"""V5F: read-only counterfactual question sensitivity diagnostic.

Keep evidence byte-for-byte unchanged, vary ONLY the question. Evaluate
frozen V5/V5B/V5D weights and complete-field candidate scores. This does NOT
claim models trained on Leader questions know Location/Topic or abstention
for unrelated questions.

Run from repository root; PYTHONPATH must include src.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from grounded_qa.models.span_model_v5 import (
    FALLBACK, load_span_checkpoint, prepare_span_input,
)
from grounded_qa.tokenization.tokenizer import decode

CHECKPOINTS = {
    "v5": Path("experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt"),
    "v5b": Path("experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt"),
    "v5d": Path("experiments/balanced_boundary_v5d/checkpoints/v5d_balanced_seed42_3000.pt"),
}
EXPECTED_SHA256 = {
    "v5": "d29a7664129364d81cce8fb875db92296338ce1a2d1c77e7098062a9ae8b88a8",
    "v5b": "616e4e5e3324655a8a93a3bf4f73c1825e32ebed8b88f18bbbe393d4d828f2c1",
    "v5d": "2a256d2e29a9d71c72d5ffd0cdc6180089edb96fb2606fd2efcdca6287c7e165",
}
# These are diagnostic prompts, not trained target types.
QUESTIONS = (
    ("leader_original", "leader", "What is the leader of Workshop W0001?"),
    ("leader_paraphrase", "leader", "Who leads Workshop W0001?"),
    ("leader_reworded", "leader", "Which person is in charge of Workshop W0001?"),
    ("location", "location", "Where is Workshop W0001 located?"),
    ("topic", "topic", "What is the topic of Workshop W0001?"),
    ("unrelated", "unrelated", "What will the weather be tomorrow?"),
)
EVIDENCE = (
    ("a", "Fictional Workshop W0001. Location: Room Elm. Leader: Dbfi. Topic: Python."),
    ("b", "Fictional Workshop W0001. Topic: Astronomy. Leader: Dblt. Location: Room Cedar."),
    ("c", "Fictional Workshop W0001. Leader: Ebfet. Location: Room Pine. Topic: Gardening."),
    ("d", "Fictional Workshop W0001. Topic: Python. Location: Room Ash. Leader: Ebaa."),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_rows():
    """A small exploratory probe: four fixed evidence blocks × six questions."""
    rows = []
    for evidence_id, information in EVIDENCE:
        candidates = complete_field_spans(information)
        values = {c["field"].lower(): c["value"] for c in candidates}
        if set(values) != {"leader", "location", "topic"}:
            raise ValueError("Expected three independently answerable source fields")
        for question_id, target, question in QUESTIONS:
            rows.append({
                "evidence_id": evidence_id,
                "information": information,
                "information_sha256": hashlib.sha256(information.encode()).hexdigest(),
                "question_id": question_id,
                "target_field": target,
                "question": question,
                "expected_field": target if target != "unrelated" else None,
                "expected_value": values.get(target),
                "expect_abstain": target == "unrelated",
            })
    return rows


@torch.no_grad()
def probe_one(model, row):
    tokens, mask = prepare_span_input(row["information"], row["question"])
    was_training = model.training
    model.eval()
    try:
        starts, ends, answer_logits = model(tokens, mask)
        probability = float(torch.sigmoid(answer_logits[0]).item())
        fields = complete_field_spans(row["information"], model.config.max_span_bytes)
        if len(fields) != 3:
            raise ValueError("Missing complete field from evidence")
        candidates = []
        for field in fields:
            s, e = field["start"], field["end"]
            if not mask[0, s:e + 1].all().item():
                raise ValueError("Field outside source mask")
            if decode(tokens[0, s:e + 1].tolist()) != field["value"]:
                raise ValueError("Field slice does not decode to expected value")
            candidates.append({
                **field,
                "start_logit": float(starts[0, s].item()),
                "end_logit": float(ends[0, e].item()),
                "joint_score": float((starts[0, s] + ends[0, e]).item()),
            })
        candidates.sort(key=lambda c: (-c["joint_score"], c["start"]))
        winner = candidates[0]
        predicted_field = winner["field"].lower() if probability >= 0.5 else None
        predicted_value = winner["value"] if predicted_field is not None else None
        return {
            "answerability_probability": probability,
            "predicted_field": predicted_field,
            "predicted_value": predicted_value,
            "predicted_text": FALLBACK if predicted_field is None
                              else f"{winner['field']}: {winner['value']}.",
            "candidate_fields": candidates,
            "scores_by_field": {c["field"].lower(): c["joint_score"] for c in candidates},
            "selection_margin": candidates[0]["joint_score"] - candidates[1]["joint_score"],
            "question_field_correct": (
                predicted_field == row["expected_field"]
                if row["expected_field"] is not None else None),
            "unrelated_abstained": predicted_field is None if row["expect_abstain"] else None,
        }
    finally:
        model.train(was_training)


def compare_pairs(rows):
    """Measures changes against identical evidence's original Leader question."""
    keyed = {(r["model"], r["evidence_id"], r["question_id"]): r for r in rows}
    comparisons = []
    for row in rows:
        if row["question_id"] == "leader_original":
            continue
        original = keyed[(row["model"], row["evidence_id"], "leader_original")]
        before = original["scores_by_field"]
        after = row["scores_by_field"]
        target = row["target_field"]
        target_advantage = None
        if target in ("leader", "location", "topic"):
            target_advantage = after[target] - max(
                score for field, score in after.items() if field != target)
        comparisons.append({
            "model": row["model"], "evidence_id": row["evidence_id"],
            "question_id": row["question_id"], "target_field": target,
            "original_selected_field": original["predicted_field"],
            "changed_selected_field": row["predicted_field"],
            "selection_changed": original["predicted_field"] != row["predicted_field"],
            "answerability_delta": row["answerability_probability"]
                                   - original["answerability_probability"],
            "score_deltas": {field: after[field] - before[field] for field in before},
            "target_advantage": target_advantage,
            "question_field_correct": row["question_field_correct"],
            "unrelated_abstained": row["unrelated_abstained"],
        })
    return comparisons


def summarize(records, comparisons):
    lines = [
        "# V5F · Counterfactual question sensitivity", "",
        "Exploratory and read-only; unchanged evidence for each question pair.",
        "Models were trained primarily on Leader questions. Different question "
        "types are out-of-distribution tests, not expected trained competencies.",
        "", "## Selected fields by question type", "",
        "| Model | Question | Leader | Location | Topic | Abstained | Correct target |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    aggregates = []
    for name in CHECKPOINTS:
        for question_id, target, _ in QUESTIONS:
            subset = [r for r in records
                      if r["model"] == name and r["question_id"] == question_id]
            if len(subset) != len(EVIDENCE):
                raise ValueError("Incomplete question/evidence matrix")
            counts = {field: sum(r["predicted_field"] == field for r in subset)
                      for field in ("leader", "location", "topic")}
            abstained = sum(r["predicted_field"] is None for r in subset)
            correct = sum(r["question_field_correct"] is True for r in subset)
            if target == "unrelated":
                correct = sum(r["unrelated_abstained"] is True for r in subset)
            entry = {"model": name, "question_id": question_id, "target": target,
                     "selected": counts, "abstained": abstained, "correct": correct,
                     "total": len(subset)}
            aggregates.append(entry)
            lines.append(
                f"| {name.upper()} | {question_id} | {counts['leader']} | "
                f"{counts['location']} | {counts['topic']} | {abstained} | "
                f"{correct}/{len(subset)} |"
            )
    lines += [
        "", "## Paired changes against original Leader question", "",
        "| Model | Question | Field selection changed | Target advantage (individual examples) |",
        "|---|---|---:|---|",
    ]
    for name in CHECKPOINTS:
        for question_id, _, _ in QUESTIONS[1:]:
            subset = [r for r in comparisons
                      if r["model"] == name and r["question_id"] == question_id]
            values = ", ".join(
                "n/a" if r["target_advantage"] is None
                else f"{r['target_advantage']:.2f}"
                for r in sorted(subset, key=lambda r: r["evidence_id"])
            )
            lines.append(
                f"| {name.upper()} | {question_id} | "
                f"{sum(r['selection_changed'] for r in subset)}/{len(subset)} | {values} |"
            )
    lines += ["", "## Reading these results", "",
              "- A changed selected field alone is not evidence of understanding.",
              "- A paraphrase should preserve the target field, but score stability "
              "is a separate observable measure.",
              "- An unrelated question has no evidence-grounded answer; abstention "
              "is desirable but not a learned capability we can assume.",
              "- This is a small, knowingly out-of-distribution diagnostic on "
              "seen and unseen name families; not an accuracy claim.",
              "- The prior V5E holdout is consumed and is not reused here.",
              ""]
    return aggregates, "\n".join(lines)


def run(output_dir, checkpoints=CHECKPOINTS):
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; never overwrite a prior run")
    if set(checkpoints) != set(EXPECTED_SHA256):
        raise ValueError("Must supply all three original checkpoints")
    hashes = {name: sha256(path) for name, path in checkpoints.items()}
    if hashes != EXPECTED_SHA256:
        raise ValueError("Checkpoint identity changed; aborting")
    designs = design_rows()
    results = []
    for name, path in checkpoints.items():
        model, _ = load_span_checkpoint(path)
        for row in designs:
            result = {"model": name, **row, **probe_one(model, row)}
            results.append(result)
    pairs = compare_pairs(results)
    aggregates, summary = summarize(results, pairs)
    if len(results) != 72 or len(pairs) != 60:
        raise ValueError("Unexpected paired matrix size")
    metadata = {
        "experiment": "v5f_question_counterfactual_v1",
        "checkpoint_sha256": hashes,
        "source_code_sha256": sha256(Path(__file__)),
        "evidence_count": len(EVIDENCE),
        "questions_per_evidence": len(QUESTIONS),
        "prediction_count": len(results),
        "pair_count": len(pairs),
        "selected_field_counts": aggregates,
        "prior_v5e_holdout_reused": False,
        "training_performed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, data in (
        ("all_predictions.json", results), ("paired_deltas.json", pairs),
        ("metrics.json", metadata),
    ):
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)


if __name__ == "__main__":
    main()
