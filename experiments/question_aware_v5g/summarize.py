"""Readable V5G TRAIN/DEV composition and DEV prediction summary.

Run after DEV evaluation. Reserved EVAL is never read or generated.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from experiments.question_aware_v5g.data_design import read_partition


def tab(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def build_report(train, dev, predictions):
    report = ["# V5G: Data and Development Prediction Summary", "",
              "DEV only; the reserved evaluation set is not accessed.", ""]
    for label, data in (("Training", train), ("Development", dev)):
        sources = {r["source_id"]: r["information"] for r in data}
        labels = Counter((r["target_field"], r["answerability"]) for r in data)
        structures = Counter(
            "+".join(f for f in ("Leader", "Location", "Topic") if f + ":" in info)
            for info in sources.values()
        )
        report += [f"## {label} data", "",
                   f"{len(sources)} evidence sources and {len(data)} question rows.", "",
                   tab(["Question", "Answerable", "Missing target"],
                       [(f, labels[(f, "answerable")], labels[(f, "unanswerable")])
                        for f in ("leader", "location")]), "",
                   "**Field presence in evidence**", "",
                   tab(["Fields present", "Sources"], sorted(structures.items())), ""]
    n = len(predictions)
    correct = sum(bool(r["correct"]) for r in predictions)
    failure = Counter((r["target_field"], r["predicted_field"] or "abstain")
                      for r in predictions)
    pairs = defaultdict(dict)
    for row in predictions:
        if row["question_variant"] == "canonical":
            pairs[row["source_id"]][row["target_field"]] = row
    both_correct = sum(p["leader"]["correct"] and p["location"]["correct"]
                       for p in pairs.values())
    both_present = [p for p in pairs.values()
                    if p["leader"]["expected_value"] is not None
                    and p["location"]["expected_value"] is not None]
    correct_switch = sum(p["leader"]["predicted_field"] == "leader"
                         and p["location"]["predicted_field"] == "location"
                         for p in both_present)
    report += ["## Development predictions", "",
               f"Exact accuracy: **{correct}/{n} ({correct/n:.1%})**.",
               f"Both questions correct on same evidence: **{both_correct}/{len(pairs)}**.",
               f"Correct field switches when both fields exist: **{correct_switch}/{len(both_present)}**.",
               f"False abstentions: **{sum(r['false_abstention'] for r in predictions)}**.", "",
               tab(["Asked for", "Predicted", "Count"],
                   [(a, b, count) for (a, b), count in sorted(failure.items())]), ""]
    for name, condition in (
        ("Wrong-field selection", lambda r: r["target_field"] == "leader" and r["predicted_field"] == "location"),
        ("False abstention", lambda r: r["false_abstention"]),
        ("Correct extraction", lambda r: r["correct"] and r["predicted_field"] is not None),
        ("Correct abstention", lambda r: r["correct"] and r["predicted_field"] is None),
    ):
        matches = sorted((r for r in predictions if condition(r)), key=lambda r: r["source_id"])
        if not matches:
            continue
        r = matches[0]
        chosen = (f"{r['predicted_field'].title()}: {r['predicted_value']}"
                  if r["predicted_field"] else "Insufficient information.")
        report += [f"### Example: {name}", "",
                   f"Source: `{r['source_id']}`", "",
                   f"Evidence: {r['information']}", "",
                   f"Question: {r['question']}", "",
                   f"Expected: **{r['answer']}**", "",
                   f"Actual: **{chosen}**", "",
                   f"Answerability probability: {r['answerability_probability']:.4f}", "",
                   tab(["Candidate field", "Joint logit score"],
                       [(c["field"], f"{c['joint_score']:.3f}") for c in r["candidate_fields"]]), ""]
    report += ["## Interpretation notes", "",
               "A correct abstention differs from an incorrect refusal to answer.",
               "Joint-logit scores are not calibrated probabilities.",
               "This report does not measure unseen paraphrases, unrelated questions or Topic transfer.",
               "Do not run reserved EVAL until the model and evaluator are frozen.", ""]
    return "\n".join(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Refusing to overwrite an existing report")
    train, dev = read_partition("train"), read_partition("dev")
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    actual = {r["example_id"]: (r["information"], r["question"]) for r in rows}
    expected = {r["example_id"]: (r["information"], r["question"]) for r in dev}
    if actual != expected or len(rows) != len(dev):
        raise ValueError("Predictions do not correspond to registered DEV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(train, dev, rows), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
