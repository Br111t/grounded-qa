"""V5L read-only paired-question score comparison.

Pair rows by source_id, not by inferred field label. Both canonical questions
must be available in the registered experimental dataset. This is an ORACLE
PAIRING DIAGNOSTIC, not a deployable single-question inference algorithm.
TRAIN/DEV only; original model/calibration untouched; EVAL sealed.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.question_aware_v5i.diagnose_a_prime_field_distributions import (
    describe,
)


def pair_sources(records):
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
    paired = []
    for source_id, examples in sorted(by_source.items()):
        if len(examples) != 2:
            raise ValueError(f"Expected two canonical questions per {source_id}")
        by_question = {r["requested_field"]: r for r in examples}
        if set(by_question) != {"leader", "location"}:
            raise ValueError("Expected exactly Leader and Location questions")
        leader, location = by_question["leader"], by_question["location"]
        if leader["question"] == location["question"]:
            raise ValueError("Distinct questions required")
        paired.append({
            "source_id": source_id,
            "leader_present": leader["answerable"],
            "location_present": location["answerable"],
            "leader_max_joint": leader["max_joint"],
            "location_max_joint": location["max_joint"],
            "leader_probability": leader["calibrated_probability"],
            "location_probability": location["calibrated_probability"],
            "joint_delta_location_minus_leader": location["max_joint"] - leader["max_joint"],
            "p_delta_location_minus_leader": location["calibrated_probability"] - leader["calibrated_probability"],
            "leader_top_field": leader["top_candidate_field"],
            "location_top_field": location["top_candidate_field"],
        })
    if len(paired) != 24:
        raise ValueError("Expected 24 paired sources")
    return paired


def run(input_metrics, output_dir):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory already exists")
    data = json.loads(input_metrics.read_text(encoding="utf-8"))
    if (data.get("analysis") != "v5k_read_only_probability_and_threshold_diagnostic"
            or data.get("no_training") is not True
            or data.get("reserved_eval_accessed") is not False
            or set(data.get("records", {})) != {"train", "dev"}):
        raise ValueError("Expected V5K TRAIN/DEV probability diagnostic only")
    results = {}
    for split in ("train", "dev"):
        pairs = pair_sources(data["records"][split])
        scenarios = {
            "both_present": [p for p in pairs if p["leader_present"] and p["location_present"]],
            "leader_missing": [p for p in pairs if not p["leader_present"] and p["location_present"]],
            "location_missing": [p for p in pairs if p["leader_present"] and not p["location_present"]],
        }
        if [len(scenarios[k]) for k in ("both_present","leader_missing","location_missing")] != [12,6,6]:
            raise ValueError("Unexpected paired source distribution")
        results[split] = {
            "pairs": pairs,
            "groups": {key: {
                "joint_difference": describe(rows, "joint_delta_location_minus_leader"),
                "probability_difference": describe(rows, "p_delta_location_minus_leader"),
            } for key, rows in scenarios.items()},
        }
    lines = [
        "# V5L — Paired-question comparison, diagnostic only", "",
        "Paired canonical Leader/Location questions for the same evidence source.",
        "**The pairing uses both questions and is not yet a single-question inference method.**",
        "No learned parameters, new thresholds, or EVAL access.", "",
        "| Split | Source condition | n | Mean (Location − Leader) joint | Min | Median | Max | Mean p difference |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev"):
        for key, group in results[split]["groups"].items():
            d, p = group["joint_difference"], group["probability_difference"]
            lines.append(
                f"| {split} | {key} | {d['count']} | {d['mean']:.3f} | "
                f"{d['min']:.3f} | {d['median']:.3f} | {d['max']:.3f} | {p['mean']:.4f} |"
            )
    lines += [
        "", "## Cautions", "",
        "- Difference in raw scores is subject to question/field scale shift.",
        "- In both-present sources, neither question should abstain; score difference is not evidence of absence.",
        "- Never infer the requested field by a gold target label in deployment.",
        "- Pairing and scenario labels are used for retrospective diagnostics only.",
        "- Report descriptive group separation, not an optimized DEV classification threshold.",
        "- This experiment informs a later single-question architecture; it does not establish generalization.", "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({
        "experiment": "v5l_paired_question_read_only",
        "source_metrics": str(input_metrics), "results": results,
        "no_training": True, "paired_questions_required": True,
        "reserved_eval_accessed": False,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.input_metrics, a.output_dir)


if __name__ == "__main__":
    main()
