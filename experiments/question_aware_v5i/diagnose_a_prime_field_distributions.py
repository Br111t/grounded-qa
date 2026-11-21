"""Read-only V5K group distributions from the existing probability diagnostic.

Uses saved TRAIN/DEV per-row predictions; no model, data generation, training,
threshold fitting or EVAL. The requested field is an audit group, not an input.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("leader", "location")
FEATURES = ("max_joint", "calibrated_probability")
THRESHOLD = 0.5


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Need nonempty values and quantile in [0,1]")
    v = sorted(values)
    position = (len(v) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return v[lower] + (v[upper] - v[lower]) * (position - lower)


def describe(rows, key):
    v = [float(r[key]) for r in rows]
    return {
        "count": len(v), "min": min(v), "p25": quantile(v, 0.25),
        "median": quantile(v, 0.5), "p75": quantile(v, 0.75),
        "max": max(v), "mean": sum(v) / len(v),
    }


def grouped(records):
    result = {}
    for field in FIELDS:
        for present in (True, False):
            items = [r for r in records
                     if r["requested_field"] == field and r["answerable"] is present]
            if not items:
                raise ValueError(f"Missing group {field}, answerable={present}")
            name = f"{field}_{'present' if present else 'missing'}"
            result[name] = {
                "distribution": {f: describe(items, f) for f in FEATURES},
                "candidate_count_distribution": dict(sorted(Counter(
                    str(r["candidate_count"]) for r in items).items())),
                "top_scored_field_counts": dict(sorted(Counter(
                    r["top_candidate_field"] for r in items).items())),
                "false_abstentions": sum(
                    r["answerable"] and r["calibrated_probability"] < THRESHOLD
                    for r in items),
                "unsupported_answers": sum(
                    not r["answerable"] and r["calibrated_probability"] >= THRESHOLD
                    for r in items),
                "records": sorted(items, key=lambda r: r["calibrated_probability"]),
            }
    return result


def run(input_path, output_dir):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Choose an unused output directory: {output_dir}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if (data.get("analysis") != "v5k_read_only_probability_and_threshold_diagnostic"
            or data.get("reserved_eval_accessed") is not False
            or data.get("no_training") is not True):
        raise ValueError("Expected immutable TRAIN/DEV-only V5K diagnostic")
    records = data["records"]
    if set(records) != {"train", "dev"}:
        raise ValueError("Unexpected split names (EVAL must remain sealed)")
    if any(len(records[split]) != 48 for split in ("train", "dev")):
        raise ValueError("Expected 48 TRAIN and 48 DEV rows")
    result = {split: grouped(records[split]) for split in ("train", "dev")}
    dev_errors = [
        r for r in records["dev"]
        if r["answerable"] and r["calibrated_probability"] < THRESHOLD
    ]
    if len(dev_errors) != 3:
        raise ValueError("Expected three known DEV false abstentions")
    output = {
        "analysis": "v5k_max_joint_probability_by_field_presence",
        "source_report": str(input_path),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "calibration": data["calibration"],
        "threshold": THRESHOLD,
        "groups": result,
        "dev_false_abstentions": dev_errors,
        "no_training": True,
        "no_dev_threshold_selection": True,
        "reserved_eval_accessed": False,
    }
    lines = [
        "# V5K joint score and probability distributions by requested field", "",
        "Read-only regrouping of already recorded TRAIN and DEV predictions.",
        "Groups are for evaluation only: no target_field label is used in inference.",
        "Threshold remains 0.50. No EVAL, optimization or checkpoint writes.", "",
    ]
    for split in ("train", "dev"):
        lines += [
            f"## {split.upper()} distributions", "",
            "| Requested | Presence | n | Joint mean | Joint min | Joint median | Joint max | p mean | p min | p median | p max | Abstain errors | Unsupported errors |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field in FIELDS:
            for present in (True, False):
                group = result[split][f"{field}_{'present' if present else 'missing'}"]
                j = group["distribution"]["max_joint"]
                p = group["distribution"]["calibrated_probability"]
                lines.append(
                    f"| {field} | {'present' if present else 'missing'} | {j['count']} | "
                    f"{j['mean']:.3f} | {j['min']:.3f} | {j['median']:.3f} | {j['max']:.3f} | "
                    f"{p['mean']:.4f} | {p['min']:.4f} | {p['median']:.4f} | {p['max']:.4f} | "
                    f"{group['false_abstentions']} | {group['unsupported_answers']} |"
                )
        lines += ["", "### Candidate-count and top-field breakdown", "",
                  "| Requested | Presence | Candidate counts | Top-field counts |",
                  "| --- | --- | --- | --- |"]
        for field in FIELDS:
            for present in (True, False):
                group = result[split][f"{field}_{'present' if present else 'missing'}"]
                lines.append(
                    f"| {field} | {'present' if present else 'missing'} | "
                    f"{group['candidate_count_distribution']} | "
                    f"{group['top_scored_field_counts']} |"
                )
        lines.append("")
    lines += [
        "## DEV false abstentions, same registered 0.50 threshold", "",
        "| Example | Requested | Correct top field? | Candidate count | Joint score | Original p | Calibrated p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(dev_errors, key=lambda r: r["calibrated_probability"]):
        lines.append(
            f"| {row['example_id']} | {row['requested_field']} | "
            f"{row['correct_top_field_if_answered']} | {row['candidate_count']} | "
            f"{row['max_joint']:.3f} | {row['original_probability']:.5f} | "
            f"{row['calibrated_probability']:.5f} |"
        )
    lines += [
        "", "## Reading the results", "",
        "- Compare present/missing *within each requested field*, not only pooled means.",
        "- Check whether Location-present scores are systematically below Leader-present.",
        "- Check candidate-count distributions, since missing-target sources have fewer fields.",
        "- The pooled joint score may separate source structure, not semantic answerability.",
        "- Full per-group rows plus p25/p75 are retained in metrics.json.",
        "- DEV has been inspected repeatedly; no DEV-fitted threshold or new performance claim.", "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-metrics", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    run(args.input_metrics, args.output_dir)


if __name__ == "__main__":
    main()
