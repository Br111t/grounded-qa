"""Reconcile an immutable V5D logit report from its saved prediction records.

This is a read-only report repair: does not run models or overwrite artifacts.
Use on the existing logit_probe_run1 directory to fix exclusive-category
recovery counts and compute raw-end/joint logit score margins.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def enrich(r):
    r = dict(r)
    r["recovered_wrong_raw_end"] = bool(
        not r["raw_end_correct"] and r["selected_span_correct"])
    r["lost_correct_raw_end"] = bool(
        r["raw_end_correct"] and not r["selected_span_correct"])
    r["abstained"] = r["selected_start"] is None
    r["end_score_margin"] = round(
        r["best_end_score"] - r["gold_end_score"], 6)
    r["gold_joint_score"] = round(
        r["gold_start_score"] + r["gold_end_score"], 6)
    r["best_joint_score"] = r["top_joint_spans"][0]["score"]
    r["joint_score_margin"] = round(
        r["best_joint_score"] - r["gold_joint_score"], 6)
    if (r["exact"] != r["selected_span_correct"]
            or r["end_score_margin"] < -0.00001
            or r["joint_score_margin"] < -0.00001):
        raise ValueError("Prediction correctness or margin invariant failed")
    return r


def measure(rows):
    return {
        "n": len(rows),
        "exact": sum(r["exact"] for r in rows),
        "raw_end_correct": sum(r["raw_end_correct"] for r in rows),
        "selected_span_correct": sum(r["selected_span_correct"] for r in rows),
        "recovery": sum(r["recovered_wrong_raw_end"] for r in rows),
        "lost_correct_end": sum(r["lost_correct_raw_end"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "gold_end_top5": sum(r["gold_end_rank"] <= 5 for r in rows),
        "end_gap_le1": sum(r["end_score_margin"] <= 1 for r in rows),
        "end_gap_le3": sum(r["end_score_margin"] <= 3 for r in rows),
        "joint_gap_le1": sum(r["joint_score_margin"] <= 1 for r in rows),
        "joint_gap_le3": sum(r["joint_score_margin"] <= 3 for r in rows),
        "median_end_gap": round(median(r["end_score_margin"] for r in rows), 4),
        "median_joint_gap": round(median(r["joint_score_margin"] for r in rows), 4),
    }


def render(overview, examples):
    lines = [
        "# V5D corrected logit diagnostic", "",
        "Reconstructed from saved prediction logits, without retraining. "
        "Correctness and recovery are independent booleans.", "",
        "## Correctness and recovery", "",
        "| Model | Exact | Gold end top 5 | Recovery | Lost correct end | Abstained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in overview.items():
        lines.append(f"| {name.upper()} | {m['exact']}/{m['n']} | "
                     f"{m['gold_end_top5']}/{m['n']} | {m['recovery']} | "
                     f"{m['lost_correct_end']} | {m['abstained']} |")
    lines += ["", "## Raw end and joint-span score margins", "",
              "Margin = best score minus gold score (logits, NOT probabilities). "
              "Lower positive values mean closer competition.", "",
              "| Model | End gap <= 1 | End gap <= 3 | Joint gap <= 1 | Joint gap <= 3 | Median end gap | Median joint gap |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in overview.items():
        lines.append(f"| {name.upper()} | {m['end_gap_le1']}/{m['n']} | "
                     f"{m['end_gap_le3']}/{m['n']} | {m['joint_gap_le1']}/{m['n']} | "
                     f"{m['joint_gap_le3']}/{m['n']} | {m['median_end_gap']} | "
                     f"{m['median_joint_gap']} |")
    lines += ["", "## Recovered examples", "",
              "| Model | Expected | Predicted | End rank | End margin | Joint margin |",
              "|---|---|---|---:|---:|---:|"]
    for r in examples:
        lines.append(f"| {r['model'].upper()} | {r['expected']} | "
                     f"{r['predicted']} | {r['gold_end_rank']} | "
                     f"{r['end_score_margin']:.3f} | {r['joint_score_margin']:.3f} |")
    lines += ["", "## Caveats", "",
              "This is DEV-only descriptive measurement. Gold joint-span rank "
              "cannot be reconstructed from top-five joint candidates alone. "
              "The original exclusive `error_category` field is retained "
              "for provenance but must not be used to count recoveries.",
              "No final holdout accessed or checkpoint modified.", ""]
    return "\n".join(lines)


def reconcile(input_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Immutable output directory already exists: {output_dir}")
    old = json.loads((input_dir / "metrics.json").read_text(encoding="utf-8"))
    cases = json.loads((input_dir / "all_cases.json").read_text(encoding="utf-8"))
    if old.get("experiment") != "v5d_logit_diagnostic" or len(cases) != 288:
        raise ValueError("Expected 288 V5D development logit records")
    rows = [enrich(r) for r in cases]
    groups = defaultdict(list)
    for r in rows:
        groups[r["model"]].append(r)
        if r["checkpoint_sha256"] != old["checkpoint_sha256"][r["model"]]:
            raise ValueError("Checkpoint SHA256 mismatch")
    if set(groups) != {"v5", "v5b", "v5d"}:
        raise ValueError("Missing or unexpected model")
    overview = {name: measure(groups[name]) for name in ("v5", "v5b", "v5d")}
    if any(m["n"] != 96 or m["exact"] != m["selected_span_correct"]
           for m in overview.values()):
        raise ValueError("Model count or exact/span invariant failed")
    details = {
        "schema_version": 2, "source_dataset_sha256": old["dataset_sha256"],
        "source_checkpoint_sha256": old["checkpoint_sha256"],
        "measurements": overview,
        "per_group": {
            name: {group: measure([r for r in groups[name] if r["group"] == group])
                   for group in sorted({r["group"] for r in groups[name]})}
            for name in overview
        },
    }
    sample = sorted((r for r in rows if r["recovered_wrong_raw_end"]),
                    key=lambda r: (r["model"], r["example_id"]))[:10]
    summary = render(overview, sample)
    # Finish validation and rendering before creating the immutable directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "all_cases.corrected.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.corrected.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.corrected.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved consistent report files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        reconcile(args.input_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"Reconciliation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
