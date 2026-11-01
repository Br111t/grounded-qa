"""Read-only V5D development evaluation with JSON + Markdown run reports."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.balanced_boundary_v5d.generate_data_v5d import GROUPS, PATHS, read_split
from experiments.variable_span_v5b.evaluate_span_v5b import evaluate_group
from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from grounded_qa.models.span_model_v5 import load_span_checkpoint


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    partitions = {group: [] for group in GROUPS}
    for row in rows:
        name = parse_fields(row["information"], row["source_id"])["Leader"]
        group = f"length{len(name)}_{'vowel' if name[-1].lower() in 'aeiou' else 'consonant'}"
        partitions[group].append(row)
    if any(len(part) != 24 for part in partitions.values()):
        raise ValueError("V5D development cell counts changed")
    return partitions


def report(checkpoint: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Reports are immutable; choose a new directory: {output_dir}")
    rows = read_split("dev")
    model, info = load_span_checkpoint(checkpoint)
    if info.get("answerability_threshold") != 0.5:
        raise ValueError("Unexpected checkpoint answerability threshold")
    if info.get("experiment") == "v5d_balanced_boundaries":
        expected = hashlib.sha256(PATHS["train"].read_bytes()).hexdigest()
        if info.get("extra_training_sha256") != expected:
            raise ValueError("Checkpoint training-file hash mismatch")
    scores = {}
    for group, cell in grouped(rows).items():
        score = evaluate_group(model, cell, show_errors=5)
        scores[group] = {
            key: value for key, value in score.items()
            if key != "mismatches"
        }
        scores[group]["sample_errors"] = [
            {"id": identifier, "predicted": prediction,
             "expected": expected, "span": span}
            for identifier, prediction, expected, span in score["mismatches"]
        ]
    exact = sum(cell["correct"] for cell in scores.values())
    ends = sum(cell["end_head"] for cell in scores.values())
    starts = sum(cell["start_head"] for cell in scores.values())
    result = {
        "experiment": "v5d_balanced_boundaries",
        "evaluation_partition": "dev_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(PATHS["dev"].read_bytes()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_metadata": info,
        "overall": {"exact": exact, "total": len(rows),
                    "start_head": starts, "end_head": ends},
        "groups": scores,
    }
    lines = [
        "# V5D run summary",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`",
        f"DEV dataset SHA256: `{result['dev_sha256']}`",
        "",
        "## Observations (DEV only)",
        f"Exact: {exact}/{len(rows)} ({exact/len(rows):.1%}); "
        f"start head: {starts}/{len(rows)}; end head: {ends}/{len(rows)}",
        "",
        "| Group | Exact | Start head | End head | Truncated | Overextended | False abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, score in scores.items():
        lines.append(
            f"| {group} | {score['correct']}/24 | {score['start_head']}/24 | "
            f"{score['end_head']}/24 | {score['too_short']} | "
            f"{score['too_long']} | {score['false_abstain']} |"
        )
    lines.extend([
        "", "## Interpretation",
        "Record observations first. Any causal explanation is a hypothesis, not a proven mechanism.",
        "",
        "## Next test",
        "Compare against unchanged V5/V5B checkpoints on the **same V5D DEV records**. "
        "Report both regressions and improvements. Do not repeatedly tune on DEV.",
        "",
        "No final holdout accessed; no model weights changed.",
        "",
    ])
    # Create directory only after all inference and formatting succeed.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Immutable report written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        report(args.checkpoint, args.output_dir)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5D evaluation stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
