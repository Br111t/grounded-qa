"""Paired DEV and single-pass frozen HOLDOUT decoder comparison.

No training. No gold access until both predictions are made. All outputs
are new immutable directories. Holdout execution requires a DEV freeze
manifest matching the exact code and three checkpoint hashes.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.balanced_boundary_v5d.generate_data_v5d import PATHS, read_split
from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from experiments.field_boundary_decoder_v5e.field_decoder import predict_complete_field
from grounded_qa.models.span_model_v5 import (
    FALLBACK, load_span_checkpoint, predict_span,
)

ROOT = Path("experiments/field_boundary_decoder_v5e")
CODE_FILES = [ROOT / "field_decoder.py", ROOT / "compare_decoders.py", ROOT / "generate_holdout.py"]
BASELINE_NAMES = ("v5", "v5b", "v5d")
DEV_HASH = "5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(checkpoints):
    return {
        "algorithm": "complete_field_values_joint_logit_v1",
        "code_sha256": {str(p): sha(p) for p in CODE_FILES},
        "checkpoints_sha256": {n: sha(p) for n, p in checkpoints.items()},
        "threshold": 0.5,
        "max_candidates": "all_syntactically_complete_fields",
    }


def verify_baseline(cases, model_name, row, predicted, selected):
    record = cases.get((model_name, row["example_id"]))
    if record is None:
        raise ValueError("Baseline example missing from frozen diagnostic")
    if record["predicted"] != predicted or (
        record["selected_start"], record["selected_end"]
    ) != (selected if selected is not None else (None, None)):
        raise ValueError("Baseline prediction drift; stopping comparison")


def compare_one(model, name, row, baseline_cases=None):
    baseline, probability, old_span = predict_span(
        model, row["information"], row["question"], threshold=0.5
    )
    alternative, alt_probability, new_span, candidate_scores = predict_complete_field(
        model, row["information"], row["question"], threshold=0.5
    )
    if abs(probability - alt_probability) > 1e-6:
        raise ValueError("Baseline and alternative answerability disagree")
    if baseline_cases is not None:
        verify_baseline(baseline_cases, name, row, baseline, old_span)
    # Gold only consulted after BOTH candidate algorithms have predicted.
    expected = row["answer"]
    old_ok = baseline == expected
    new_ok = alternative == expected
    status = ("improved" if new_ok and not old_ok else
              "regressed" if old_ok and not new_ok else "unchanged")
    leader = parse_fields(row["information"], row["source_id"]).get("Leader")
    group = ("missing" if leader is None else
             f"length{len(leader)}_{'vowel' if leader[-1].lower() in 'aeiou' else 'consonant'}")
    return {
        "model": name, "example_id": row["example_id"], "group": group,
        "baseline": baseline, "alternative": alternative, "expected": expected,
        "baseline_span": old_span, "alternative_span": new_span,
        "baseline_correct": old_ok, "alternative_correct": new_ok,
        "changed_prediction": baseline != alternative,
        "status": status, "answerability_probability": probability,
        "candidate_fields": candidate_scores,
    }


def aggregate(records):
    return {
        "total": len(records),
        "baseline_exact": sum(r["baseline_correct"] for r in records),
        "alternative_exact": sum(r["alternative_correct"] for r in records),
        "improved": sum(r["status"] == "improved" for r in records),
        "regressed": sum(r["status"] == "regressed" for r in records),
        "unchanged": sum(r["status"] == "unchanged" for r in records),
        "changed_predictions": sum(r["changed_prediction"] for r in records),
        "unchanged_correct": sum(r["status"] == "unchanged" and r["baseline_correct"]
                                 for r in records),
        "unchanged_wrong": sum(r["status"] == "unchanged" and not r["baseline_correct"]
                               and not r["alternative_correct"] for r in records),
    }


def summarize(records, phase):
    overall = {name: aggregate([r for r in records if r["model"] == name])
               for name in BASELINE_NAMES}
    groups = {
        name: {group: aggregate([r for r in records if r["model"] == name
                                  and r["group"] == group])
               for group in sorted({r["group"] for r in records})}
        for name in BASELINE_NAMES
    }
    for m in overall.values():
        if m["improved"] + m["regressed"] + m["unchanged"] != m["total"]:
            raise ValueError("Comparison partition invariant failed")
        if m["baseline_exact"] + m["improved"] - m["regressed"] != m["alternative_exact"]:
            raise ValueError("Accuracy delta invariant failed")
    lines = [
        f"# Complete-field decoder · {phase.upper()} comparison",
        "", "No training. Same model weights and inputs in both arms.",
        "A change can still be unchanged if both answers are wrong.",
        "", "| Model | Baseline exact | Field exact | Improved | Regressed | Unchanged | Changed predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in overall.items():
        lines.append(
            f"| {name.upper()} | {m['baseline_exact']}/{m['total']} | "
            f"{m['alternative_exact']}/{m['total']} | {m['improved']} | "
            f"{m['regressed']} | {m['unchanged']} | {m['changed_predictions']} |"
        )
    lines += ["", "## By group", "",
              "| Model | Group | Baseline exact | Field exact | Improved | Regressed | Unchanged |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, categories in groups.items():
        for group, m in categories.items():
            if m["total"]:
                lines.append(
                    f"| {name.upper()} | {group} | {m['baseline_exact']}/{m['total']} | "
                    f"{m['alternative_exact']}/{m['total']} | {m['improved']} | "
                    f"{m['regressed']} | {m['unchanged']} |"
                )
    lines += ["", "## Selected paired outcomes (first 3 by ID in each change class)", "",
              "| Model | Status | Expected | Baseline | Alternative |",
              "|---|---|---|---|---|"]
    for name in BASELINE_NAMES:
        for state in ("improved", "regressed", "unchanged"):
            matches = sorted((r for r in records if r["model"] == name
                              and r["status"] == state),
                             key=lambda r: r["example_id"])[:3]
            for r in matches:
                lines.append(
                    f"| {name.upper()} | {state} | {r['expected']} | "
                    f"{r['baseline']} | {r['alternative']} |"
                )
    lines += ["", "## Interpretation constraints", "",
              "The alternative only constrains candidate spans to complete field "
              "values. It does **not** identify the field requested by the question. "
              "It is a deliberately narrow synthetic-grammar decoder experiment.",
              "DEV is exploratory; HOLDOUT is single-use and cannot be used to "
              "choose another decoder rule without a new untouched test set.",
              ""]
    return overall, groups, "\n".join(lines)


def load_dev_baselines(path, checkpoint_hashes):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 288:
        raise ValueError("Expected exactly 288 saved baseline predictions")
    ids = {}
    for r in rows:
        key = (r["model"], r["example_id"])
        if key in ids or r["checkpoint_sha256"] != checkpoint_hashes[r["model"]]:
            raise ValueError("Duplicate/mismatched baseline record")
        ids[key] = r
    return ids


def run(phase, checkpoints, output_dir, baseline_cases_file=None, freeze=None,
        holdout_file=None):
    if output_dir.exists():
        raise FileExistsError("Results must be immutable; choose an unused output directory")
    if set(checkpoints) != set(BASELINE_NAMES):
        raise ValueError("Exactly V5/V5B/V5D checkpoints required")
    identity = fingerprints(checkpoints)
    if phase == "dev":
        if baseline_cases_file is None or freeze is not None or holdout_file is not None:
            raise ValueError("DEV requires --baseline-cases only")
        if sha(PATHS["dev"]) != DEV_HASH:
            raise ValueError("DEV dataset has changed")
        rows = read_split("dev")
        baseline_cases = load_dev_baselines(baseline_cases_file,
                                            identity["checkpoints_sha256"])
        baseline_file_sha = sha(baseline_cases_file)
        source_hash = sha(PATHS["dev"])
    elif phase == "holdout":
        if freeze is None or holdout_file is None or baseline_cases_file is not None:
            raise ValueError("HOLDOUT requires --freeze and --holdout-file only")
        locked = json.loads(freeze.read_text(encoding="utf-8"))
        if locked.get("identity") != identity or locked.get("phase") != "dev" or (
            locked.get("dev_sha256") != DEV_HASH):
            raise ValueError("Decoder/checkpoint identity changed since DEV freeze")
        if holdout_file.resolve() in (PATHS["dev"].resolve(), PATHS["train"].resolve()):
            raise ValueError("Cannot reuse V5D training or development records as holdout")
        rows = [json.loads(line) for line in holdout_file.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 108 or len({r["example_id"] for r in rows}) != 108:
            raise ValueError("Holdout must have 108 uniquely identified rows")
        if any(not r["source_id"].startswith("v5e_holdout_") for r in rows):
            raise ValueError("Unsupported holdout identities")
        baseline_cases = None
        baseline_file_sha = locked.get("baseline_cases_sha256")
        from experiments.field_boundary_decoder_v5e.generate_holdout import build_holdout
        canonical = "".join(json.dumps(r, sort_keys=True) + "\\n"
                            for r in build_holdout()).encode("utf-8")
        if holdout_file.read_bytes() != canonical:
            raise ValueError("Holdout differs from pre-registered fixed-seed design")
        source_hash = sha(holdout_file)
        if source_hash == DEV_HASH:
            raise ValueError("Holdout cannot match DEV")
    else:
        raise ValueError("Unknown evaluation phase")
    output = []
    for name in BASELINE_NAMES:
        model, info = load_span_checkpoint(checkpoints[name])
        if info.get("answerability_threshold") != 0.5:
            raise ValueError("Unexpected model answerability threshold")
        if name == "v5d" and (
            info.get("experiment") != "v5d_balanced_boundaries" or
            info.get("extra_training_sha256") != sha(PATHS["train"])
        ):
            raise ValueError("V5D checkpoint provenance mismatch")
        for row in rows:
            output.append(compare_one(model, name, row, baseline_cases))
    overall, groups, report = summarize(output, phase)
    manifest = {
        "phase": phase, "identity": identity,
        "dev_sha256": DEV_HASH,
        "baseline_cases_sha256": baseline_file_sha,
        "evaluation_sha256": source_hash,
        "evaluation_count": len(rows),
        "summary": overall,
        "per_group": groups,
        "holdout_not_used": phase == "dev",
    }
    # Create files only after every inference and verification succeeds.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "all_pairs.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    if phase == "dev":
        (output_dir / "freeze.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "holdout"), required=True)
    for name in BASELINE_NAMES:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cases", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--holdout-file", type=Path)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        run(args.phase, {n: getattr(args, n) for n in BASELINE_NAMES},
            args.output_dir, args.baseline_cases, args.freeze, args.holdout_file)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Decoder comparison stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
