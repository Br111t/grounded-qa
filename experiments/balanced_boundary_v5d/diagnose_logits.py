"""Read-only examination of start/end logits and joint span choices.

Gold labels are accessed ONLY after model inference. For each example, this
reports the top candidate boundary positions, gold-boundary score ranks, and
highest scoring valid spans under the exact V5 decoder scoring rule.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from experiments.balanced_boundary_v5d.generate_data_v5d import PATHS, read_split
from experiments.balanced_boundary_v5d.evaluate_v5d import grouped
from experiments.balanced_boundary_v5d.reconcile_logits import enrich, measure
from grounded_qa.models.span_model_v5 import (
    load_span_checkpoint, predict_span, prepare_span_input, training_span_target,
)


def ranked_candidates(logits, allowed, k=5):
    """Highest scores first, then position for deterministic ties."""
    return [
        {"position": p, "score": round(float(logits[p]), 6)}
        for p in sorted(allowed, key=lambda p: (-float(logits[p]), p))[:k]
    ]


def boundary_rank(logits, allowed, gold):
    """1-based rank with deterministic lowest-position tie break."""
    order = sorted(allowed, key=lambda p: (-float(logits[p]), p))
    return order.index(gold) + 1


def top_joint_spans(starts, ends, allowed, max_span_bytes, k=5):
    """Mirror V5's valid-span search: inclusive end, maximum byte length."""
    first, last = allowed[0], allowed[-1]
    spans = []
    for start in range(first, last + 1):
        limit = min(last, start + max_span_bytes - 1)
        for end in range(start, limit + 1):
            spans.append((float(starts[start]) + float(ends[end]), start, end))
    # Python stable order ensures equal-scored pairs prefer lower start/end.
    spans.sort(key=lambda item: -item[0])
    return [
        {"start": s, "end": e, "score": round(score, 6)}
        for score, s, e in spans[:k]
    ]


def examine(model, row, model_name, checkpoint_hash):
    # Perform all model predictions without exposing references to the model.
    prediction, probability, selected = predict_span(
        model, row["information"], row["question"], threshold=0.5
    )
    tokens, mask = prepare_span_input(row["information"], row["question"])
    with torch.no_grad():
        start_logits, end_logits, _ = model(tokens, mask)
    starts = start_logits[0].detach().cpu().tolist()
    ends = end_logits[0].detach().cpu().tolist()
    allowed = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    top_starts = ranked_candidates(starts, allowed)
    top_ends = ranked_candidates(ends, allowed)
    top_spans = top_joint_spans(starts, ends, allowed, model.config.max_span_bytes)

    # Reference labels used strictly for scoring, after inference.
    answerable, gold_start, gold_end = training_span_target(row, tokens, mask)
    if not answerable:
        raise ValueError("Unexpected unanswerable V5D development record")
    raw_start, raw_end = top_starts[0]["position"], top_ends[0]["position"]
    selected_start, selected_end = selected if selected is not None else (None, None)
    correct = prediction == row["answer"]
    return {
        "model": model_name,
        "checkpoint_sha256": checkpoint_hash,
        "example_id": row["example_id"],
        "expected": row["answer"],
        "predicted": prediction,
        "group": ("length" + str(gold_end - gold_start + 1)
                  + ("_vowel" if row["answer"][-2].lower() in "aeiou"
                     else "_consonant")),
        "answerability_probability": round(probability, 6),
        "raw_start": raw_start,
        "raw_end": raw_end,
        "selected_start": selected_start,
        "selected_end": selected_end,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "gold_start_rank": boundary_rank(starts, allowed, gold_start),
        "gold_end_rank": boundary_rank(ends, allowed, gold_end),
        "gold_start_score": round(float(starts[gold_start]), 6),
        "gold_end_score": round(float(ends[gold_end]), 6),
        "best_start_score": top_starts[0]["score"],
        "best_end_score": top_ends[0]["score"],
        "top_starts": top_starts,
        "top_ends": top_ends,
        "top_joint_spans": top_spans,
        "exact": correct,
        "raw_end_correct": raw_end == gold_end,
        "selected_span_correct": selected == (gold_start, gold_end),
        "error_category": (
            "exact" if correct
            else "abstention" if selected is None
            else "decoder_disagrees_correct_raw_end" if raw_end == gold_end
            else "decoder_recovers_wrong_raw_end"
            if selected == (gold_start, gold_end)
            else "raw_and_selected_wrong"
        ),
    }


def choose_cases(cases, per_category=3):
    """Choose deterministic representative cases, not a cherry-picked success list."""
    categories = (
        "decoder_recovers_wrong_raw_end",
        "decoder_disagrees_correct_raw_end",
        "raw_and_selected_wrong",
        "abstention",
        "exact",
    )
    selected = []
    for group in ("length4_vowel", "length4_consonant",
                  "length5_vowel", "length5_consonant"):
        for category in categories:
            matching = sorted(
                (r for r in cases if r["group"] == group
                 and r["error_category"] == category),
                key=lambda r: r["example_id"],
            )
            selected.extend(matching[:per_category])
    return selected


def run(checkpoints, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Report directory already exists: {output_dir}")
    rows = read_split("dev")
    data_hash = hashlib.sha256(PATHS["dev"].read_bytes()).hexdigest()
    all_cases = []
    model_hashes = {}
    for name, path in checkpoints.items():
        model, info = load_span_checkpoint(path)
        if info.get("answerability_threshold") != 0.5:
            raise ValueError(f"{name}: unexpected answerability threshold")
        if name == "v5d":
            expected_hash = hashlib.sha256(PATHS["train"].read_bytes()).hexdigest()
            if (info.get("experiment") != "v5d_balanced_boundaries"
                    or info.get("extra_training_sha256") != expected_hash):
                raise ValueError("V5D training provenance mismatch")
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        model_hashes[name] = checkpoint_hash
        for row in rows:
            all_cases.append(examine(model, row, name, checkpoint_hash))
    all_cases = [enrich(r) for r in all_cases]
    counts = {name: measure([r for r in all_cases if r["model"] == name])
              for name in checkpoints}
    if len(all_cases) != 96 * len(checkpoints) or any(
            a["n"] != 96 or a["exact"] != a["selected_span_correct"]
            for a in counts.values()):
        raise ValueError("Incorrect record count or correctness inconsistency")
    sample = []
    for name in checkpoints:
        model_cases = [r for r in all_cases if r["model"] == name]
        sample += choose_cases(model_cases)
        sample += sorted((r for r in model_cases if r["recovered_wrong_raw_end"]),
                         key=lambda r: r["example_id"])[:5]
    sample = list({(r["model"], r["example_id"]): r for r in sample}.values())
    lines = [
        "# V5D logit and decoder diagnostic",
        "",
        f"V5D DEV SHA256: `{data_hash}`",
        "",
        "## Observations",
        "",
        "| Model | Gold end ranked #1 | Gold end ranked top 5 | Raw wrong, joint correct | Raw correct, joint wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in checkpoints:
        cases = [r for r in all_cases if r["model"] == name]
        lines.append(
            f"| {name.upper()} | "
            f"{sum(r['gold_end_rank'] == 1 for r in cases)}/96 | "
            f"{sum(r['gold_end_rank'] <= 5 for r in cases)}/96 | "
            f"{counts[name]['recovery']} | "
            f"{counts[name]['lost_correct_end']} |"
        )
    lines += [
        "", "## Score margins (best minus gold; logits, not probabilities)", "",
        "| Model | End gap <= 1 | End gap <= 3 | Joint gap <= 1 | Joint gap <= 3 | Median end gap | Median joint gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in checkpoints:
        a = counts[name]
        lines.append(
            f"| {name.upper()} | {a['end_gap_le1']}/{a['n']} | "
            f"{a['end_gap_le3']}/{a['n']} | {a['joint_gap_le1']}/{a['n']} | "
            f"{a['joint_gap_le3']}/{a['n']} | {a['median_end_gap']} | "
            f"{a['median_joint_gap']} |"
        )
    lines += [
        "", "## Representative predictions (deterministic selection)", "",
        "Chosen by group and error category, sorted by example ID; these are "
        "not random samples. See `all_cases.json` for all 288 observations.",
        "",
        "| Model | Group | Exact | Recovery | Expected | Predicted | Gold end rank | End gap | Joint gap |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in sample:
        lines.append(
            f"| {r['model'].upper()} | {r['group']} | {r['exact']} | "
            f"{r['recovered_wrong_raw_end']} | {r['expected']} | "
            f"{r['predicted']} | {r['gold_end_rank']} | "
            f"{r['end_score_margin']:.3f} | {r['joint_score_margin']:.3f} |"
        )
    lines += [
        "", "## Interpretation (hypotheses, not conclusions)",
        "",
        "- If the correct end often ranks #2–5, investigate scoring calibration "
        "and joint decoding before changing the encoder.",
        "- If the correct end consistently ranks far below the top candidates, "
        "investigate representation learning or the end-supervision objective.",
        "- If the raw end ranks #1 but the final span is wrong, inspect start "
        "scores and joint-score competition; do not blame the end head alone.",
        "- Compare the same examples across checkpoints; no causal claims "
        "about gradient conflict or data balance follow from this diagnostic.",
        "",
        "No training, checkpoint updates, threshold changes, or final holdout access.",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "all_cases.json").write_text(
        json.dumps(all_cases, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps({
        "experiment": "v5d_logit_diagnostic",
        "dataset_sha256": data_hash,
        "checkpoint_sha256": model_hashes,
        "example_count": len(all_cases),
        "counts": counts,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {len(all_cases)} prediction analyses to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v5", "v5b", "v5d"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        run({name: getattr(args, name) for name in ("v5", "v5b", "v5d")},
            args.output_dir)
        return 0
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Logit diagnostic stopped: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
