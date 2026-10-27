"""Report V5 and V5B exact matches AND independent span-boundary accuracy.

The gold source boundaries are derived from evaluation references for
diagnostics only, never provided to predict_span or the model.
"""

import argparse
from pathlib import Path

import torch

from evaluate_evidence_v3 import GROUPS as V3_GROUPS, read_partitions
from evidence_v3_validation import parse_fields
from generate_evidence_data_v3 import OUTPUTS
from generate_span_data_v5b import EXTRA_PATH, read_additions
from span_model_v5 import (
    FALLBACK, load_span_checkpoint, predict_span,
    prepare_span_input, training_span_target,
)

NEW_GROUPS = ("familiar5", "novel4")
GROUPS = V3_GROUPS + NEW_GROUPS
TRAIN_GROUPS = {"train", "extra_train"}


def evaluate_group(model, rows: list[dict], show_errors: int = 5) -> dict:
    """Read-only evaluation. Predictions never receive reference answers."""
    totals = {
        "correct": 0, "total": len(rows), "answerable": 0,
        "start_head": 0, "end_head": 0, "joint_start": 0, "joint_end": 0,
        "joint_span": 0, "overlap": 0, "too_long": 0, "too_short": 0,
        "false_abstain": 0, "false_answer": 0,
    }
    mismatches = []
    for row in rows:
        # First predict WITHOUT passing row["answer"] or its gold offsets.
        prediction, confidence, predicted_span = predict_span(
            model, row["information"], row["question"], threshold=0.5
        )
        tokens, mask = prepare_span_input(row["information"], row["question"])
        # Ground truth is used strictly for diagnostic metrics, after prediction.
        answerable, gold_start, gold_end = training_span_target(row, tokens, mask)
        totals["correct"] += int(prediction == row["answer"])
        if not answerable:
            totals["false_answer"] += int(prediction != FALLBACK)
            if prediction != row["answer"] and len(mismatches) < show_errors:
                mismatches.append((row["example_id"], prediction, row["answer"], predicted_span))
            continue

        totals["answerable"] += 1
        if prediction == FALLBACK:
            totals["false_abstain"] += 1

        # Independent head accuracy, even when the final answer abstained.
        with torch.no_grad():
            start_logits, end_logits, _ = model(tokens, mask)
        totals["start_head"] += int(int(start_logits[0].argmax()) == gold_start)
        totals["end_head"] += int(int(end_logits[0].argmax()) == gold_end)
        if predicted_span is not None:
            start, end = predicted_span
            totals["joint_start"] += int(start == gold_start)
            totals["joint_end"] += int(end == gold_end)
            totals["joint_span"] += int(predicted_span == (gold_start, gold_end))
            totals["overlap"] += int(start <= gold_end and end >= gold_start)
            totals["too_long"] += int(
                start <= gold_start and end >= gold_end
                and predicted_span != (gold_start, gold_end)
            )
            totals["too_short"] += int(
                start >= gold_start and end <= gold_end
                and predicted_span != (gold_start, gold_end)
            )
        if prediction != row["answer"] and len(mismatches) < show_errors:
            mismatches.append((row["example_id"], prediction, row["answer"], predicted_span))

    totals["mismatches"] = mismatches
    return totals


def evaluate_by_name_length(model, rows: list[dict]) -> dict[int, dict]:
    """Group answerable rows by leader-name length for read-only diagnostics.

    Each subgroup uses the same evaluator and metrics; names/lengths are
    calculated from reference labels only after predictions are made by
    evaluate_group. No gradients or checkpoints are updated.
    """
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["answerability"] != "answerable":
            continue
        leader = parse_fields(row["information"], row["source_id"]).get("Leader")
        if leader is None or row["answer"] != f"Leader: {leader}.":
            raise ValueError("Cannot group an unsupported answerable reference.")
        grouped.setdefault(len(leader.encode("utf-8")), []).append(row)
    return {
        length: evaluate_group(model, subset, show_errors=0)
        for length, subset in sorted(grouped.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS + ("extra_train", "all"), default="all")
    parser.add_argument("--show-errors", type=int, default=5)
    parser.add_argument("--by-length", action="store_true",
                        help="also report exact/start/end accuracy separately by leader byte length")
    args = parser.parse_args()
    if args.show_errors < 0:
        parser.error("--show-errors must be nonnegative.")
    try:
        torch.set_num_threads(1)
        model, info = load_span_checkpoint(args.checkpoint)
        if Path(info.get("training_file", "")).resolve() != OUTPUTS["train"].resolve():
            raise ValueError("V5/V5B must use the registered V3 training split.")
        if info.get("answerability_threshold") != 0.5:
            raise ValueError("V5/V5B evaluation requires fixed threshold 0.5.")
        extra_file = info.get("extra_training_file")
        if extra_file is not None and Path(extra_file).resolve() != EXTRA_PATH.resolve():
            raise ValueError("V5B checkpoint references an unregistered extra training path.")

        partitions = read_partitions()  # includes V3 cross-partition leakage validation
        additions = {name: read_additions(name) for name in NEW_GROUPS}
        partitions.update(additions)
        if args.group == "extra_train":
            partitions["extra_train"] = read_additions("extra_train")
        # Existing evaluation files and new dev files may not reuse evidence
        # or source IDs; all new leader names must be disjoint from V3 names.
        seen_info, seen_ids, v3_names, extra_names = set(), set(), set(), set()
        for group, rows in partitions.items():
            for row in rows:
                if row["information"] in seen_info or row["source_id"] in seen_ids:
                    raise ValueError("Evaluation partition overlap detected.")
                seen_info.add(row["information"])
                seen_ids.add(row["source_id"])
                leader = parse_fields(row["information"], row["source_id"]).get("Leader")
                if leader:
                    (v3_names if group in V3_GROUPS else extra_names).add(leader)
        if v3_names & extra_names:
            raise ValueError("V5B development leader names overlap with V3 groups.")

        selected = GROUPS if args.group == "all" else (args.group,)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Experiment: {info.get('experiment', 'v5_original')}")
        print("Fixed threshold 0.5; original V3 plus V5B length/pattern development probes.")
        print("Heads evaluated independently; selected spans measured separately.")
        print("Gold spans are metric references only; model is not updated.")
        dev_correct = dev_total = 0
        for group in selected:
            totals = evaluate_group(model, partitions[group], args.show_errors)
            count = totals["total"]
            print(f"\n{group}: {totals['correct']}/{count} exact matches ({totals['correct']/count:.1%})")
            if totals["answerable"]:
                n = totals["answerable"]
                print(
                    f"  Independent head accuracy: start {totals['start_head']}/{n} "
                    f"({totals['start_head']/n:.1%}), end {totals['end_head']}/{n} "
                    f"({totals['end_head']/n:.1%})"
                )
                print(
                    f"  Selected span: start {totals['joint_start']}/{n}; "
                    f"end {totals['joint_end']}/{n}; both {totals['joint_span']}/{n}; "
                    f"overlap {totals['overlap']}/{n}"
                )
                print(
                    f"  Spans covering full gold with extra text: {totals['too_long']}; "
                    f"spans inside gold but incomplete: {totals['too_short']}; "
                    f"false abstentions: {totals['false_abstain']}"
                )
            else:
                print(f"  Unsupported answers on missing evidence: {totals['false_answer']}")
            if args.by_length and totals["answerable"]:
                breakdown = evaluate_by_name_length(model, partitions[group])
                for byte_length, scores in breakdown.items():
                    n = scores["answerable"]
                    print(
                        f"  Name length {byte_length} bytes: "
                        f"exact {scores['correct']}/{n} ({scores['correct']/n:.1%}); "
                        f"start head {scores['start_head']}/{n}; "
                        f"end head {scores['end_head']}/{n}; "
                        f"selected both {scores['joint_span']}/{n}; "
                        f"false abstentions {scores['false_abstain']}"
                    )
            for example_id, predicted, expected, span in totals["mismatches"]:
                print(f"  {example_id}: predicted={predicted!r}; expected={expected!r}; span={span}")
            if group not in TRAIN_GROUPS:
                dev_correct += totals["correct"]
                dev_total += count
        if len(selected) > 1 and dev_total:
            print(f"\nOverall development (V3 and V5B probes): "
                  f"{dev_correct}/{dev_total} ({dev_correct/dev_total:.1%})")
        print("Original V3 metrics and new probes should also be compared separately.")
        print("Read-only evaluation; no final holdout read.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5B evaluation stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
