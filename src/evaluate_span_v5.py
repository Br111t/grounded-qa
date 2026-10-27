"""Evaluate V5 span extractor on exactly the unchanged V3 partitions."""

import argparse
from pathlib import Path

import torch

from evaluate_evidence_v3 import GROUPS, read_partitions
from generate_evidence_data_v3 import OUTPUTS
from span_model_v5 import load_span_checkpoint, predict_span


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS + ("all",), default="all")
    parser.add_argument("--show-errors", type=int, default=5)
    args = parser.parse_args()
    if args.show_errors < 0:
        parser.error("--show-errors must be nonnegative.")
    try:
        torch.set_num_threads(1)
        model, info = load_span_checkpoint(args.checkpoint)
        if Path(info.get("training_file", "")).resolve() != OUTPUTS["train"].resolve():
            raise ValueError("V5 checkpoint was not trained on the registered V3 training split.")
        threshold = info.get("answerability_threshold")
        if threshold != 0.5:
            raise ValueError("Unexpected answerability threshold; expected predefined 0.5.")
        partitions = read_partitions()
        selected = GROUPS if args.group == "all" else (args.group,)
        print(f"Checkpoint: {args.checkpoint}; architecture: V5 span extractor")
        print("Training group = fitting accuracy; other groups = development evaluation.")
        print("Answerability threshold: 0.5 (fixed in advance).")
        dev_correct, dev_total = 0, 0
        for group in selected:
            rows = partitions[group]
            correct = 0
            failures = []
            false_abstentions = 0
            false_answers = 0
            for row in rows:
                generated, confidence, indices = predict_span(
                    model, row["information"], row["question"],
                    threshold=threshold,
                )
                matched = generated == row["answer"]
                correct += int(matched)
                if row["answerability"] == "answerable" and generated == "Insufficient information.":
                    false_abstentions += 1
                if row["answerability"] == "unanswerable" and generated != "Insufficient information.":
                    false_answers += 1
                if not matched:
                    failures.append(
                        (row["example_id"], generated, row["answer"], confidence, indices)
                    )
            print(f"\n{group}: {correct}/{len(rows)} exact matches ({correct/len(rows):.1%})")
            if group in ("train", "known", "recombined", "unseen"):
                print(f"  False abstentions on answerable rows: {false_abstentions}")
            if group == "missing":
                print(f"  Unsupported answers on missing rows: {false_answers}")
            for example_id, generated, expected, probability, indices in failures[:args.show_errors]:
                print(
                    f"  {example_id}: generated={generated!r}; expected={expected!r}; "
                    f"P(answerable)={probability:.3f}; span={indices}"
                )
            if len(failures) > args.show_errors:
                print(f"  ... {len(failures)-args.show_errors} other mismatches omitted")
            if group != "train":
                dev_correct += correct
                dev_total += len(rows)
        if len(selected) > 1:
            print(f"\nOverall development: {dev_correct}/{dev_total} ({dev_correct/dev_total:.1%})")
        print("No weights or datasets changed; original final holdout not read.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5 evaluation stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
