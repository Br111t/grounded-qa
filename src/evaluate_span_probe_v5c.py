"""Read-only matched 2x2 diagnostic of V5/V5B end-position generalization.

Four groups isolate (length 4 vs 5) x (vowel vs consonant ending), with
identical source contexts for each matched name stem. References are used for
scoring only. Does NOT train, adjust thresholds, or change model weights.
"""

import argparse
from pathlib import Path

import torch

from evaluate_span_v5b import evaluate_group
from generate_evidence_data_v3 import OUTPUTS as V3_PATHS
from generate_span_data_v5b import EXTRA_PATH
from generate_span_probe_v5c import GROUPS, read_probe_group
from span_model_v5 import load_span_checkpoint


def _check_checkpoint(info: dict):
    if Path(info.get("training_file", "")).resolve() != V3_PATHS["train"].resolve():
        raise ValueError("Checkpoint must be trained on the registered V3 training split.")
    if info.get("answerability_threshold") != 0.5:
        raise ValueError("Unexpected answerability threshold; expected fixed 0.5.")
    extra = info.get("extra_training_file")
    if extra is not None and Path(extra).resolve() != EXTRA_PATH.resolve():
        raise ValueError("Unrecognized V5B training addition.")


def evaluate_probe(model, partitions: dict[str, list[dict]], show_errors: int = 3):
    scores = {}
    for group in GROUPS:
        score = evaluate_group(model, partitions[group], show_errors=show_errors)
        scores[group] = score
        n = score["answerable"]
        print(
            f"  {group}: exact {score['correct']}/{n} ({score['correct']/n:.1%}), "
            f"start-head {score['start_head']}/{n}, end-head {score['end_head']}/{n}, "
            f"selected span {score['joint_span']}/{n}, "
            f"truncated {score['too_short']}, overextended {score['too_long']}, "
            f"false abstentions {score['false_abstain']}"
        )
        for id_, prediction, expected, span in score["mismatches"]:
            print(
                f"    {id_}: predicted={prediction!r}; "
                f"expected={expected!r}; span={span}"
            )

    for axis, parts in (
        ("Length 4", ("length4_vowel", "length4_consonant")),
        ("Length 5", ("length5_vowel", "length5_consonant")),
        ("Vowel end", ("length4_vowel", "length5_vowel")),
        ("Consonant end", ("length4_consonant", "length5_consonant")),
    ):
        matched = sum(scores[g]["correct"] for g in parts)
        ends = sum(scores[g]["end_head"] for g in parts)
        starts = sum(scores[g]["start_head"] for g in parts)
        n = sum(scores[g]["answerable"] for g in parts)
        print(
            f"  {axis} combined: exact {matched}/{n} ({matched/n:.1%}), "
            f"start-head {starts}/{n}, end-head {ends}/{n}"
        )
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--compare-checkpoint", type=Path,
        help="optional second checkpoint, evaluated independently on identical probe rows",
    )
    parser.add_argument("--show-errors", type=int, default=2)
    args = parser.parse_args()
    if args.show_errors < 0:
        parser.error("--show-errors must be nonnegative.")
    try:
        torch.set_num_threads(1)
        # Every diagnostic input is validated before either model runs.
        partitions = {group: read_probe_group(group) for group in GROUPS}
        if sum(len(v) for v in partitions.values()) != 96:
            raise ValueError("V5C diagnostic requires 96 locked rows.")
        checkpoints = [args.checkpoint]
        if args.compare_checkpoint is not None:
            if args.compare_checkpoint.resolve() == args.checkpoint.resolve():
                raise ValueError("Comparison checkpoint must be different.")
            checkpoints.append(args.compare_checkpoint)
        print(
            "READ-ONLY V5C matched test: 4/5-character names x vowel/consonant endings; "
            "24 examples per cell."
        )
        print("Pairs have matched contexts. Fixed threshold 0.5. Gold spans only score predictions.")
        results = {}
        for checkpoint in checkpoints:
            model, info = load_span_checkpoint(checkpoint)
            _check_checkpoint(info)
            print(f"\nCheckpoint: {checkpoint}; experiment: {info.get('experiment', 'v5_original')}")
            results[str(checkpoint)] = evaluate_probe(model, partitions, args.show_errors)
        if len(checkpoints) == 2:
            print("\nPaired accuracy comparison (same 24 rows in each group):")
            first, second = (results[str(path)] for path in checkpoints)
            for group in GROUPS:
                a = first[group]["correct"]
                b = second[group]["correct"]
                print(f"  {group}: first {a}/24; second {b}/24; delta {b-a:+d} examples")
        print("No training or weight changes. Original V3/V5B files and final holdout untouched.")
        print(
            "All 96 examples are additional development diagnostics, not final-holdout "
            "evidence. Avoid tuning against them indefinitely."
        )
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5C diagnostic stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
