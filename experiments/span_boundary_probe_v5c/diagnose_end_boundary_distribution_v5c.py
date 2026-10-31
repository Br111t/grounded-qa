"""Read-only V5C diagnostic: audit V3/V5B training end-boundary coverage.

Run from repository root:
    $env:PYTHONPATH = "$PWD\src"
    python -m experiments.span_boundary_probe_v5c.diagnose_end_boundary_distribution_v5c

Does not read final holdout, update datasets/checkpoints, or train a model.
"""

import sys
from collections import Counter
from pathlib import Path

from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields, validate_v3_rows
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import OUTPUTS as V3_OUTPUTS
from experiments.variable_span_v5b.generate_span_data_v5b import EXTRA_PATH, validate_additions
from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import GROUPS, read_probe_group
from grounded_qa.models.span_model_v5 import prepare_span_input, training_span_target
from grounded_qa.tokenization.tokenizer import encode
from scripts.data.validate_data import validate_data


def read_checked(path: Path) -> list[dict]:
    rows, errors = validate_data(path)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return rows


def analyze(rows: list[dict]) -> dict:
    lengths = Counter()
    endings = Counter()
    ending_types = Counter()
    offsets = Counter()
    absolute_ends = Counter()
    last_byte_positions = Counter()
    unique_names = set()
    answerable = 0
    unanswerable = 0

    for row in rows:
        tokens, mask = prepare_span_input(row["information"], row["question"])
        is_answerable, start, end = training_span_target(row, tokens, mask)
        name = parse_fields(row["information"], row["source_id"]).get("Leader")
        if not is_answerable:
            if name is not None or start is not None or end is not None:
                raise AssertionError(f"Unexpected missing-evidence target: {row['example_id']}")
            unanswerable += 1
            continue
        if name is None or start is None or end is None:
            raise AssertionError(f"Missing grounded span: {row['example_id']}")
        name_bytes = encode(name)
        assert tokens[0, start:end + 1].tolist() == name_bytes, row["example_id"]
        assert end - start + 1 == len(name_bytes), row["example_id"]
        answerable += 1
        unique_names.add(name)
        lengths[len(name_bytes)] += 1
        endings[name[-1].lower()] += 1
        ending_types["vowel" if name[-1].lower() in "aeiou" else "consonant"] += 1
        offsets[end - start] += 1
        absolute_ends[end] += 1
        last_byte_positions[(len(name_bytes), name[-1].lower())] += 1

    return dict(total=len(rows), answerable=answerable, unanswerable=unanswerable,
                names=unique_names, lengths=lengths, endings=endings,
                ending_types=ending_types, offsets=offsets,
                absolute_ends=absolute_ends, length_endings=last_byte_positions)


def display(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"Rows: {result['total']} | answerable: {result['answerable']} | "
          f"unanswerable: {result['unanswerable']} | unique names: {len(result['names'])}")
    for key in ("lengths", "endings", "ending_types", "offsets", "absolute_ends"):
        print(f"{key}: {dict(sorted(result[key].items()))}")
    print("length x ending character:")
    for (length, char), count in sorted(result["length_endings"].items()):
        print(f"  length={length} final={char!r}: {count}")


def main() -> int:
    v3 = read_checked(V3_OUTPUTS["train"])
    validate_v3_rows(v3, "train")
    extra = read_checked(EXTRA_PATH)
    validate_additions("extra_train", extra)

    v3_stats = analyze(v3)
    extra_stats = analyze(extra)
    combined_stats = analyze(v3 + extra)

    display("V5 training: V3 only", v3_stats)
    display("V5B additional training", extra_stats)
    display("V5B combined training: V3 + additions", combined_stats)

    print("\n=== V5C DEVELOPMENT probe coverage (NOT TRAINING) ===")
    for group in GROUPS:
        rows = read_probe_group(group)
        probe = analyze(rows)
        # Compare full names only; do not update training or probe data.
        overlap_v3 = sorted(probe["names"] & v3_stats["names"])
        overlap_v5b = sorted(probe["names"] & combined_stats["names"])
        print(f"{group}: {probe['answerable']} answerable; "
              f"V3 exact name overlap={len(overlap_v3)}; "
              f"V5B exact name overlap={len(overlap_v5b)}")
        for length, ending in sorted(probe["length_endings"]):
            count = probe["length_endings"][(length, ending)]
            v3_n = v3_stats["length_endings"][(length, ending)]
            v5b_n = combined_stats["length_endings"][(length, ending)]
            print(f"  length={length} final={ending!r}: probe={count}, "
                  f"V3 train={v3_n}, V5B train={v5b_n}")
        if overlap_v5b:
            raise AssertionError(f"V5C leakage into training: {group}: {overlap_v5b}")

    print("\nPASS: All audited end indices reconstruct the reference bytes; "
          "V5C names are disjoint from training. No files or weights changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
