"""Independent structural and reference validation for evidence sensitivity V3.

Intentionally does not call the data generator or its answer function.
"""

from collections import Counter
from collections.abc import Sequence

QUESTION = "What is the leader of Workshop W0001?"
PREFIX = "Fictional Workshop W0001. "
FALLBACK = "Insufficient information."
EXPECTED_COUNTS = {"train": 168, "known": 48, "recombined": 24,
                   "unseen": 24, "missing": 12}
KNOWN_ROOMS = {"Room Maple", "Room Cedar", "Room Birch",
               "Room Pine", "Room Elm", "Room Ash"}
KNOWN_TOPICS = {"Python", "Gardening", "Astronomy"}


def parse_fields(information: str, source_id: str) -> dict[str, str]:
    if not information.startswith(PREFIX) or not information.endswith("."):
        raise ValueError(f"Unexpected evidence text structure: {source_id}")
    fields: dict[str, str] = {}
    for fragment in information[len(PREFIX):-1].split(". "):
        key, separator, value = fragment.partition(": ")
        if (not separator or key not in ("Leader", "Location", "Topic")
                or not value or key in fields):
            raise ValueError(f"Invalid or duplicate evidence field: {source_id}")
        fields[key] = value
    if fields.get("Location") not in KNOWN_ROOMS:
        raise ValueError(f"Invalid location control: {source_id}")
    if "Topic" in fields and fields["Topic"] not in KNOWN_TOPICS:
        raise ValueError(f"Invalid topic control: {source_id}")
    return fields


def validate_v3_rows(rows: Sequence[dict], group: str) -> None:
    if group not in EXPECTED_COUNTS:
        raise ValueError(f"Unexpected V3 dataset group: {group}")
    if len(rows) != EXPECTED_COUNTS[group]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[group]} {group} rows; got {len(rows)}."
        )
    seen_sources: set[str] = set()
    seen_information: set[str] = set()
    leaders: list[str] = []
    missing_count = 0

    for row in rows:
        source_id = row["source_id"]
        if not source_id.startswith(f"evidence_v3_{group}_"):
            raise ValueError(f"Source group mismatch: {source_id}")
        if row["example_id"] != f"{source_id}_q01":
            raise ValueError(f"Unexpected example ID: {source_id}")
        if source_id in seen_sources or row["information"] in seen_information:
            raise ValueError(f"Duplicated source or evidence: {source_id}")
        seen_sources.add(source_id)
        seen_information.add(row["information"])
        if row["question"] != QUESTION:
            raise ValueError(f"Evidence question is not fixed: {source_id}")

        fields = parse_fields(row["information"], source_id)
        leader = fields.get("Leader")
        if leader is not None and (
            not leader.isascii() or not leader.isalpha()
            or not leader[0].isupper() or not leader[1:].islower()
        ):
            raise ValueError(f"Invalid leader value: {source_id}")

        if leader is None:
            missing_count += 1
        else:
            leaders.append(leader)
        expected = f"Leader: {leader}." if leader is not None else FALLBACK
        label = "answerable" if leader is not None else "unanswerable"
        if row["answerability"] != label or row["answer"] != expected:
            raise ValueError(f"Incorrect label or expected answer: {source_id}")

    if group == "train":
        counts = Counter(leaders)
        if len(counts) != 48 or set(counts.values()) != {3} or missing_count != 24:
            raise ValueError("V3 training must contain 48 leaders × 3 and 24 missing cases.")
    elif group == "known":
        if len(set(leaders)) != 48 or missing_count:
            raise ValueError("Known evaluation must contain 48 distinct leader names.")
    elif group in ("recombined", "unseen"):
        counts = Counter(leaders)
        if len(counts) != 12 or set(counts.values()) != {2} or missing_count:
            raise ValueError(f"{group} evaluation must contain 12 leaders × 2.")
    elif group == "missing":
        if missing_count != 12:
            raise ValueError("All missing-evidence evaluation rows must omit the leader.")
