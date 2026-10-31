"""Independent structural and factual checks for evidence-sensitivity V2 rows.

This parser does not use the generator's answer-building function.
"""

from collections.abc import Sequence

QUESTION = "What is the leader of Workshop W0001?"
PREFIX = "Fictional Workshop W0001. "
FALLBACK = "Insufficient information."
GROUPS = ("train", "known", "unseen", "missing")


def validate_v2_rows(rows: Sequence[dict], *, group: str) -> None:
    if group not in GROUPS:
        raise ValueError(f"Invalid evidence-sensitivity group: {group!r}")
    if not rows:
        raise ValueError("Evidence-sensitivity dataset cannot be empty.")
    seen_information = set()
    for row in rows:
        source_id = row["source_id"]
        if (not source_id.startswith(f"evidence_v2_{group}_")
                or row["example_id"] != f"{source_id}_q01"):
            raise ValueError(f"Unexpected source identifier in {group}: {source_id}")
        if row["question"] != QUESTION:
            raise ValueError(f"Evidence-sensitivity question changed: {source_id}")
        information = row["information"]
        if not information.startswith(PREFIX) or not information.endswith("."):
            raise ValueError(f"Unexpected evidence text format: {source_id}")
        if information in seen_information:
            raise ValueError(f"Duplicated evidence text: {source_id}")
        seen_information.add(information)

        # Parse evidence separately from the generator, to detect an incorrect
        # answer label or wrong leader rather than merely checking JSON validity.
        parts = information[len(PREFIX):-1].split(". ")
        fields = {}
        for part in parts:
            key, separator, value = part.partition(": ")
            if (not separator or key not in ("Leader", "Location", "Topic")
                    or not value or key in fields):
                raise ValueError(f"Invalid or duplicated evidence field: {source_id}")
            fields[key] = value
        if "Location" not in fields:
            raise ValueError(f"Missing location control field: {source_id}")
        leader = fields.get("Leader")
        if leader is not None and (not leader.isalpha() or not leader.isascii()):
            raise ValueError(f"Unexpected leader format: {source_id}")
        if group == "missing" and leader is not None:
            raise ValueError(f"Unexpected leader in missing-evidence set: {source_id}")
        if group in ("known", "unseen") and leader is None:
            raise ValueError(f"Missing leader in {group} set: {source_id}")
        expected_answer = f"Leader: {leader}." if leader else FALLBACK
        expected_label = "answerable" if leader else "unanswerable"
        if row["answer"] != expected_answer or row["answerability"] != expected_label:
            raise ValueError(f"Answer does not match supplied leader evidence: {source_id}")
