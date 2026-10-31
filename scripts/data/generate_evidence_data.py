
import json
from pathlib import Path

LEADERS = ["Mira", "Tavi", "Sana", "Ari", "Niko", "Zuri", None]

QUESTION = "What is the leader of Workshop W0001?"
LOCATION = "Room Maple"

examples = []

for index, leader in enumerate(LEADERS, start=1):
    information = "Fictional Workshop W0001. "

    if leader is not None:
        information += f"Leader: {leader}. "

    information += f"Location: {LOCATION}."

    answer = (
        f"Leader: {leader}."
        if leader is not None
        else "Insufficient information."
    )

    examples.append({
        "example_id": f"evidence_{index:03d}_q01",
        "source_id": f"evidence_{index:03d}",
        "information": information,
        "question": QUESTION,
        "answerability": (
            "answerable" if leader is not None else "unanswerable"
        ),
        "answer": answer,
    })

output = Path("data/source/evidence_sensitivity_v1.jsonl")

if output.exists():
    raise FileExistsError(f"Dataset already exists: {output}")

output.write_text(
    "".join(json.dumps(row) + "\n" for row in examples),
    encoding="utf-8",
)

print(f"Generated {len(examples)} examples: {output}")
