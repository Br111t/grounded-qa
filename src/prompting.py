"""Build the text input for a grounded question without adding its target answer."""


INSTRUCTIONS = """Answer the question using only the supplied information.
Do not use outside knowledge or guess.
If all requested parts are supported, answer all of them.
If only some parts are supported, answer those parts and identify each missing part with "Insufficient information."
If none of the requested parts are supported, respond with exactly "Insufficient information."
Keep the answer concise."""


PROMPT_VERSIONS = ("original", "compact_v1")


def build_prompt(
    information: str, question: str, *, version: str = "original"
) -> str:
    """Format the input shared by future training, prediction, and evaluation.

    Only information and question are accepted. The expected answer, answerability
    label, and record IDs are kept outside the model's input.
    """
    for name, value in (("information", information), ("question", question)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string.")

    if version == "compact_v1":
        return (
            f"Information:\n{information}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:\n"
        )
    if version != "original":
        raise ValueError(f"Unsupported prompt version: {version!r}")

    return (
        f"Instructions:\n{INSTRUCTIONS}\n\n"
        f"Information:\n{information}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )
