"""Convert text to byte token IDs without a learned vocabulary or model files."""

import argparse
from collections.abc import Iterable


PAD_ID = 0  # Padding will let us put different-length sequences in one batch.
BOS_ID = 1  # Beginning of a sequence.
EOS_ID = 2  # End of a sequence.
BYTE_OFFSET = 3
VOCAB_SIZE = BYTE_OFFSET + 256


def encode(text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    """Map each UTF-8 byte to an ID, optionally adding sequence boundaries.

    A byte has a value from 0 to 255. Adding 3 keeps text bytes separate from
    the reserved PAD, BOS, and EOS IDs. Some characters use more than one byte.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    token_ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
    if add_bos:
        token_ids.insert(0, BOS_ID)
    if add_eos:
        token_ids.append(EOS_ID)
    return token_ids


def decode(token_ids: Iterable[int], *, errors: str = "strict") -> str:
    """Recover text, skipping PAD/BOS and stopping at the first EOS.

    IDs after EOS are ignored. Invalid IDs before EOS raise ValueError.
    Strict decoding rejects invalid UTF-8. Use errors='replace' only when
    inspecting imperfect future model output; it displays replacement characters.
    """
    if errors not in ("strict", "replace"):
        raise ValueError("errors must be 'strict' or 'replace'.")

    text_bytes = bytearray()
    for token_id in token_ids:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < VOCAB_SIZE
        ):
            raise ValueError(f"Token ID must be an integer from 0 to {VOCAB_SIZE - 1}: {token_id!r}")
        if token_id == EOS_ID:
            break
        if token_id in (PAD_ID, BOS_ID):
            continue
        text_bytes.append(token_id - BYTE_OFFSET)
    return text_bytes.decode("utf-8", errors=errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Room Cedar.", help="text to encode and decode")
    args = parser.parse_args()

    token_ids = encode(args.text, add_bos=True, add_eos=True)
    restored = decode(token_ids)
    print(f"Vocabulary size: {VOCAB_SIZE} (256 bytes + PAD, BOS, EOS)")
    # ascii() keeps the demo readable even in terminals with limited encodings.
    print(f"Text: {ascii(args.text)}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {ascii(restored)}")
    print(f"Round trip matches: {restored == args.text}")
    print("This converts text to numbers and back; it does not answer questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
