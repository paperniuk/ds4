#!/usr/bin/env python3
"""Write a token id list for DS4_QWEN4_MTP_DRAFT_VOCAB.

The MTP draft only needs its argmax, so it can be scored over the output rows
that are likely to win: the old Qwen vocabulary (the ids below --keep-below,
mostly Latin script and code), every later token of the chosen scripts, and the
control tokens. Tokens outside the list can still be generated, they just are
never drafted, so pick the scripts you actually write in.

    python3 gguf-tools/qwen4_mtp_draft_vocab.py model.gguf --scripts cyrillic > draft-vocab.txt
"""
import argparse
import struct
import sys

SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCRIPTS = {
    "cyrillic": [(0x400, 0x530)],
    "greek": [(0x370, 0x400)],
    "hebrew": [(0x590, 0x600)],
    "arabic": [(0x600, 0x700)],
    "cjk": [(0x3000, 0xA000), (0xAC00, 0xD7B0), (0xF900, 0xFB00), (0xFF00, 0xFFF0)],
}


def read_tokens(path):
    with open(path, "rb") as fp:
        def unpack(fmt):
            return struct.unpack("<" + fmt, fp.read(struct.calcsize("<" + fmt)))

        def string():
            return fp.read(unpack("Q")[0])

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                item, count = unpack("IQ")
                if item == 8:
                    return [string() for _ in range(count)]
                fp.seek(SCALAR_BYTES[item] * count, 1)
                return None
            fp.seek(SCALAR_BYTES[kind], 1)
            return None

        magic, _version, _tensors, n_kv = unpack("4sIQQ")
        if magic != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        for _ in range(n_kv):
            key = string()
            val = value(unpack("I")[0])
            if key == b"tokenizer.ggml.tokens":
                return val
    sys.exit(f"{path}: no tokenizer.ggml.tokens")


def byte_decoder():
    """GPT-2 byte-level BPE: the printable stand-in of every byte."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = keep[:]
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--keep-below", type=int, default=98304, help="keep every id below this (default 98304)")
    ap.add_argument("--scripts", default="cyrillic", help="comma separated: " + ", ".join(SCRIPTS))
    args = ap.parse_args()
    ranges = [r for name in args.scripts.split(",") if name for r in SCRIPTS[name.strip()]]
    to_byte = byte_decoder()
    kept = 0
    for i, raw in enumerate(read_tokens(args.gguf)):
        token = raw.decode("utf-8", "replace")
        keep = i < args.keep_below
        if not keep and len(token) < 40 and token.startswith("<") and token.endswith(">"):
            keep = True                                   # control tokens
        if not keep:
            try:
                text = bytes(to_byte[ch] for ch in token).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                text = ""
            keep = any(lo <= ord(ch) < hi for ch in text for lo, hi in ranges)
        if keep:
            print(i)
            kept += 1
    print(f"{kept} token ids", file=sys.stderr)


if __name__ == "__main__":
    main()
