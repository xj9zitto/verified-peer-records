"""
compare_pdf_text.py — word-level diff between two PDFs.

Used to confirm that a source tree reproduces a submitted PDF: colour
and layout are ignored, only the word sequence is compared, and each
differing span is printed with a little context.

Usage: python scripts/compare_pdf_text.py a.pdf b.pdf [--max 40]
"""
from __future__ import annotations
import argparse
import difflib
import re

import fitz


def words(path: str) -> list[str]:
    text = "\n".join(p.get_text() for p in fitz.open(path))
    text = re.sub(r"-\n(?=[a-z])", "", text)        # rejoin hyphenation
    return re.findall(r"[A-Za-z0-9.,;:%()\[\]+\-=/]+", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max", type=int, default=40)
    args = ap.parse_args()
    wa, wb = words(args.a), words(args.b)
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    print(f"{len(wa)} vs {len(wb)} words; ratio {sm.ratio():.4f}; "
          f"{len(ops)} differing spans")
    for tag, i1, i2, j1, j2 in ops[: args.max]:
        ctx = " ".join(wa[max(0, i1 - 6):i1])
        print(f"- {tag:<7} ...{ctx} | A: {' '.join(wa[i1:i2])[:160]!r}"
              f" | B: {' '.join(wb[j1:j2])[:160]!r}")


if __name__ == "__main__":
    main()
