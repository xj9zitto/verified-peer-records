"""
scan_tex_damage.py — find escape-sequence damage in LaTeX sources.

Python string edits on LaTeX can silently turn "\\\\" into "\\" (a table
row terminator becomes a control space) or "\\f", "\\t", "\\b", "\\r",
"\\a", "\\v" into control characters.  This reports, per file, control
characters other than newline and lines inside tabular-like contexts
that end in a single backslash.

Usage: python scripts/scan_tex_damage.py <dir> [<dir> ...]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

SINGLE_END = re.compile(r"(?<!\\)\\[ \t]*$")


def scan(path: Path):
    text = path.read_text(encoding="utf-8")
    ctrl = [(i, hex(ord(c))) for i, c in enumerate(text)
            if ord(c) < 32 and c not in "\n\r"]
    single = [n + 1 for n, line in enumerate(text.split("\n"))
              if "&" in line and SINGLE_END.search(line)
              and not line.rstrip().endswith("\\\\")]
    return ctrl, single


def main() -> int:
    bad = 0
    for d in sys.argv[1:] or ["paper_journal_r1"]:
        files = sorted(Path(d).glob("sections/*.tex")) + [Path(d) / "main.tex"]
        for f in files:
            if not f.exists():
                continue
            ctrl, single = scan(f)
            if ctrl or single:
                bad += 1
                print(f"{f}: control chars {len(ctrl)} {ctrl[:5]}; "
                      f"single-backslash row ends at lines {single}")
    print("RESULT:", "DAMAGE FOUND" if bad else "clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
