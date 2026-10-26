"""
strip_rev_marks.py — start a revision from a marked manuscript.

The round-1 resubmission marks every passage that answered the first
reviews with \\rev{...} (blue).  The round-2 revision must mark only
what changes in response to the second reviews, so the first-round
marks are removed while the macro itself is kept for reuse.

Copies <src>/{main.tex, references.bib, sections/, figures/} to <dst>
and replaces every \\rev{X} in the section files by X.  Matching is
brace-aware: escaped braces (\\{ \\}) and %-comments are skipped, and
nesting (\\rev{\\emph{...}}, \\caption{\\rev{...}}, \\subsection{\\rev{...}})
is handled.  main.tex is copied unchanged, so its \\rev definition stays.

Self-checks, per file: each removal deletes exactly 6 characters
("\\rev{" and its closing "}"), the brace balance is unchanged, and no
\\rev{ remains.

Usage:
    python scripts/strip_rev_marks.py --src paper_journal_resubmission \\
                                      --dst paper_journal_r1
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

REV = re.compile(r"\\rev\{")


def _closing_brace(text: str, open_idx: int) -> int:
    """Index of the brace closing the one at ``open_idx``."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2              # skip the escaped character (\{ \} \% ...)
            continue
        if ch == "%":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced brace opened at {open_idx}")


def _balance(text: str) -> int:
    t = re.sub(r"\\[{}]", "", text)
    t = re.sub(r"(?<!\\)%.*", "", t)
    return t.count("{") - t.count("}")


def _in_comment(text: str, idx: int) -> bool:
    line_start = text.rfind("\n", 0, idx) + 1
    return re.search(r"(?<!\\)%", text[line_start:idx]) is not None


def strip(text: str) -> tuple[str, int]:
    n_removed = 0
    pos = 0
    while True:
        m = REV.search(text, pos)
        if not m:
            break
        if _in_comment(text, m.start()):
            pos = m.end()
            continue
        open_idx = m.end() - 1
        close_idx = _closing_brace(text, open_idx)
        text = (text[:m.start()] + text[open_idx + 1:close_idx]
                + text[close_idx + 1:])
        n_removed += 1
        pos = m.start()          # re-scan: the body may hold another \rev
    return text, n_removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("paper_journal_resubmission"))
    ap.add_argument("--dst", type=Path, default=Path("paper_journal_r1"))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing destination section folder.")
    args = ap.parse_args()

    if (args.dst / "sections").exists() and not args.force:
        sys.exit(f"{args.dst}/sections exists; refusing to overwrite "
                 "(use --force).")
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.src / "main.tex", args.dst / "main.tex")
    shutil.copy2(args.src / "references.bib", args.dst / "references.bib")
    if (args.dst / "figures").exists():
        shutil.rmtree(args.dst / "figures")
    shutil.copytree(args.src / "figures", args.dst / "figures")
    (args.dst / "sections").mkdir(exist_ok=True)

    total = 0
    for f in sorted((args.src / "sections").glob("*.tex")):
        raw = f.read_text(encoding="utf-8")
        out, n = strip(raw)
        assert len(raw) - len(out) == 6 * n, f.name
        assert _balance(raw) == _balance(out), f.name
        assert not [m for m in REV.finditer(out)
                    if not _in_comment(out, m.start())], f.name
        (args.dst / "sections" / f.name).write_text(out, encoding="utf-8")
        print(f"  {f.name:<40} {n:>3} marks removed")
        total += n
    print(f"Removed {total} round-1 marks -> {args.dst}/sections")


if __name__ == "__main__":
    main()
