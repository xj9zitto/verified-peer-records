"""
check_latex_refs.py

Static consistency check for the manuscript, standing in for a compiler
that is not installed on this machine.  Reports undefined \\ref and
\\cite targets, duplicate labels, unbalanced environments, and labels
that are never referenced.

Usage: python scripts/check_latex_refs.py [paper_dir]
"""

from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DIR = Path("paper_journal")

RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")
RE_EQREF = re.compile(r"\\eqref\{([^}]+)\}")
RE_CITE = re.compile(r"\\cite[tp]?\{([^}]+)\}")
RE_BIBKEY = re.compile(r"@\w+\{([^,]+),")
RE_BEGIN = re.compile(r"\\begin\{(\w+\*?)\}")
RE_END = re.compile(r"\\end\{(\w+\*?)\}")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    tex_files = sorted(root.rglob("*.tex"))
    if not tex_files:
        print("no .tex files under %s" % root)
        return 1

    labels: Counter = Counter()
    label_where: dict = {}
    refs: dict = {}
    cites: dict = {}
    env_problems = []

    for f in tex_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in RE_LABEL.finditer(text):
            labels[m.group(1)] += 1
            label_where.setdefault(m.group(1), f.name)
        for rx in (RE_REF, RE_EQREF):
            for m in rx.finditer(text):
                refs.setdefault(m.group(1), set()).add(f.name)
        for m in RE_CITE.finditer(text):
            for key in m.group(1).split(","):
                cites.setdefault(key.strip(), set()).add(f.name)

        begins = Counter(RE_BEGIN.findall(text))
        ends = Counter(RE_END.findall(text))
        for env in set(begins) | set(ends):
            if begins[env] != ends[env]:
                env_problems.append(
                    "%s: %s begin=%d end=%d"
                    % (f.name, env, begins[env], ends[env]))

    bib_keys = set()
    for b in sorted(root.rglob("*.bib")):
        bib_keys |= set(k.strip() for k in
                        RE_BIBKEY.findall(b.read_text(encoding="utf-8",
                                                      errors="replace")))

    ok = True

    dupes = [l for l, c in labels.items() if c > 1]
    if dupes:
        ok = False
        print("DUPLICATE LABELS:")
        for l in sorted(dupes):
            print("  %s (x%d)" % (l, labels[l]))

    undefined = sorted(r for r in refs if r not in labels)
    if undefined:
        ok = False
        print("UNDEFINED REFERENCES:")
        for r in undefined:
            print("  %-28s referenced in %s"
                  % (r, ", ".join(sorted(refs[r]))))

    miss_cite = sorted(c for c in cites if c not in bib_keys)
    if miss_cite:
        ok = False
        print("MISSING BIB ENTRIES:")
        for c in miss_cite:
            print("  %-28s cited in %s" % (c, ", ".join(sorted(cites[c]))))

    if env_problems:
        ok = False
        print("UNBALANCED ENVIRONMENTS:")
        for p in env_problems:
            print("  %s" % p)

    unused = sorted(l for l in labels if l not in refs)
    if unused:
        print("UNREFERENCED LABELS (not an error):")
        for l in unused:
            print("  %-28s in %s" % (l, label_where[l]))

    print("\n%d tex files, %d labels, %d distinct refs, %d citations, "
          "%d bib entries" % (len(tex_files), len(labels), len(refs),
                              len(cites), len(bib_keys)))
    print("RESULT: %s" % ("OK" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
