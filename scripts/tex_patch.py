"""tex_patch.py <patchfile.py> -- apply exact-match edits to a LaTeX file.

The patch file defines FILE (path relative to the project root) and any
of: EDITS, a list of (old, new) pairs, each old occurring exactly once;
SPANS, a list of (start_anchor, end_anchor, new) replacing text from the
start anchor (inclusive) up to the end anchor (exclusive).  Writing
patches as Python raw strings in a file avoids shell escaping, which
silently drops backslashes and breaks on apostrophes.  The result is
rejected if it would introduce control characters (eaten escapes).
"""
import runpy
import sys
from pathlib import Path

ns = runpy.run_path(sys.argv[1])
p = Path(ns["FILE"])
s = p.read_text(encoding="utf-8")
for a, b, new in ns.get("SPANS", []):
    assert s.count(a) == 1, f"span start not unique: {a[:60]!r}"
    i = s.index(a)
    j = s.index(b, i)
    s = s[:i] + new + s[j:]
for old, new in ns.get("EDITS", []):
    n = s.count(old)
    assert n == 1, f"{n} matches for: {old[:70]!r}"
    s = s.replace(old, new)
bad = [c for c in s if ord(c) < 32 and c not in "\n\r\t"]
assert not bad, f"control characters introduced: {bad[:5]}"
p.write_text(s, encoding="utf-8")
print("patched", p)
