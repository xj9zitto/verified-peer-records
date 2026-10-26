"""scan_c3_labels.py [dir] -- list every bare \\Cth (C3 without its
recovery policy) in the manuscript sections, so each reported number can
be checked to name \\Cthd or \\Ctho (Reviewer 2, round 2, comment 1)."""
import re
import sys
from pathlib import Path

BARE = re.compile(r"\\Cth(?![dho])")
d = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_journal_r1")
for f in sorted((d / "sections").glob("*.tex")):
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if BARE.search(line):
            print("%-36s %4d  %s" % (f.name, i, line.strip()[:120]))
