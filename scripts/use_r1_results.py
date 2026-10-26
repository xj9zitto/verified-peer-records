"""use_r1_results.py -- unpack the released R1 results (results/r1/, some
gzip-compressed) into results/, where every analysis script and
scripts/verify_paper_numbers_r2.py expect them.

Usage: python scripts/use_r1_results.py
"""
import gzip
import shutil
from pathlib import Path

SRC, DST = Path("results/r1"), Path("results")
n = 0
for p in SRC.rglob("*"):
    if p.is_dir():
        continue
    rel = p.relative_to(SRC)
    if p.suffix == ".gz":
        out = DST / rel.with_suffix("")
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    n += 1
print("unpacked %d files into %s" % (n, DST))
