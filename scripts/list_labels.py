"""list_labels.py <main.aux> -- print section/table/figure/algorithm numbers
as compiled, for cross-checking pointers in the response letter."""
import re
import sys

aux = open(sys.argv[1], encoding="utf-8").read()
# IEEEtran writes subsection numbers as {\mbox {IV-D}}; unwrap them.
aux = re.sub(r"\{[\\]mbox\s*\{([^}]*)\}\}", r"{\1}", aux)
for m in re.finditer(r"[\\]newlabel\{((?:sec|tab|alg|fig|eq):[^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", aux):
    print(f"{m.group(1):<34} {m.group(2):<10} p.{m.group(3)}")
