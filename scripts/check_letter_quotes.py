"""
check_letter_quotes.py [paper_dir] -- every manuscript quotation in the
response letter (the newtext blocks) must appear verbatim in the
manuscript sections, so the letter cannot drift from the paper.

Quotes are split at elisions ([\\dots]) and at the bold "Section ..."
labels.  Each sentence longer than 40 characters is compared after
normalising whitespace, dropping citations, mapping cross-references
(\\ref{...} or a written-out number) to a placeholder, and removing
\\rev{}, \\emph{} wrappers.  A quoted sentence may stop early (its final
period may stand where the paper continues) and may capitalise the first
letter after an elision.
"""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_journal_r1")
letter = (d / "Response_to_Reviewers.tex").read_text(encoding="utf-8")
paper = "".join(p.read_text(encoding="utf-8")
                for p in sorted((d / "sections").glob("*.tex")))

BS = "\\\\"   # a literal backslash inside a regex


def norm(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = re.sub("~?" + BS + r"cite\{[^}]*\}", "", s)
    s = re.sub(r"(Table|Section|Fig\.|Algorithm)s?~?" + BS + r"ref\{[^}]*\}",
               lambda m: m.group(1) + " X", s)
    s = re.sub(r"(Table|Section|Fig\.|Algorithm)s?[ ~][IVX]+(-[A-G])?\b",
               lambda m: m.group(1) + " X", s)
    s = re.sub(BS + r"(rev|emph|textbf|texttt)\{", "", s)
    s = re.sub(BS + r"(medskip|noindent)", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(BS + r"[,;]", " ", s)
    s = s.replace("~", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


P = norm(paper)
blocks = re.findall(BS + r"begin\{newtext\}(.*?)" + BS + r"end\{newtext\}",
                    letter, re.S)
bad = n = 0
for b in blocks:
    b = re.sub(BS + r"textbf\{[^}]*\}", "|", b)          # drop labels
    for frag in re.split(r"\[" + BS + r"dots\]|\|", b):
        for sent in re.split(r"(?<=\.)\s+(?=[A-Z\\])", norm(frag)):
            sent = sent.strip()
            if len(sent) < 40:
                continue
            n += 1
            core = sent.rstrip(".").strip()
            if core in P or (core[0].lower() + core[1:]) in P:
                continue
            bad += 1
            print("NOT IN PAPER:", sent[:140])
print("%d quoted sentences checked, %d not found" % (n, bad))
sys.exit(1 if bad else 0)
