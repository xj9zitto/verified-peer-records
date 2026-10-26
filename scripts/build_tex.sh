#!/bin/bash
# build_tex.sh <dir> <main-basename>  -- pdflatex + bibtex + 2x pdflatex
# Prints errors, undefined refs/cites, overfull boxes and the page count.
set -u
D="${1:-paper_journal_r1}"; M="${2:-main}"
MT="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64"
cd "$D" || exit 1
"$MT/pdflatex.exe" -interaction=nonstopmode "$M.tex" > "$M.build.log" 2>&1
if grep -qF 'bibdata' "$M.aux" 2>/dev/null; then
  "$MT/bibtex.exe" "$M" > "$M.bibtex.log" 2>&1 || { echo "BIBTEX FAILED"; tail -5 "$M.bibtex.log"; }
fi
"$MT/pdflatex.exe" -interaction=nonstopmode "$M.tex" > "$M.build.log" 2>&1
"$MT/pdflatex.exe" -interaction=nonstopmode "$M.tex" > "$M.build.log" 2>&1
echo "errors:   $(grep -c '^! ' "$M.log")"
grep -A3 '^! ' "$M.log" | head -12
echo "undefined: $(grep -c -E 'Citation .* undefined|Reference .* undefined' "$M.log")"
grep -E 'Citation .* undefined|Reference .* undefined' "$M.log" | head -8
echo "overfull: $(grep -c 'Overfull \hbox' "$M.log")"
PYTHONUTF8=1 ../.venv/Scripts/python.exe -c "import fitz,sys;print(\"Pages:\", fitz.open(sys.argv[1]).page_count)" "$M.pdf"
