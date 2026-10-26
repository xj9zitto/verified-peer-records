"""
make_submission_pdfs.py

Renders the resubmission correspondence to PDF: the cover letter, the
statement of difference, and the point-by-point response to reviewers.

The response is rendered from Response_to_Reviewers.md so the PDF cannot
drift from the working document; the other two are held here because
their text is correspondence rather than a working artifact.

Usage:
    python scripts/make_submission_pdfs.py [--only cover|sod|response]
"""

from __future__ import annotations
import argparse
import html as _html
import re
import sys
from pathlib import Path

import fitz

OUT_DIR = Path("paper_journal_resubmission")
RESPONSE_MD = OUT_DIR / "Response_to_Reviewers.md"

PAPER_ID = "IoT-71260-2026"
NEW_TITLE = ("Content-Bound Provenance for Retrieved Evidence in "
             "Industrial LLM Safety Agents")
OLD_TITLE = ("PureChain-Verified Retrieval for Trustworthy Battery-Safety "
             "Decisions in Industrial IoT")

CSS = """
body { font-family: sans-serif; font-size: 10.5px; line-height: 1.45; }
h1 { font-size: 15px; margin-top: 2px; margin-bottom: 8px; }
h2 { font-size: 12.5px; margin-top: 13px; margin-bottom: 5px; }
h3 { font-size: 11px; margin-top: 11px; margin-bottom: 4px; }
h4 { font-size: 10.5px; margin-top: 9px; margin-bottom: 3px; }
p  { margin-top: 0px; margin-bottom: 7px; }
li { margin-bottom: 3px; }
blockquote { color: #444444; margin-left: 14px; margin-right: 10px;
             margin-bottom: 7px; }
table { margin-top: 4px; margin-bottom: 8px; }
th { font-size: 9.5px; text-align: left; padding-right: 9px;
     padding-bottom: 2px; }
td { font-size: 9.5px; padding-right: 9px; padding-bottom: 1px; }
.small { font-size: 9.5px; color: #444444; }
hr { margin-top: 8px; margin-bottom: 8px; }
"""


# ── minimal markdown -> html ──────────────────────────────────

def _inline(s: str) -> str:
    s = _html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+?)`", r"<i>\1</i>", s)
    return s


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        stripped = ln.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < n \
                and set(lines[i + 1].strip()) <= set("|-: "):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table>")
            out.append("<tr>" + "".join("<th>%s</th>" % _inline(h)
                                        for h in header) + "</tr>")
            for r in rows:
                out.append("<tr>" + "".join("<td>%s</td>" % _inline(c)
                                            for c in r) + "</tr>")
            out.append("</table>")
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            lvl = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (lvl, _inline(m.group(2)), lvl))
            i += 1
            continue

        if stripped.startswith("---"):
            out.append("<hr/>")
            i += 1
            continue

        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>%s</blockquote>" % _inline(" ".join(buf)))
            continue

        if re.match(r"^[-*]\s+", stripped):
            out.append("<ul>")
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                i += 1
                while i < n and lines[i].strip() and \
                        not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()) \
                        and not lines[i].strip().startswith(("#", "|", ">")):
                    item += " " + lines[i].strip()
                    i += 1
                out.append("<li>%s</li>" % _inline(item))
            out.append("</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            out.append("<ol>")
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                i += 1
                while i < n and lines[i].strip() and \
                        not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()) \
                        and not lines[i].strip().startswith(("#", "|", ">")):
                    item += " " + lines[i].strip()
                    i += 1
                out.append("<li>%s</li>" % _inline(item))
            out.append("</ol>")
            continue

        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and \
                not re.match(r"^([-*#>|]|\d+\.)\s*", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>%s</p>" % _inline(" ".join(buf)))

    return "\n".join(out)


# ── rendering ─────────────────────────────────────────────────

def render(html: str, path: Path, margin: int = 62) -> None:
    story = fitz.Story(html=html, user_css=CSS)
    writer = fitz.DocumentWriter(str(path))
    page = fitz.paper_rect("a4")
    area = page + (margin, margin, -margin, -margin)
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = fitz.open(str(path))
    print("  %-42s %d page(s)" % (path.name, doc.page_count))
    doc.close()


# ── documents ─────────────────────────────────────────────────

COVER = """
<h1>Cover Letter</h1>
<p><b>To:</b> Dr. Bo Li, Editor-in-Chief, IEEE Internet of Things Journal<br/>
Hong Kong University of Science and Technology<br/>
ioteict@cse.ust.hk</p>
<p><b>Handling Editor:</b> Prof. Cheng Li</p>
<p><b>Resubmission of manuscript {pid}</b></p>
<p>Dear Dr. Li,</p>
<p>Please find enclosed a resubmission of manuscript <b>{pid}</b>,
previously titled &ldquo;{old}&rdquo; and returned on 30 August 2026 with
the decision that it was not acceptable at that time and should be
resubmitted as a new submission. The revised manuscript is titled
&ldquo;{new}&rdquo;.</p>
<p>Both reviewers raised concerns that we judged to be well founded, and
two of them went to the substance of the paper.</p>
<p>Reviewer 2 observed that our handle-verified condition recovered the
falsified observable header, which a provenance mechanism cannot do, and
that its reported accuracy was therefore not comparable with the other
conditions. This was correct. We reimplemented that condition with a
recovery procedure it can actually deploy and re-ran the study. Removing
the oracle did not weaken the result. It revealed that the content-bound
predicate significantly outperforms handle checking, a difference the
oracle had been concealing.</p>
<p>The same reviewer asked whether our checker performed any independent
physical validation. On re-examining the implementation we found that
the checker described in the manuscript was not invoked in the battery
pipeline at all. We say so here instead of quietly repairing it, and the
checker is now implemented, unit-tested, integrated, and ablated.</p>
<p>The request for conventional baselines produced a third finding, which
we report in full. A gradient-boosted classifier predicts substantially
better than our agent on honest input, while collapsing just as
completely under falsified telemetry. With the reviewers' guidance on
scope, this led us to re-centre the title, abstract, contributions, and
conclusion on the integrity of retrieved evidence, which is what the
paper demonstrates, and away from end-to-end decision quality.</p>
<p>A detailed point-by-point response accompanies this letter. The
manuscript was rewritten and is submitted as a new manuscript that
stands on its own, and the paragraphs above and the enclosed response
set out what changed relative to {pid}. To spare the reviewers a
line-by-line comparison, the material we added or substantially revised
in answer to their comments is set in blue throughout the manuscript,
and a new table carries a blue caption. The work is original, is not
under consideration elsewhere, and all authors have approved the
submission. Its relationship to our accepted BCCA 2026 short paper is
set out in the enclosed Statement of Difference. That paper is now cited
and distinguished in the Related Work section, and our new evidence
revises its central claim instead of extending it.</p>
<p>Sincerely,<br/><br/>
Mohamed Abubakar Dini Rage<br/>
<span class="small">Department of IT Convergence Engineering<br/>
Kumoh National Institute of Technology, Gumi 39177, South Korea<br/>
mohamedabubakar407@gmail.com</span></p>
""".format(pid=PAPER_ID, old=OLD_TITLE, new=NEW_TITLE)


SOD = """
<h1>Statement of Difference from Prior Conference Work</h1>
<p class="small">IEEE Internet of Things Journal resubmission of {pid}:
&ldquo;{new}&rdquo;</p>
<h2>Prior Conference Paper</h2>
<p><b>Authors:</b> Mohamed Abubakar Dini Rage, Muhammad Sannan Khaliq,
Md Mehedi Hasan Somrat, Subroto Kumar Ghosh, Dong-Seong Kim, Jae Min Lee<br/>
<b>Title:</b> PureChain-Verified RAG: An Online Integrity Layer for
Industrial LLM Decision Agents<br/>
<b>Venue:</b> IEEE 8th International Conference on Blockchain Computing
and Applications (BCCA 2026), 16&ndash;20 Nov. 2026, Barcelona, Spain<br/>
<b>Paper ID:</b> 2688413990<br/>
<b>Status:</b> Accepted as a Short Paper, 3 July 2026.</p>
<h2>Relationship to the Present Submission</h2>
<p>The present submission is a substantially extended and independently
written study that shares one underlying synthetic corpus with the BCCA
2026 short paper. The conference paper is explicitly preliminary: it
evaluates a single existence-only provenance predicate at one corruption
rate on one four-action logistics task with one model backbone, and its
own conclusion names content-binding verification, valid-provenance
attacks, outcome-bearing peer evidence, and a multi-rate corruption
sweep as the agenda for the journal version. The present submission
delivers all four, in a different primary application domain that does
not appear in the conference paper. The conference corpus is retained
only as one secondary cross-domain ablation. Approximate share of new
content: more than 90 percent.</p>
<h2>What Is New or Substantially Extended</h2>
<p><b>1. New primary application domain and benchmark.</b> The
conference paper evaluates a four-action logistics task over a
120-scenario corpus. The present submission's primary contribution is a
battery-safety decision agent evaluated on a 235-scenario stratified
subset of a synthetic Li-ion thermal-runaway benchmark. The
outcome-bearing peer-record representation and all battery-safety
results are absent from the conference paper.</p>
<p><b>2. Two additional decision conditions.</b> The conference paper
evaluates three conditions; the present submission adds an
ungrounded-LLM baseline and a content-bound verified agent, evaluating
five.</p>
<p><b>3. Content-bound verification with genuine on-chain recovery.</b>
The conference predicate checks only that a handle resolves to a
committed event, and recovers by substituting records from a canonical
corpus. The present submission implements the re-fetch its conclusion
anticipated: a content-bound predicate compares a digest recomputed from
the retrieved record against the digest committed on-chain, 337 unique
peer records are anchored on the live PureChain mainnet, and rejected
records are reconstructed from the committed payloads, achieving full
detection and reconstruction of content-tampered records.</p>
<p><b>4. A revised, not merely extended, central claim.</b> This is the
most substantive difference. Under a threat model that includes
valid-provenance substitution, and once the existence-only predicate is
given a recovery procedure it can deploy, that predicate confers no
measurable protection: replicated over 20 independent attack seeds its
advantage over no verification is 0.000, with a 95 percent interval of
[0.000, 0.000]. The benefit reported in the conference paper is
therefore specific to fabricated handles. The journal manuscript states
this explicitly in Related Work.</p>
<p><b>5. Attribution of the effect to the mechanism.</b> Holding the
attack and verification code fixed while replacing the language model
with a deterministic decision rule reproduces the benefit of content
binding, establishing that it is a property of the verification layer
rather than of the specific LLM and retrieval design. No comparable
analysis appears in the conference paper.</p>
<p><b>6. Formal security treatment and systems evaluation.</b> The
present submission adds a security statement with explicit
cryptographic, consensus, and endpoint assumptions; thin-client endpoint
checks validated against the live chain; a quantitative comparison of
the ledger anchor against digital signatures and tamper-evident logs on
trust assumptions, storage, cost, and latency; and a discussion of the
privacy cost of committing payloads. The conference paper reports a
single deployment latency measurement.</p>
<p><b>7. Ablations and safety-oriented diagnostics.</b> A physics-gate
ablation, a state-only ablation, a shuffled-outcome control,
class-balanced diagnostics, urgent-event recall, under-triage rate, and
a cost-sensitive risk measure swept over a hundredfold range of
missed-runaway cost.
None appear in the conference paper.</p>
<h2>Citation of the Prior Conference Paper</h2>
<p>The conference paper is cited and distinguished in the Related Work
section of the present manuscript, identified as a preliminary
short-paper study of the handle-existence predicate on the shared legacy
corpus, together with the statement that the present evidence revises
its central claim.</p>
<h2>Text and Figure Reuse</h2>
<p>No text, figures, tables, or numerical results from the BCCA 2026
paper are reused verbatim. The two papers use different figures,
different notation for the verification predicate and recovery
procedure, and independently written prose throughout. The single point
of overlap is the underlying 120-scenario legacy corpus, re-evaluated
under a different attack protocol with different metrics and an added
baseline.</p>
<p>Signed on behalf of the authors,<br/><br/>
Mohamed Abubakar Dini Rage<br/>
<span class="small">Department of IT Convergence Engineering<br/>
Kumoh National Institute of Technology, South Korea<br/>
mohamedabubakar407@gmail.com</span></p>
""".format(pid=PAPER_ID, new=NEW_TITLE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cover", "sod", "response"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering into %s/" % OUT_DIR)

    if args.only in (None, "cover"):
        # slightly tighter margins so the signature block stays on page 1
        render(COVER, OUT_DIR / "Cover Letter.pdf", margin=45)
    if args.only in (None, "sod"):
        render(SOD, OUT_DIR / "Statement of Difference.pdf")
    if args.only in (None, "response"):
        if not RESPONSE_MD.exists():
            print("  missing %s" % RESPONSE_MD)
            sys.exit(1)
        body = md_to_html(RESPONSE_MD.read_text(encoding="utf-8"))
        render(body, OUT_DIR / "Response to Reviewers.pdf")


if __name__ == "__main__":
    main()
