#!/usr/bin/env python
"""Build a punctuation-restoration gold set from public-domain prose.

The CC0 fixture transcriptions gave 102 marks — about 1-2 standard errors of
separation between artifacts, which cannot settle anything. This produces a set
large enough to.

Sentences are taken from Project Gutenberg plain text (public domain), which
means the punctuation is a real editor's rather than mine. That matters: gold I
wrote myself would encode my own habits, and the models would be scored against
the preferences of whoever picked the sentences.

Selection is deliberately conservative, and every filter exists to remove a way
the score could be wrong rather than merely inconvenient:

  * Gutenberg front/back matter stripped by the START/END markers.
  * Verse, tables and headers dropped — lines that are short, ALL-CAPS, or
    heavily indented. A punctuation model scored on a chapter heading measures
    nothing.
  * Sentences with quotation marks, em-dashes, semicolons, parentheses or
    ellipses are REJECTED, not stripped. Those are exactly where a period vs a
    colon is a stylistic coin-flip, and the CC0 attempt showed that is most of
    what a small sample ends up measuring. What survives is prose whose internal
    commas and terminal stop are close to forced.
  * Length window 8-40 words, so a single sentence neither dominates nor
    contributes one mark.

    python tools/build_punct_gold.py <out.json> <text.txt> [<text.txt> ...]
"""
import json
import re
import sys

MARKS = ".,?"          # what these models can emit AND gold reliably contains
REJECT = set('"“”‘’«»()[]—–;…*_')


def strip_gutenberg(text):
    m = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG.*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    return text


def paragraphs(text):
    for para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in para.splitlines()]
        if not lines:
            continue
        # Verse/heading heuristics: indented blocks and short all-caps lines.
        if sum(1 for ln in lines if ln.startswith("    ")) > len(lines) / 2:
            continue
        joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if len(joined) < 60:
            continue
        if joined.isupper():
            continue
        yield joined


def sentences(para):
    # Split on a terminal mark followed by space + capital. Conservative: it
    # under-splits rather than cutting mid-sentence, and an under-split sentence
    # is simply longer, not wrong.
    for s in re.split(r"(?<=[.?])\s+(?=[A-ZÀ-Þ])", para):
        s = s.strip()
        if s:
            yield s


def acceptable(s):
    if any(c in REJECT for c in s):
        return False
    if not s.endswith((".", "?")):
        return False
    n = len(s.split())
    if not (8 <= n <= 40):
        return False
    # Must contain at least one internal comma or be a clean single clause;
    # and no digits-heavy or roman-numeral fragments.
    if re.search(r"\d", s):
        return False
    if re.search(r"\b[IVXLC]{2,}\b", s):
        return False
    # Reject headings and tables of contents. Moby Dick's TOC is long
    # paragraphs of chapter titles that pass every test above and are NOT
    # sentences — scoring a punctuation model on "The Deck Towards the End of
    # the First Night Watch." measures nothing. Title Case is the tell: real
    # prose has few capitalised words mid-sentence, headings are mostly caps.
    words = s.split()
    body = words[1:]  # the first word is capitalised in both cases
    if body:
        capped = sum(1 for w in body if w[:1].isupper())
        if capped / len(body) > 0.4:
            return False
    # Reject transcription/licence boilerplate, which is prose but is about the
    # edition rather than from it.
    if re.search(r"(?i)gutenberg|proofread|ebook|public domain|transcrib|"
                 r"\bLibrary\b|copyright|distributed proofreading", s):
        return False
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out_path, srcs = sys.argv[1], sys.argv[2:]
    picked, per_src = [], {}
    for path in srcs:
        text = strip_gutenberg(open(path, encoding="utf-8", errors="replace").read())
        got = 0
        for para in paragraphs(text):
            for s in sentences(para):
                if acceptable(s):
                    picked.append({"src": path.split("/")[-1], "text": s})
                    got += 1
                    if got >= 40:
                        break
            if got >= 40:
                break
        per_src[path.split("/")[-1]] = got
    marks = sum(sum(1 for c in p["text"] if c in MARKS) for p in picked)
    words = sum(len(p["text"].split()) for p in picked)
    json.dump({"sentences": picked}, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {out_path}: {len(picked)} sentences, {words} words, {marks} marks")
    for k, v in per_src.items():
        print(f"   {k:<22} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
