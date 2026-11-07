"""Derive block bounding boxes by aligning gold block text to OCR word boxes. No inference here:
the word boxes come from a Tesseract TSV produced elsewhere (hosted CI).

Algorithm (deterministic):
  1. tokens: NFKC, long s -> s, lower case, keep word characters only; gold tokens remember their block.
  2. exact alignment of the gold and OCR token streams (difflib, autojunk off);
  3. inside each unmatched gap bounded by exact anchors, pair tokens positionally when the gap
     lengths are within 2x and the character similarity is >= 0.6 (absorbs OCR misspellings);
  4. per block: drop matched words whose OCR stream index is far from the block's median index
     (> 3 x block token count + 5), then box = union of the remaining word boxes;
  5. a block gets a box only if >= 30% of its tokens matched and >= 2 words matched (or all of a
     one-token block). Otherwise the box stays null - an unaligned block is not guessed.
The match ratio is recorded per block so consumers can filter weak boxes.
"""
from __future__ import annotations

import difflib
import re
import unicodedata


def tokens(s: str) -> list[str]:
    s = unicodedata.normalize('NFKC', s or '').replace('ſ', 's').lower()
    return [t for t in re.sub(r'[^\w]+', ' ', s).split() if t]


def read_tsv(text: str) -> list[dict]:
    """Tesseract TSV -> word dicts {text, box}. Level-5 rows with non-empty text only."""
    out = []
    lines = text.splitlines()
    if not lines:
        return out
    head = lines[0].split('\t')
    ix = {k: i for i, k in enumerate(head)}
    for line in lines[1:]:
        f = line.split('\t')
        if len(f) < len(head) or f[ix['level']] != '5' or not f[ix['text']].strip():
            continue
        x, y, w, h = (int(f[ix[k]]) for k in ('left', 'top', 'width', 'height'))
        out.append(dict(text=f[ix['text']], box=[x, y, x + w, y + h]))
    return out


def align(blocks: list[dict], words: list[dict]) -> list[tuple[list[int] | None, float]]:
    gold, owner = [], []
    for bi, b in enumerate(blocks):
        for t in tokens(b.get('text', '')):
            gold.append(t)
            owner.append(bi)
    ocr, wref = [], []
    for wi, w in enumerate(words):
        for t in tokens(w['text']):
            ocr.append(t)
            wref.append(wi)
    match = {}  # gold index -> ocr index
    sm = difflib.SequenceMatcher(None, gold, ocr, autojunk=False)
    blocks_m = sm.get_matching_blocks()
    prev_g = prev_o = 0
    for mb in blocks_m:
        gap_g, gap_o = mb.a - prev_g, mb.b - prev_o
        if gap_g and gap_o and max(gap_g, gap_o) <= 2 * min(gap_g, gap_o):
            for k in range(min(gap_g, gap_o)):
                gi, oi = prev_g + k, prev_o + k
                if difflib.SequenceMatcher(None, gold[gi], ocr[oi]).ratio() >= 0.6:
                    match[gi] = oi
        for k in range(mb.size):
            match[mb.a + k] = mb.b + k
        prev_g, prev_o = mb.a + mb.size, mb.b + mb.size
    res = []
    for bi, b in enumerate(blocks):
        gidx = [i for i, o in enumerate(owner) if o == bi]
        if not gidx:
            res.append((None, 0.0))
            continue
        oidx = sorted(match[g] for g in gidx if g in match)
        ratio = len(oidx) / len(gidx)
        if oidx:
            med = oidx[len(oidx) // 2]
            lim = 3 * len(gidx) + 5
            oidx = [o for o in oidx if abs(o - med) <= lim]
        wset = sorted({wref[o] for o in oidx})
        enough = (len(gidx) == 1 and len(wset) == 1) or (len(wset) >= 2 and ratio >= 0.3)
        if not enough:
            res.append((None, ratio))
            continue
        bx = [words[w]['box'] for w in wset]
        res.append(([min(x[0] for x in bx), min(x[1] for x in bx), max(x[2] for x in bx), max(x[3] for x in bx)],
                    ratio))
    return res
