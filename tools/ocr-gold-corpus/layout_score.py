#!/usr/bin/env python3
"""Layout / reading-order / table scorer for docs/ocr-gold-corpus-v1.json. No inference, stdlib only.

Companion of the text scorer tools/kaggle/ocr-real-scans/score.py (CER/WER), whose normalisation is
reused for block text so both scorers agree on what "the same text" is.

Prediction format (per page): a list of blocks, in the predicted reading order:
    {"bbox": [x0, y0, x1, y1] | null, "type": "text|title|table|list|figure|footnote|header|footer",
     "text": "...", "table": {"cells": [{"row", "col", "text"}, ...]} | null, "html": "<table>..." | null}
A predicted table may give cells, an HTML <table> (``html`` or ``text``) or a Markdown pipe table.

Matching (one-to-one):
  * box mode   - gold blocks that have a bbox are matched to predicted blocks with a bbox by maximum
                 IoU (greedy over all pairs in descending IoU; ties by gold order), threshold IoU >= 0.5.
  * text mode  - gold blocks without a bbox (or predictions without boxes) are matched by normalised
                 text similarity (difflib ratio >= 0.5), greedy over descending similarity.
Metrics per page:
  block_precision / block_recall / block_f1   over matched pairs (both modes counted)
  mean_iou                                    over box-mode pairs
  type_accuracy                               matched pairs whose types agree (aliases folded)
  reading_order_tau                           Kendall tau-b between gold order and predicted order of the
                                              matched pairs (1 = same order, -1 = reversed; None if < 2)
  table_cell_f1 / table_content_f1            per matched gold table: F1 over (row, col, text) cells and over
                                              the cell-text multiset; averaged over gold tables (unmatched
                                              gold table = 0)
usage: layout_score.py GOLD_CORPUS_JSON PRED_JSON [OUT_JSON]
       PRED_JSON = {"<page id>": [blocks...], ...}
"""
from __future__ import annotations

import difflib
import html as _html
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kaggle' / 'ocr-real-scans'))
from score import norm  # noqa: E402

TYPES = ('text', 'title', 'table', 'list', 'figure', 'footnote', 'header', 'footer')
ALIASES = {
    'paragraph': 'text', 'plain text': 'text', 'plain_text': 'text', 'abstract': 'text', 'caption': 'text',
    'formula': 'text', 'equation': 'text', 'code': 'text', 'section-header': 'title', 'section_header': 'title',
    'heading': 'title', 'doc_title': 'title', 'paragraph_title': 'title', 'list_item': 'list', 'list-item': 'list',
    'picture': 'figure', 'image': 'figure', 'chart': 'figure', 'page-header': 'header', 'page_header': 'header',
    'page-footer': 'footer', 'page_footer': 'footer', 'page_number': 'footer', 'page-number': 'footer',
    'footnote': 'footnote', 'page_footnote': 'footnote',
}
IOU_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5


def canon_type(t: str | None) -> str:
    t = (t or '').strip().lower()
    return t if t in TYPES else ALIASES.get(t, ALIASES.get(t.replace(' ', '_'), t))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def text_sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def kendall_tau_b(x: list, y: list) -> float | None:
    """Kendall tau-b of two equally long rankings (ties allowed). None when fewer than 2 items."""
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / den if den else None


class _TableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.cell, self.spans = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'tr':
            self.rows.append([])
        elif tag in ('td', 'th'):
            if not self.rows:
                self.rows.append([])
            self.cell = []
            self.spans = (int(a.get('rowspan') or 1), int(a.get('colspan') or 1))

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.rows[-1].append((' '.join(''.join(self.cell).split()), *self.spans))
            self.cell = None

    def handle_data(self, d):
        if self.cell is not None:
            self.cell.append(d)


def cells_from_html(s: str) -> list[dict]:
    p = _TableHTML()
    p.feed(s)
    taken, out = set(), []
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs in row:
            while (r, c) in taken:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            out.append(dict(row=r, col=c, text=text))
            c += cs
    return out


def cells_from_markdown(s: str) -> list[dict]:
    rows = [l.strip() for l in s.splitlines() if l.strip().startswith('|') or l.count('|') >= 2]
    rows = [r for r in rows if not re.fullmatch(r'\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?', r)]
    out = []
    for r, line in enumerate(rows):
        parts = line.strip().strip('|').split('|')
        out += [dict(row=r, col=c, text=t.strip()) for c, t in enumerate(parts)]
    return out


def table_cells(block: dict) -> list[dict]:
    t = block.get('table') or {}
    if t.get('cells'):
        return t['cells']
    for key in ('html', 'text'):
        s = block.get(key) or ''
        if '<table' in s.lower() or '<td' in s.lower():
            return cells_from_html(s)
    s = block.get('text') or ''
    return cells_from_markdown(s) if '|' in s else []


def _cell_key(text: str) -> str:
    return norm(_html.unescape(text or '')).lower()


def table_f1(gold: list[dict], pred: list[dict]) -> tuple[float, float]:
    """(structure-aware cell F1 over (row, col, text), content F1 over the cell-text multiset)."""
    g = Counter((c['row'], c['col'], _cell_key(c['text'])) for c in gold if _cell_key(c['text']))
    p = Counter((c['row'], c['col'], _cell_key(c['text'])) for c in pred if _cell_key(c['text']))

    def f1(a: Counter, b: Counter) -> float:
        if not a and not b:
            return 1.0
        tp = sum((a & b).values())
        if not tp:
            return 0.0
        pr, rc = tp / sum(b.values()), tp / sum(a.values())
        return 2 * pr * rc / (pr + rc)

    gc = Counter(k[2] for k in g.elements())
    pc = Counter(k[2] for k in p.elements())
    return f1(g, p), f1(gc, pc)


def match(gold: list[dict], pred: list[dict]) -> list[tuple[int, int, str, float]]:
    """One-to-one (gold index, pred index, mode, score) pairs."""
    pairs, used_g, used_p = [], set(), set()
    cand = [(iou(g['bbox'], p['bbox']), gi, pi) for gi, g in enumerate(gold) if g.get('bbox')
            for pi, p in enumerate(pred) if p.get('bbox')]
    for s, gi, pi in sorted(cand, key=lambda t: (-t[0], t[1], t[2])):
        if s < IOU_THRESHOLD:
            break
        if gi in used_g or pi in used_p:
            continue
        pairs.append((gi, pi, 'box', s))
        used_g.add(gi)
        used_p.add(pi)
    cand = [(text_sim(g.get('text', ''), p.get('text', '') or p.get('html', '') or ''), gi, pi)
            for gi, g in enumerate(gold) if gi not in used_g
            for pi, p in enumerate(pred) if pi not in used_p]
    for s, gi, pi in sorted(cand, key=lambda t: (-t[0], t[1], t[2])):
        if s < TEXT_THRESHOLD:
            break
        if gi in used_g or pi in used_p:
            continue
        # a gold block with a box is only text-matched to a prediction without a box
        if gold[gi].get('bbox') and pred[pi].get('bbox'):
            continue
        pairs.append((gi, pi, 'text', s))
        used_g.add(gi)
        used_p.add(pi)
    return sorted(pairs)


def score_page(gold_blocks: list[dict], pred_blocks: list[dict]) -> dict:
    gold = sorted(gold_blocks, key=lambda b: b.get('order', 0))
    pairs = match(gold, pred_blocks)
    n = len(pairs)
    prec = n / len(pred_blocks) if pred_blocks else (1.0 if not gold else 0.0)
    rec = n / len(gold) if gold else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ious = [s for _, _, m, s in pairs if m == 'box']
    types = [canon_type(gold[g]['type']) == canon_type(pred_blocks[p].get('type')) for g, p, _, _ in pairs]
    tau = kendall_tau_b([g for g, _, _, _ in pairs], [p for _, p, _, _ in pairs])
    by_g = {g: p for g, p, _, _ in pairs}
    cell, content = [], []
    for gi, g in enumerate(gold):
        if canon_type(g['type']) != 'table' or not table_cells(g):
            continue
        if gi in by_g:
            a, b = table_f1(table_cells(g), table_cells(pred_blocks[by_g[gi]]))
        else:
            a, b = 0.0, 0.0
        cell.append(a)
        content.append(b)
    mean = lambda v: sum(v) / len(v) if v else None
    return dict(gold_blocks=len(gold), pred_blocks=len(pred_blocks), matched=n,
                matched_by_box=len(ious), matched_by_text=n - len(ious),
                block_precision=prec, block_recall=rec, block_f1=f1, mean_iou=mean(ious),
                type_accuracy=mean(types), reading_order_tau=tau,
                table_cell_f1=mean(cell), table_content_f1=mean(content), gold_tables=len(cell))


def score_corpus(corpus: dict, preds: dict) -> dict:
    pages = {}
    for p in corpus['pages']:
        if p['id'] in preds:
            pages[p['id']] = score_page(p['blocks'], preds[p['id']])
    keys = ('block_f1', 'mean_iou', 'type_accuracy', 'reading_order_tau', 'table_cell_f1', 'table_content_f1')
    summary = {}
    for k in keys:
        v = [r[k] for r in pages.values() if r[k] is not None]
        summary[k] = sum(v) / len(v) if v else None
    summary['pages'] = len(pages)
    summary['missing_pages'] = sorted(set(p['id'] for p in corpus['pages']) - set(preds))
    return dict(summary=summary, pages=pages)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    res = score_corpus(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    s = json.dumps(res, indent=1)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(s + '\n')
    print(json.dumps(res['summary'], indent=1))
