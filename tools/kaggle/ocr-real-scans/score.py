#!/usr/bin/env python3
"""Scoring for the real-scan OCR set (docs/ocr-real-scans-v1.json). No inference.

Normalisation (applied identically to prediction and ground truth):
  1. strip model markup: Unlimited-OCR-2 `<BLOCK>label[x0, y0, x1, y1]<FORK>` and the native CLI's inline
     `label[x0, y0, x1, y1]` prefixes, any <tag> (HTML table tags, MiniCPM OTSL <fcel>/<nl>/...),
     markdown emphasis/heading/table syntax (`**`, `__`, leading `#`, `|`, `---` rules), `&amp;`-style entities;
  2. NFKC; long s -> s; typographic quotes/apostrophes/dashes -> ASCII; soft hyphen removed;
  3. join line-end hyphenation `x-<ws>y` (y lower-case) -> `xy`; collapse whitespace.
Metrics: cer (normalised, case-sensitive), cer_nospace (all whitespace removed: insensitive to cell/column
separators), wer (lower-cased, punctuation-stripped tokens), word_recall (multiset overlap / GT words;
the primary metric where the GT is partial, e.g. CORD receipts), len_ratio (pred/GT chars).

usage: score.py summarize OUT_MD RESULT_JSON [RESULT_JSON ...]   (merge shard receipts, write table)
"""
import html, json, re, sys, unicodedata
from collections import Counter

try:
    from rapidfuzz.distance import Levenshtein as _L
    def lev(a, b): return _L.distance(a, b)
except Exception:  # pure-python fallback
    def lev(a, b):
        if len(a) < len(b): a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

TRANS = str.maketrans({'ſ': 's', '„': '"', '“': '"', '”': '"', '»': '"', '«': '"', '‚': "'", '‘': "'", '’': "'",
                       '›': "'", '‹': "'", '—': '-', '–': '-', '‐': '-', '‑': '-', '−': '-', '¬': '-', '­': None})


def strip_markup(s):
    s = re.sub(r'(?:<BLOCK>)?[a-z_]*(?:<LEVEL\d>)?\[\d+, ?\d+, ?\d+, ?\d+\](?:<FORK>)?', '\n', s)
    s = re.sub(r'<[^<>]{1,60}>', ' ', s)
    s = html.unescape(s)
    s = re.sub(r'^\s*#{1,6}\s+', '', s, flags=re.M)
    s = re.sub(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$', ' ', s, flags=re.M)
    s = s.replace('**', '').replace('__', '').replace('|', ' ')
    return s


def norm(s):
    s = unicodedata.normalize('NFKC', strip_markup(s or '')).translate(TRANS)
    s = re.sub(r'(\w)-\s+(?=[a-zäöüß])', r'\1', s)
    return ' '.join(s.split())


def words(s):
    return [w for w in re.sub(r'[^\w]+', ' ', s.lower()).split() if w]


def score(pred, gt):
    p, g = norm(pred), norm(gt)
    pw, gw = words(p), words(g)
    pn, gn = p.replace(' ', ''), g.replace(' ', '')
    rec = sum((Counter(pw) & Counter(gw)).values()) / max(1, len(gw))
    return dict(cer=round(lev(p, g) / max(1, len(g)), 4), cer_nospace=round(lev(pn, gn) / max(1, len(gn)), 4),
                wer=round(lev(pw, gw) / max(1, len(gw)), 4), word_recall=round(rec, 4),
                len_ratio=round(len(p) / max(1, len(g)), 3))


def agreement(a, b):
    """Token-ID agreement of native (a) vs original (b)."""
    if not a or not b: return None
    fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
    return dict(ids_exact=a == b, first_difference=fd, prefix_agreement=round((fd if fd is not None else len(a)) / max(1, len(b)), 4),
                native_n=len(a), reference_n=len(b))


def summarize(out_md, paths):
    runs, meta = [], []
    for p in paths:
        d = json.load(open(p)); meta.append({k: d.get(k) for k in ('mode', 'commit', 'cpu_model', 'cpu_count', 'mem_total_gb', 'elapsed_seconds')})
        runs += d.get('runs', [])
    man = json.load(open(paths[0])).get('pages_meta') or {}
    by = {}
    for r in runs: by.setdefault(r['page'], {})[r['arm']] = r
    arms = ['uocr2-f16-fast-tiled', 'uocr2-original-f32', 'minicpm-f16', 'minicpm-original-f32']
    L = ['# OCR on real scans (crispembed-ocr-real-scans)', '', 'Shards: ' + '; '.join(
        f"{m['mode']} @ `{(m['commit'] or '')[:12]}` {m.get('cpu_model')} x{m.get('cpu_count')} {m.get('mem_total_gb')} GB, "
        f"{(m.get('elapsed_seconds') or 0)/3600:.2f} h" for m in meta), '',
         'Metrics after the normalisation in `score.py` (markup stripped). CER / CER without spaces / WER / word recall; '
         'wall s; peak RSS MB. `-` = not run (time budget) or failed (see JSON).', '',
         '| page | ' + ' | '.join(arms) + ' |', '|---|' + '---|' * len(arms)]
    for pid in sorted(by, key=lambda x: (man.get(x, {}).get('priority', 9), x)):
        cells = []
        for a in arms:
            r = by[pid].get(a)
            if not r or r.get('cer') is None:
                cells.append('-' if not r else f"{r.get('skipped') or ('exit ' + str(r.get('exit')))}"); continue
            cap = ' CAP' if r.get('cap_hit') else ''
            cells.append(f"{r['cer']:.3f} / {r['cer_nospace']:.3f} / {r['wer']:.3f} / {r['word_recall']:.3f}; "
                         f"{r.get('wall_seconds')} s; {r.get('peak_rss_mb')} MB{cap}")
        L.append(f'| {pid} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Native vs original token agreement', '', '| page | engine | ids exact | first diff | prefix agr. | native n | ref n | '
          'CER native | CER original |', '|---|---|---|---|---|---|---|---|---|']
    agg = {}
    for pid in sorted(by):
        for eng, na, ra in (('Unlimited-OCR-2', 'uocr2-f16-fast-tiled', 'uocr2-original-f32'), ('MiniCPM-OCR', 'minicpm-f16', 'minicpm-original-f32')):
            n, r = by[pid].get(na), by[pid].get(ra)
            if not n or not r or n.get('cer') is None or r.get('cer') is None: continue
            g = agreement(n.get('ids'), r.get('ids')) or {}
            L.append(f"| {pid} | {eng} | {g.get('ids_exact')} | {g.get('first_difference')} | {g.get('prefix_agreement')} | "
                     f"{g.get('native_n')} | {g.get('reference_n')} | {n['cer']} | {r['cer']} |")
            a = agg.setdefault(eng, dict(pairs=0, exact=0, cer_n=0.0, cer_r=0.0, text_equal=0))
            a['pairs'] += 1; a['exact'] += bool(g.get('ids_exact')); a['cer_n'] += n['cer']; a['cer_r'] += r['cer']
            a['text_equal'] += norm(n.get('text')) == norm(r.get('text'))
    L += ['', '| engine | paired pages | IDs exact | normalised text equal | mean CER native | mean CER original |', '|---|---|---|---|---|---|']
    for eng, a in agg.items():
        L.append(f"| {eng} | {a['pairs']} | {a['exact']} | {a['text_equal']} | {a['cer_n']/a['pairs']:.4f} | {a['cer_r']/a['pairs']:.4f} |")
    open(out_md, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return by, agg


if __name__ == '__main__':
    if sys.argv[1] == 'summarize':
        summarize(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
