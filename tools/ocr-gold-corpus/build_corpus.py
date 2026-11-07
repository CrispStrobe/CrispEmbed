#!/usr/bin/env python3
"""Build docs/ocr-gold-corpus-v1.json: licensed OCR gold pages with layout / reading-order annotations.

No inference anywhere in this file. Three subcommands, each writing the corpus JSON:

  fetch OUT_JSON            network only (Wikisource/Commons APIs, Hub datasets-server JSON rows,
                            one download per Wikimedia image to pin its sha256). Pages listed in
                            candidates.json; pages reused from docs/ocr-real-scans-v2.json keep that
                            manifest's image pin byte-for-byte. Blocks get null bboxes here.
  cord-pin IN OUT PARQUET   fill CORD image sha256/bytes from the pinned Hub parquet (hosted CI).
  boxes IN OUT TSV_DIR      derive Wikisource block bboxes by aligning each block's transcription to
                            Tesseract word boxes (TSV_DIR/<page id>.tsv, produced on hosted CI by
                            tools/ci-heavy/ocr_gold_corpus.sh); see align_boxes.py.

Licence policy (enforced in fetch): image licence must be public domain / CC0 / CC BY / CC BY-SA;
anything containing NC or ND, or an empty/unknown licence, is rejected and listed in
``rejected_sources`` with the reason. Wikisource transcriptions are CC BY-SA 4.0.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import wikisource_layout as WL  # noqa: E402

UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}
WS_TEXT_LICENSE = ('CC BY-SA 4.0 (Wikisource contributors; transcription of a public-domain work, '
                   'see the page history at the revision URL)')
CORD_REV = '7f0115a4b758a71d6473b8d085751692da2fef98'
CORD_FILE = 'data/test-00000-of-00001-9c204eb3f4e11791.parquet'
CORD_URL = f'https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{CORD_REV}/{CORD_FILE}'
CORD_PARQUET_SHA = '51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7'
CORD_ROWS = 'https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default&split=test'
CORD_LICENSE = 'CC BY 4.0'
CORD_ATTR = ('CORD: A Consolidated Receipt Dataset for Post-OCR Parsing, Park et al., NAVER CLOVA '
             '(naver-clova-ix/cord-v2)')


def get(url, data=None, tries=4):
    err = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=180).read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(4 * (i + 1))
    raise err


def api(host, **p):
    p['format'] = 'json'
    p['formatversion'] = '2'
    return json.loads(get(f'https://{host}/w/api.php', urllib.parse.urlencode(p).encode()))


def licence_ok(lic: str) -> tuple[bool, str]:
    s = (lic or '').strip()
    u = s.upper().replace('-', ' ')
    if not s:
        return False, 'no licence in Commons metadata (unclear)'
    if re.search(r'\bNC\b|NON ?COMMERCIAL', u):
        return False, f'licence {s!r} is non-commercial'
    if re.search(r'\bND\b|NO ?DERIV', u):
        return False, f'licence {s!r} forbids derivatives'
    if 'PUBLIC DOMAIN' in u or u.startswith('PD') or 'CC0' in u or re.match(r'^CC BY( SA)?( \d(\.\d)?)?', u):
        return True, ''
    return False, f'licence {s!r} not on the allow-list (PD/CC0/CC BY/CC BY-SA)'


# ---------------------------------------------------------------- Wikisource

def ws_revision(wiki, title, revid=None):
    host = f'{wiki}.wikisource.org'
    if revid:
        q = api(host, action='query', revids=revid, prop='proofread|revisions', rvprop='ids|content|timestamp',
                rvslots='main')
    else:
        q = api(host, action='query', titles=title, prop='proofread|revisions', rvprop='ids|content|timestamp',
                rvslots='main')
    pg = q['query']['pages'][0]
    rev = pg['revisions'][0]
    return pg, rev


def ws_blocks(wiki, title, wikitext):
    html = api(f'{wiki}.wikisource.org', action='parse', title=title, text=WL.sectioned_wikitext(wikitext),
               prop='text', contentmodel='wikitext', disablelimitreport=1)['parse']['text']
    return WL.blocks_from_html(html, wiki)


def commons_image(wiki, title, width=1600):
    m = re.match(r'^[^:]+:(.+?)(?:/(\d+))?$', title)
    fname, n = m.group(1), m.group(2)
    ii = dict(action='query', titles='File:' + fname, prop='imageinfo',
              iiprop='url|sha1|size|extmetadata|mime|timestamp', iiurlwidth=str(width))
    if n:
        ii['iiurlparam'] = f'page{n}-{width}px'
    fh = 'commons.wikimedia.org'
    fp = api(fh, **ii)['query']['pages'][0]
    if fp.get('missing') or 'imageinfo' not in fp:
        fh = f'{wiki}.wikisource.org'
        fp = api(fh, **ii)['query']['pages'][0]
    inf = fp['imageinfo'][0]
    em = inf.get('extmetadata', {})
    clean = lambda k: re.sub(r'\s+', ' ', re.sub('<[^>]+>', '', em.get(k, {}).get('value', ''))).strip()[:300]
    ts = inf['timestamp'].replace('-', '').replace(':', '').replace('T', '').replace('Z', '')
    paged = bool(n)
    file_url = inf['url'].split('?')[0]
    if paged:
        bare = inf['thumburl'].split('?')[0]
        policy = ('per-page render of a multi-page DjVu/PDF (Wikimedia thumbnail): bytes are re-rendered on '
                  'demand and have been observed to drift; the sha256 is the gate, the revision-qualified URL '
                  'is tried first, then the bare URL; on mismatch re-render page '
                  f'{n} of the pinned file revision (file_sha1) at {width}px and record the new sha256 as a '
                  'new corpus version rather than editing this one')
    else:
        bare = file_url  # original upload: immutable bytes for a given file revision
        policy = 'original uploaded file at the pinned revision (immutable bytes)'
    url = f'{bare}?{ts}'
    img = get(url)
    w = inf.get('thumbwidth') if paged else inf.get('width')
    h = inf.get('thumbheight') if paged else inf.get('height')
    image = dict(url=url, sha256=hashlib.sha256(img).hexdigest(), bytes=len(img), width=w, height=h,
                 derived_from=dict(file='File:' + fname, page=n, file_url=file_url, file_sha1=inf['sha1'], host=fh),
                 fallback_urls=[bare] if bare != url else [],
                 revision=dict(timestamp=inf['timestamp'], file_sha1=inf['sha1'],
                               api=f'https://{fh}/w/api.php?' + urllib.parse.urlencode(dict(
                                   action='query', titles='File:' + fname, prop='imageinfo',
                                   iiprop='timestamp|sha1', iilimit=1, format='json'))),
                 bytes_policy=policy)
    lic = dict(license=clean('LicenseShortName'), usage_terms=clean('UsageTerms'),
               copyrighted=clean('Copyrighted') not in ('False', 'false', ''), attribution=clean('Artist')
               or clean('Credit'))
    if clean('Copyrighted') in ('False', 'false'):
        lic['copyrighted'] = False
    return image, lic


def apply_manual(blocks, ops):
    """Manual edits from visual inspection of the pinned scan (all recorded in the corpus)."""
    out = list(blocks)
    for op in ops:
        kind = op[0]
        if kind == 'header':  # printed running head / page number missing from the transcription
            out.insert(0, dict(type='header', text=op[1], type_provenance='manual:scan-inspection', section='header'))
        elif kind == 'footer':
            out.append(dict(type='footer', text=op[1], type_provenance='manual:scan-inspection', section='footer'))
        elif kind == 'title_before':  # printed title/numeral before the block starting with op[2]
            i = next(i for i, b in enumerate(out) if b['text'].startswith(op[2]))
            out.insert(i, dict(type='title', text=op[1], type_provenance='manual:scan-inspection', section='body'))
        elif kind == 'drop_line':  # wiki-only line (e.g. verse numbering) not printed on the scan
            for b in out:
                b['text'] = '\n'.join(l for l in b['text'].split('\n') if l.strip() != op[1])
        elif kind == 'retype':  # (retype, text-prefix, new type)
            for b in out:
                if b['text'].startswith(op[1]):
                    b['type'], b['type_provenance'] = op[2], 'manual:scan-inspection'
        elif kind == 'table_header':  # (table_header, first-cell-prefix, "cell|cell|...")
            for b in out:
                if b['type'] == 'table' and b['text'].startswith(op[1]):
                    t = b['table']
                    for c in t['cells']:
                        c['row'] += 1
                    t['cells'][:0] = [dict(row=0, col=i, rowspan=1, colspan=1, header=True, text=x,
                                           provenance='manual:scan-inspection') for i, x in enumerate(op[2].split('|'))]
                    t['rows'] += 1
                    b['text'] = ' '.join(op[2].split('|')) + '\n' + b['text']
        else:
            raise SystemExit(f'unknown manual op {op!r}')
    out = [b for b in out if b['text'] or b['type'] in ('figure', 'table')]
    for i, b in enumerate(out):
        b['order'] = i
        b['id'] = f'b{i}'
    return out


def finalise_block(b):
    b.setdefault('bbox', None)
    b.setdefault('bbox_provenance', None)
    return {k: b[k] for k in ('id', 'order', 'type', 'text', 'bbox', 'bbox_provenance', 'type_provenance', 'section')
            if k in b} | {k: v for k, v in b.items() if k in ('table', 'items', 'markers', 'note_id', 'cord_category')}


def ws_page(c, v2):
    wiki, title = c['wiki'], c['title']
    pg, rev = ws_revision(wiki, title, c.get('revision_id'))
    q = pg.get('proofread', {}).get('quality')
    if q != 4:
        return None, f'{wiki}:{title}: proofread quality {q} (only Validated = 4 is accepted)'
    if c.get('v2_id'):
        base = next(p for p in v2['pages'] if p['id'] == c['v2_id'])
        image, lic = dict(base['image']), base['image_license']
        if 'bytes_policy' not in image:
            image['bytes_policy'] = ('pinned in docs/ocr-real-scans-v2.json (revision-qualified URL, sha256 gate); '
                                     'see that manifest for the drift history')
    else:
        image, lic = commons_image(wiki, title)
    ok, why = licence_ok(lic.get('license', ''))
    if not ok:
        return None, f'{wiki}:{title}: {why}'
    blocks = apply_manual(ws_blocks(wiki, title, rev['slots']['main']['content']), c.get('manual', []))
    return dict(
        id=c['id'], tags=c['tags'], lang=c['tags'][0] if c['tags'][0] in ('en', 'de', 'fr') else None,
        source='wikisource', v2_id=c.get('v2_id'), image=image, image_license=lic,
        ground_truth_license=WS_TEXT_LICENSE,
        provenance=dict(
            kind='Wikisource proofread transcription, status Validated (quality 4)', wiki=f'{wiki}.wikisource.org',
            page=title, revision_id=rev['revid'], revision_timestamp=rev['timestamp'],
            revision_url=f'https://{wiki}.wikisource.org/w/index.php?oldid={rev["revid"]}',
            text_and_types='derived: rendered revision HTML (sectioned header/body/footer) -> blocks, see '
                           'tools/ocr-gold-corpus/wikisource_layout.py',
            reading_order='derived: transcription document order (header, body, footnotes, footer)',
            bboxes='pending: derived on hosted CI by Tesseract word alignment (build_corpus.py boxes)',
            manual_edits=[list(op) for op in c.get('manual', [])]),
        blocks=[finalise_block(b) for b in blocks]), None


# ---------------------------------------------------------------- CORD

def cord_rows(rows):
    out = {}
    for start in range(0, 100, 100):
        d = json.loads(get(f'{CORD_ROWS}&offset={start}&length=100'))
        for r in d['rows']:
            if r['row_idx'] in rows:
                out[r['row_idx']] = json.loads(r['row']['ground_truth'])
    return out


def quad_box(q):
    xs = [q[f'x{i}'] for i in range(1, 5)]
    ys = [q[f'y{i}'] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def cord_page(c, gt):
    meta = gt['meta']
    groups = {}
    for line in gt['valid_line']:
        key = (line['category'].split('.')[0], line['group_id'])
        groups.setdefault(key, []).append(line)

    def words(lines):
        ws = [w for l in lines for w in l['words']]
        ws.sort(key=lambda w: (round(sum(quad_box(w['quad'])[1::2]) / 2 / 12), quad_box(w['quad'])[0]))
        return ws

    blocks = []
    menu = [(k, v) for k, v in groups.items() if k[0] == 'menu']
    if menu:
        fields = []
        rows = []
        for (_, gid), lines in menu:
            by_sub = {}
            for l in lines:
                by_sub.setdefault(l.get('sub_group_id', 0), []).append(l)
            for sub, ls in by_sub.items():
                rows.append((min(quad_box(w['quad'])[1] for l in ls for w in l['words']), gid, sub, ls))
                for l in ls:
                    f = l['category'].split('.', 1)[1]
                    if f not in fields:
                        fields.append(f)
        rows.sort()
        cells, boxes = [], []
        for r, (_, gid, sub, ls) in enumerate(rows):
            for l in ls:
                ws = sorted(l['words'], key=lambda w: quad_box(w['quad'])[0])
                bx = union([quad_box(w['quad']) for w in ws])
                boxes.append(bx)
                cells.append(dict(row=r, col=fields.index(l['category'].split('.', 1)[1]), rowspan=1, colspan=1,
                                  header=False, text=' '.join(w['text'] for w in ws), bbox=bx,
                                  field=l['category']))
        blocks.append(dict(type='table', bbox=union(boxes),
                           text='\n'.join(' '.join(c['text'] for c in sorted(cells, key=lambda c: c['col'])
                                                    if c['row'] == r) for r in range(len(rows))),
                           table=dict(rows=len(rows), cols=len(fields), columns=fields, cells=cells),
                           cord_category='menu',
                           type_provenance='derived:cord-v2 menu.* lines -> table (rows=group/sub_group, cols=field)'))
    for (cat, gid), lines in groups.items():
        if cat == 'menu':
            continue
        ws = words(lines)
        blocks.append(dict(type='text', bbox=union([quad_box(w['quad']) for w in ws]),
                           text=' '.join(w['text'] for w in ws), cord_category=f'{cat}#{gid}',
                           type_provenance=f'derived:cord-v2 {cat}.* group -> text'))
    W, H = meta['image_size']['width'], meta['image_size']['height']
    clip = lambda b: [max(0, min(W, b[0])), max(0, min(H, b[1])), max(0, min(W, b[2])), max(0, min(H, b[3]))]
    for b in blocks:  # a few annotator quads overhang the image edge by a pixel or two
        b['bbox'] = clip(b['bbox'])
        for cell in (b.get('table') or {}).get('cells', []):
            cell['bbox'] = clip(cell['bbox'])
    blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
    for i, b in enumerate(blocks):
        b.update(id=f'b{i}', order=i, section='body', bbox_provenance='manual:cord-v2 word quads (union)')
    return dict(
        id=c['id'], tags=c['tags'], lang=None, source='cord-v2', v2_id=c.get('v2_id'),
        image=dict(hf_parquet=dict(url=CORD_URL, sha256=CORD_PARQUET_SHA, row=c['row'], column='image.bytes'),
                   sha256=c.get('sha256'), bytes=c.get('bytes'), width=meta['image_size']['width'],
                   height=meta['image_size']['height'],
                   bytes_policy='image bytes stored in the pinned Hub parquet (immutable at the revision); '
                                'licence permits redistribution with attribution'),
        image_license=dict(license=CORD_LICENSE, attribution=CORD_ATTR), ground_truth_license=CORD_LICENSE,
        provenance=dict(kind='CORD-v2 manual word quads and field categories', dataset='naver-clova-ix/cord-v2',
                        revision=CORD_REV, split='test', row=c['row'],
                        text_and_types='derived from manual annotations: menu.* -> one table block '
                                       '(rows by group_id/sub_group_id, columns by field), other groups -> text',
                        reading_order='derived: blocks by top edge, then left edge',
                        bboxes='manual (dataset word quads; block box = union)',
                        scope='PARTIAL: only annotated fields; store header/address/separators are not annotated'),
        blocks=[finalise_block(b) for b in blocks])


# ---------------------------------------------------------------- commands

def cmd_fetch(out):
    cand = json.loads((HERE / 'candidates.json').read_text())
    v2 = json.loads((ROOT / 'docs/ocr-real-scans-v2.json').read_text())
    pages, rejected = [], list(cand.get('rejected_sources', []))
    for c in cand['wikisource']:
        try:
            p, why = ws_page(c, v2)
        except Exception as e:  # noqa: BLE001
            p, why = None, f'{c["wiki"]}:{c["title"]}: fetch/parse error {e!r}'
        if p:
            pages.append(p)
            print('ok', p['id'], len(p['blocks']), sorted({b['type'] for b in p['blocks']}), flush=True)
        else:
            rejected.append(dict(source=f'{c["wiki"]}.wikisource: {c["title"]}', reason=why.split(': ', 2)[-1]))
            print('REJECT', why, flush=True)
        time.sleep(0.5)
    gts = cord_rows({c['row'] for c in cand['cord']})
    for c in cand['cord']:
        if c.get('v2_id'):
            im = next(p for p in v2['pages'] if p['id'] == c['v2_id'])['image']
            c = dict(c, sha256=im['sha256'], bytes=im['bytes'])
        pages.append(cord_page(c, gts[c['row']]))
        print('ok', c['id'], flush=True)
    m = dict(
        schema='crispembed-ocr-gold-corpus-v1',
        description=('Licensed real-page OCR gold corpus with layout annotations: block boxes, block types '
                     '(text/title/table/list/figure/footnote/header/footer), reading order and table structure. '
                     'Images are fetched by URL / pinned parquet and verified by sha256 (not committed). Every '
                     'annotation records whether it is derived (rule-based from Wikisource HTML, Tesseract word '
                     'alignment, CORD field grouping) or manual (dataset annotators, scan inspection).'),
        builder='tools/ocr-gold-corpus/build_corpus.py (fetch; cord-pin and boxes on hosted CI)',
        scorer='tools/ocr-gold-corpus/layout_score.py (blocks) + tools/kaggle/ocr-real-scans/score.py (text)',
        companion='docs/ocr-real-scans-v2.json (unchanged; v2_id links reused pages)',
        block_types=['text', 'title', 'table', 'list', 'figure', 'footnote', 'header', 'footer'],
        bbox_format='[x0, y0, x1, y1] pixels of the pinned image; null = not annotated',
        policy=cand['policy'], pages=pages, rejected_sources=rejected)
    Path(out).write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')
    print('wrote', out, len(pages), 'pages')


def cmd_cord_pin(src, dst, parquet):
    import pyarrow.parquet as pq
    if hashlib.sha256(Path(parquet).read_bytes()).hexdigest() != CORD_PARQUET_SHA:
        raise SystemExit('parquet sha256 mismatch')
    t = pq.read_table(parquet, columns=['image']).column('image').to_pylist()
    m = json.loads(Path(src).read_text())
    for p in m['pages']:
        if p['source'] == 'cord-v2':
            b = t[p['image']['hf_parquet']['row']]['bytes']
            h = hashlib.sha256(b).hexdigest()
            if p['image'].get('sha256') and p['image']['sha256'] != h:
                raise SystemExit(f"{p['id']}: parquet image {h} != pinned {p['image']['sha256']}")
            p['image']['sha256'], p['image']['bytes'] = h, len(b)
    Path(dst).write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')


def cmd_boxes(src, dst, tsv_dir, tool_version):
    import align_boxes as A
    m = json.loads(Path(src).read_text())
    for p in m['pages']:
        if p['source'] != 'wikisource':
            continue
        f = Path(tsv_dir) / f'{p["id"]}.tsv'
        if not f.exists():
            continue
        size = A.page_size(f.read_text())
        im = p['image']
        if size and (size[0], size[1]) != (im['width'], im['height']):
            # Wikimedia snaps paged renders to standard widths (e.g. 1920px for a 1600px request);
            # the API-reported thumb size is not the size of the served bytes.
            im['api_reported_size'] = [im['width'], im['height']]
            im['width'], im['height'] = size
            im['size_provenance'] = 'measured from the verified bytes (Tesseract page row)'
        words = A.read_tsv(f.read_text())
        res = A.align(p['blocks'], words)
        for b, (box, ratio) in zip(p['blocks'], res):
            b['bbox'] = box
            b['bbox_provenance'] = (f'derived:{tool_version} word alignment (matched {ratio:.2f} of block words)'
                                    if box else None)
        p['provenance']['bboxes'] = (f'derived: {tool_version} word boxes aligned to the transcription '
                                     '(tools/ocr-gold-corpus/align_boxes.py); null where too few words matched')
    Path(dst).write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['fetch']:
        cmd_fetch(a[1])
    elif a[:1] == ['cord-pin']:
        cmd_cord_pin(*a[1:4])
    elif a[:1] == ['boxes']:
        cmd_boxes(*a[1:5])
    else:
        raise SystemExit(__doc__)
