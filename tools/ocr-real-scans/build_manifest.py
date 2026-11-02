#!/usr/bin/env python3
"""Build docs/ocr-real-scans-v1.json: real document scans with permissive licences and ground truth.

No inference. Fetches (network only) the pinned Wikisource revisions + Commons scans and the
CORD-v2 test parquet at a pinned Hub revision, records sha256 of every image, derives the
ground-truth text and applies the documented, per-page manual edits listed below.

usage: build_manifest.py WORKDIR [OUT_JSON]
"""
import hashlib, json, os, sys, urllib.request
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wikisource as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORD_REV = '7f0115a4b758a71d6473b8d085751692da2fef98'
CORD_FILE = 'data/test-00000-of-00001-9c204eb3f4e11791.parquet'
CORD_URL = f'https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{CORD_REV}/{CORD_FILE}'
WS_TEXT_LICENSE = ('CC BY-SA 4.0 (Wikisource contributors; transcription of a public-domain work, '
                   'see the page history at the revision URL)')

# id, wiki, title, tags, priority (1 = run first), edits [(op, arg...)]
# Edits: de.wikisource does not transcribe running heads / page numbers / section numerals that
# are printed on the scan; they are added from visual inspection of the pinned scan and marked here.
WS = [
    ('en-prose-constitution-13', 'en', 'Page:The English Constitution (1894).djvu/13', ['en', 'prose', 'clean'], 1, []),
    ('de-fraktur-grimm-319', 'de', 'Seite:Kinder- und Haus-Märchen 1837 Band 2.djvu/319', ['de', 'fraktur', 'umlauts', 'prose'], 1,
     [('prepend', '— 310 —')]),
    ('en-table-climatic-26', 'en', 'Page:Climatic Cycles and Tree-Growth - 1919.djvu/26', ['en', 'table'], 1,
     [('insert_before', '1 feet. 6,125', 'Group. Altitude. Date of cutting. Cutting season. Remarks.')]),
    ('de-antiqua-kafka-schloss-211', 'de', 'Seite:De Kafka Schloß 211.jpg', ['de', 'antiqua', 'umlauts', 'prose'], 1,
     [('append', '[211]')]),
    ('en-form-economic-journal-67', 'en', 'Page:The Economic Journal Volume 1.djvu/67', ['en', 'form', 'table'], 1, []),
    ('de-2col-meyers-b13-103', 'de', 'Seite:Meyers b13 s0103.jpg', ['de', 'fraktur', 'multi-column', 'dense', 'umlauts'], 2,
     [('prepend', 'Plagiocephalus — Planché. 103')]),
    ('en-2col-eb1911-v04-246', 'en', 'Page:EB1911 - Volume 04.djvu/246', ['en', 'multi-column', 'figures'], 2, []),
    ('de-degraded-freiligrath-46', 'de', 'Seite:Ça ira! Freiligrath-1846.pdf/46', ['de', 'fraktur', 'degraded', 'bleed-through', 'verse'], 1,
     [('drop_line', '5'), ('drop_line', '10'), ('prepend', '44')]),
    ('en-report-amtrak-23', 'en', 'Page:Derailment of Amtrak Passenger Train 188 Philadelphia, Pennsylvania May 12, 2015.djvu/23',
     ['en', 'prose', 'modern'], 2, []),
    ('de-table-rgbl-1867-403', 'de', 'Seite:Reichsgesetzblatt (Austria) 1867 0403.jpg', ['de', 'fraktur', 'table', 'dot-leaders', 'umlauts'], 2,
     [('prepend', 'LXI. Stück. 146. Gesetz vom 21. December 1867. 403')]),
    ('de-antiqua-kafka-betrachtung-94', 'de', 'Seite:De Kafka Betrachtung (1908) 94.jpg', ['de', 'antiqua', 'umlauts', 'prose'], 2,
     [('replace', 'Wenn ich einem schönen', 'VII Wenn ich einem schönen'), ('replace', 'Denn wir sind wie', 'VIII Denn wir sind wie'),
      ('append', '94')]),
    ('de-mixed-welzheim-188', 'de', 'Seite:Oberamt Welzheim 188.jpg', ['de', 'fraktur', 'latin-antiqua-mix', 'dense', 'umlauts'], 3,
     [('prepend', '188 Ortsbeschreibung.')]),
    ('en-old-statutes-v1-772', 'en', 'Page:United States Statutes at Large Volume 1.djvu/772', ['en', 'degraded', 'form', 'old'], 3, []),
    ('en-prose-constitution-81', 'en', 'Page:The English Constitution (1894).djvu/81', ['en', 'prose', 'clean'], 3, []),
    ('en-2col-eb1911-v25-5', 'en', 'Page:EB1911 - Volume 25.djvu/5', ['en', 'multi-column', 'list'], 3, []),
]
CORD = [(29, 1), (4, 2), (16, 2), (20, 3), (14, 3)]  # (test row, priority)


def edit(gt, edits):
    lines = gt.split('\n')
    for e in edits:
        if e[0] == 'prepend': lines.insert(0, e[1])
        elif e[0] == 'append': lines.append(e[1])
        elif e[0] == 'drop_line':
            lines.remove(e[1])
        elif e[0] == 'insert_before':
            i = next(i for i, l in enumerate(lines) if l.startswith(e[1])); lines.insert(i, e[2])
        elif e[0] == 'replace':
            t = '\n'.join(lines); assert e[1] in t, e; lines = t.replace(e[1], e[2], 1).split('\n')
    return '\n'.join(lines)


def cord_gt(g):
    rows = defaultdict(list)
    for ln in g['valid_line']:
        for w in ln['words']:
            q = w['quad']
            rows[w['row_id']].append(((q['x1'] + q['x4']) / 2, (q['y1'] + q['y2'] + q['y3'] + q['y4']) / 4, w['text']))
    rs = sorted(rows.values(), key=lambda ws: sum(w[1] for w in ws) / len(ws))
    return '\n'.join(' '.join(w[2] for w in sorted(ws)) for ws in rs)


def max_tokens(gt):
    return int(min(4096, max(768, 0.6 * len(gt) + 256)))


def main():
    work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'docs/ocr-real-scans-v1.json'
    pages = []
    for pid, wiki, title, tags, prio, edits in WS:
        r = W.one(str(work / 'ws'), wiki, title)
        assert r['quality'] == 4, (title, r['quality'])            # validated (proofread by two users)
        assert r['copyrighted'] == 'False' and 'public domain' in r['license'].lower(), (title, r['license'])
        gt = edit(r['gt'], edits)
        rev_url = f'https://{wiki}.wikisource.org/w/index.php?oldid={r["revid"]}'
        pages.append(dict(
            id=pid, tags=tags, priority=prio, source='wikisource',
            image=dict(url=r['image_url'], sha256=r['img_sha256'], bytes=r['img_bytes'], width=r['w'], height=r['h'],
                       derived_from=dict(file=r['file'], page=r['page'], file_url=r['file_url'], file_sha1=r['file_sha1'],
                                         host=r['file_host'])),
            image_license=dict(license=r['license'], usage_terms=r['usage'], copyrighted=False,
                               attribution=f'{r["artist"] or "unknown"}; scan: {r["credit"] or r["file_host"]}'),
            ground_truth=gt,
            ground_truth_provenance=dict(
                kind='Wikisource proofread transcription, status Validated (quality 4)', wiki=f'{wiki}.wikisource.org',
                page=title, revision_id=r['revid'], revision_timestamp=r['rev_ts'], revision_url=rev_url,
                derivation='rendered page HTML -> text (tables row-major, cells space-joined; quality banner, hidden '
                           'padding and de.wikisource navigation headings/metadata header removed)',
                manual_edits=[list(e) for e in edits],
                scope=('full page incl. running head/page number' if wiki == 'en' else
                       'full page; running head/page number added by manual edit where printed')),
            ground_truth_license=WS_TEXT_LICENSE,
            max_tokens=max_tokens(gt)))
        print(pid, len(gt), r['img_bytes'], flush=True)
    pq_path = work / 'cord-test.parquet'
    if not pq_path.exists():
        urllib.request.urlretrieve(CORD_URL, pq_path)
    pq_sha = hashlib.sha256(pq_path.read_bytes()).hexdigest()
    import pyarrow.parquet as pq
    rows = pq.read_table(pq_path).to_pylist()
    for row, prio in CORD:
        r = rows[row]; g = json.loads(r['ground_truth']); b = r['image']['bytes']
        gt = cord_gt(g)
        pages.append(dict(
            id=f'receipt-cord-test-{row}', tags=['receipt', 'photo', 'id-en-mixed'], priority=prio, source='cord-v2',
            image=dict(hf_parquet=dict(url=CORD_URL, sha256=pq_sha, row=row, column='image.bytes'),
                       sha256=hashlib.sha256(b).hexdigest(), bytes=len(b),
                       width=g['meta']['image_size']['width'], height=g['meta']['image_size']['height']),
            image_license=dict(license='CC BY 4.0', attribution='CORD: A Consolidated Receipt Dataset for Post-OCR Parsing, '
                               'Park et al., NAVER CLOVA (naver-clova-ix/cord-v2)'),
            ground_truth=gt,
            ground_truth_provenance=dict(
                kind='CORD-v2 valid_line word annotations', dataset='naver-clova-ix/cord-v2', revision=CORD_REV,
                derivation='words grouped by row_id, rows sorted by mean quad y, words by x, space-joined',
                manual_edits=[],
                scope='PARTIAL: only annotated fields (menu/subtotal/total); store header is blurred in the image '
                      'and not annotated, separators/dashes not annotated. Use word recall as the primary metric.'),
            ground_truth_license='CC BY 4.0', max_tokens=max_tokens(gt) if len(gt) > 400 else 1024))
        print(row, len(gt), flush=True)
    pages.sort(key=lambda p: p['priority'])
    doc = dict(
        schema='crispembed-ocr-real-scans-v1',
        description='Real document scans with permissive licences and ground truth for OCR quality measurement. '
                    'Images are fetched at runtime by URL and verified by sha256 (not committed).',
        builder='tools/ocr-real-scans/build_manifest.py',
        normalization='see tools/kaggle/ocr-real-scans/score.py',
        pages=pages,
        rejected_sources=REJECTED)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    print('pages', len(pages), '->', out)


REJECTED = [
    dict(source='FUNSD', reason='licence restricts use to non-commercial research/education; rejected (non-commercial)'),
    dict(source='SROIE (ICDAR 2019)', reason='distributed under ICDAR RRC terms for research; no explicit permissive licence (unclear)'),
    dict(source='DocVQA', reason='RRC registration, research-only terms (non-commercial)'),
    dict(source='Fox benchmark (ucsbnlp/Fox)', reason='no licence stated on the dataset card; page images from third-party documents (unclear)'),
    dict(source='OmniDocBench (opendatalab)', reason='no dataset licence on the Hub card; pages collected from third-party PDFs '
         'with mixed rights (unclear)'),
    dict(source='IAM Handwriting', reason='non-commercial research licence from the FKI; Hub mirrors tagged MIT do not change the '
         'source terms (non-commercial)'),
    dict(source='olmOCR-bench (allenai)', reason='ODC-BY, but the PDFs come from third-party sources with their own rights and the '
         'ground truth is unit-test assertions, not full-page transcriptions (unfit for CER)'),
    dict(source='Project Gutenberg', reason='plain text only; no page scans aligned with the text'),
    dict(source='Wikisource candidates with <ref> footnotes, chess diagrams or broken table markup',
         reason='ground truth would not match the printed page (footnote markers relocated, diagram text); e.g. '
                'Through the Looking-Glass p.11, Federalist (Dawson) p.197, Popular Science Monthly v1 p.71, '
                'Burney Tagebuch p.55, Gartenlaube 1875 p.660, Whymper p.105'),
]

if __name__ == '__main__':
    main()
