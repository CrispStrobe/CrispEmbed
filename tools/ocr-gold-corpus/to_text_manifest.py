#!/usr/bin/env python3
"""Render the OCR gold corpus (docs/ocr-gold-corpus-v1.json) as a page-text manifest in the
docs/ocr-real-scans-v2.json shape, so text-only OCR kernels (tools/kaggle/ocr-real-scans) can score
the full corpus.

ground_truth = the page's block texts in reading order, joined by newlines (the same full-page scope
as v2). max_tokens = the v2 page's frozen cap when the gold page reuses a v2 page (v2_id), otherwise
min(4096, len(ground_truth) // 2 + 128), marked `max_tokens_provenance: derived`. Gold text is never
edited.

usage: to_text_manifest.py GOLD_JSON V2_JSON OUT_JSON
"""
import json
import sys


def convert(gold, v2):
    caps = {p['id']: p['max_tokens'] for p in v2['pages']}
    pages = []
    for p in gold['pages']:
        gt = '\n'.join(b['text'] for b in sorted(p['blocks'], key=lambda b: b['order']))
        cap = caps.get(p.get('v2_id'))
        pages.append(dict(id=p['id'], tags=p['tags'], priority=1, source=p['source'], image=p['image'],
                          image_license=p.get('image_license'), ground_truth=gt,
                          ground_truth_license=p.get('ground_truth_license'),
                          max_tokens=cap or min(4096, len(gt) // 2 + 128),
                          max_tokens_provenance='v2 frozen cap' if cap else 'derived: min(4096, len(gt)//2+128)'))
    return dict(schema='crispembed-ocr-gold-text-manifest-v1', derived_from='docs/ocr-gold-corpus-v1.json', pages=pages)


if __name__ == '__main__':
    g, v, o = sys.argv[1:4]
    json.dump(convert(json.load(open(g)), json.load(open(v))), open(o, 'w'), indent=1, ensure_ascii=False)
