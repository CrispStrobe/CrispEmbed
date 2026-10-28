#!/usr/bin/env python3
"""Helpers for the crispembed-unlimited-ocr2-trial Kaggle kernel.

  render <outdir>               synthetic, license-clean pages + ground truth (pages.json)
  reference <pages.json> <out> <deadline_epoch> [ids...]
                                original PyTorch Unlimited-OCR-2 (tools/unlimited-ocr2/reference.py
                                loading path) on the given page ids; writes <out> after every page.

All page text is original to this file (no third-party content). Hosted-only, like reference.py.
"""
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PARA = [
    'Quarterly Operations Report',
    '',
    'The warehouse in Leipzig completed its move to the new conveyor system in',
    'September. Throughput rose from 1,840 to 2,315 parcels per hour, while the',
    'error rate on outgoing labels fell below 0.4 percent for the first time.',
    '',
    'Two issues remain open. The cold storage unit on level 2 still reports',
    'temperature spikes during the night shift, and the loading dock scanner',
    'needs a firmware update before the holiday season begins on 1 December.',
]
TABLE_HEAD = ['Product', 'Region', 'Units', 'Revenue']
TABLE_ROWS = [['Notebook A5', 'North', '1,250', '8,750.00'], ['Desk Lamp', 'South', '430', '12,040.50'],
              ['Office Chair', 'East', '96', '17,184.00'], ['Monitor 27"', 'West', '212', '42,188.00'],
              ['Total', '', '1,988', '80,162.50']]
RECEIPT = ['CORNER BAKERY', 'Hauptstrasse 12, 10115 Berlin', 'Receipt No. 004817', '2026-10-28 08:42',
           '2x Croissant        3.80', '1x Rye Bread        4.20', '1x Cappuccino       3.10',
           '1x Orange Juice     2.90', 'Subtotal           14.00', 'VAT 7%              0.98',
           'TOTAL              14.98', 'Paid by card', 'Thank you!']
GERMAN = [
    'Mitteilung an alle Mitarbeiterinnen und Mitarbeiter',
    '',
    'Ab Montag gelten neue Öffnungszeiten für die Kantine im Erdgeschoß.',
    'Frühstück gibt es von 7:30 bis 9:00 Uhr, Mittagessen ab 11:45 Uhr.',
    'Bitte beachten Sie: Übergrößen bei Arbeitsschuhen müssen bis Freitag',
    'bestellt werden. Für Rückfragen wenden Sie sich an Frau Müller.',
    '',
    'Grüße aus der Geschäftsführung, Jörg Weiß',
]
COL_L = ['Local News', '', 'The city council approved', 'a plan to extend the tram', 'line by four stations.',
         'Construction starts in', 'spring and should finish', 'within two years.']
COL_R = ['Weather', '', 'Sunny in the morning with', 'clouds later in the day.', 'Highs near 18 degrees,',
         'light wind from the west.', 'Rain is expected again', 'on Thursday evening.']
A4 = [
    'Annual Maintenance Plan 2027',
    '',
    'This document describes the scheduled maintenance work for the main office',
    'building. All tasks are planned outside regular working hours whenever',
    'possible, so that disruption for staff and visitors stays minimal.',
    '',
    '1. Heating and ventilation',
    'The central heating boiler will be inspected in January. Filters in all',
    'ventilation units are replaced every three months, starting in February.',
    'The cooling system on the roof receives a full service in late April.',
    '',
    '2. Electrical systems',
    'Emergency lighting is tested monthly. A complete inspection of all',
    'distribution boards takes place in June and requires a short power cut',
    'on a Saturday morning between 06:00 and 10:00.',
    '',
    '3. Elevators and access',
    'Both elevators are serviced in March and September. During the service',
    'only one elevator is available. The access card readers at the main',
    'entrance get a software update in the second quarter.',
    '',
    '4. Outdoor areas',
    'The parking lot markings are renewed in May. Trees along the north side',
    'are trimmed in November after the leaves have fallen. Snow clearing is',
    'handled by an external contractor from December to March.',
    '',
    'Questions about this plan can be sent to the facilities team at any time.',
    'Please report urgent defects directly by phone, extension 4410.',
]


def font(size, mono=False):
    pats = (['/usr/share/fonts/**/DejaVuSansMono.ttf', '/usr/**/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf']
            if mono else
            ['/usr/share/fonts/**/DejaVuSans.ttf', '/usr/**/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',
             '/usr/share/fonts/**/LiberationSans-Regular.ttf'])
    from PIL import ImageFont
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return ImageFont.truetype(hits[0], size), hits[0]
    return ImageFont.load_default(size=size), 'pillow-default'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def text_page(lines, size, fs, x=60, y=60, lh=None, mono=False):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', size, 'white'); d = ImageDraw.Draw(img)
    f, fp = font(fs, mono)
    lh = lh or int(fs * 1.5)
    for i, ln in enumerate(lines):
        d.text((x, y + lh * i), ln, fill='black', font=f)
    return img, fp


def render(outdir):
    from PIL import Image, ImageDraw, ImageFilter
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pages = []

    def add(pid, img, gt, note, fp):
        p = out / f'{pid}.png'; img.save(p, format='PNG')
        pages.append(dict(id=pid, path=str(p), size=list(img.size), ground_truth=gt, note=note, font=fp,
                          sha256=sha(p)))

    img, fp = text_page(PARA, (1024, 768), 24, x=50, y=60)
    add('paragraphs', img, '\n'.join(l for l in PARA if l), 'multi-paragraph English, 1024x768', fp)

    # table with ruled grid
    img = Image.new('RGB', (1024, 768), 'white'); d = ImageDraw.Draw(img); f, fp = font(26)
    xs = [60, 330, 520, 700, 940]; y0, rh = 80, 70
    rows = [TABLE_HEAD] + TABLE_ROWS
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            d.text((xs[c] + 12, y0 + r * rh + 20), cell, fill='black', font=f)
    for r in range(len(rows) + 1):
        d.line([(xs[0], y0 + r * rh), (xs[-1], y0 + r * rh)], fill='black', width=2)
    for x in xs:
        d.line([(x, y0), (x, y0 + len(rows) * rh)], fill='black', width=2)
    add('table', img, '\n'.join(' '.join(c for c in row if c) for row in rows), 'ruled 4x6 table, 1024x768', fp)

    img, fp = text_page(RECEIPT, (520, 900), 22, x=30, y=40, lh=58, mono=True)
    add('receipt', img, '\n'.join(RECEIPT), 'narrow receipt (monospace), 520x900', fp)

    img, fp = text_page(GERMAN, (1024, 768), 24, x=50, y=70)
    add('german', img, '\n'.join(l for l in GERMAN if l), 'German with umlauts/eszett, 1024x768', fp)

    img = Image.new('RGB', (1240, 900), 'white'); d = ImageDraw.Draw(img); f, fp = font(26)
    for i, ln in enumerate(COL_L):
        d.text((60, 80 + 46 * i), ln, fill='black', font=f)
    for i, ln in enumerate(COL_R):
        d.text((660, 80 + 46 * i), ln, fill='black', font=f)
    d.line([(630, 70), (630, 480)], fill='gray', width=1)
    add('two-column', img, '\n'.join(l for l in COL_L + COL_R if l), 'two columns, reading order L then R, 1240x900', fp)

    img, fp = text_page(A4, (1654, 2339), 30, x=140, y=160, lh=56)
    add('a4-dense', img, '\n'.join(l for l in A4 if l), 'A4 at 200 dpi (1654x2339), long text: tiling-relevant', fp)

    # scan simulation: the German page at A4-ish resolution, slight skew, blur, noise, JPEG round-trip
    import io
    base, fp = text_page(GERMAN, (1240, 1754), 34, x=110, y=150, lh=60)
    rnd = random.Random(1234)
    base = base.rotate(0.8, resample=Image.BICUBIC, expand=False, fillcolor='white').filter(ImageFilter.GaussianBlur(0.7))
    px = base.load()
    for _ in range(25000):
        x, y = rnd.randrange(base.size[0]), rnd.randrange(base.size[1]); v = rnd.randrange(150, 256)
        px[x, y] = (v, v, v)
    buf = io.BytesIO(); base.convert('L').save(buf, format='JPEG', quality=55); buf.seek(0)
    add('scan-sim', Image.open(buf).convert('RGB'), '\n'.join(l for l in GERMAN if l),
        'simulated scan of the German page: 0.8 deg skew, blur, speckle, JPEG q55, 1240x1754', fp)
    (out / 'pages.json').write_text(json.dumps(pages, indent=2, ensure_ascii=False))
    print(json.dumps([(p['id'], p['size'], p['font']) for p in pages]))


def reference(pages_json, out_json, deadline, ids):
    sys.path.insert(0, str(ROOT / 'tools/unlimited-ocr2'))
    import reference as R
    pages = [p for p in json.load(open(pages_json)) if not ids or p['id'] in ids]
    cfg = json.loads(R.CONTRACT.read_text())
    rep = dict(schema='crispembed-unlimited-ocr2-trial-reference-v1', revision=cfg['revision'], cases=[], completed=False)
    save = lambda: Path(out_json).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    model_dir = Path(out_json).parent / 'checkpoint'
    t0 = time.time(); R.fetch(cfg, model_dir); rep['fetch_seconds'] = round(time.time() - t0, 1); save()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.Tensor.cuda = lambda self, *a, **k: self
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, use_safetensors=True,
                                      torch_dtype=torch.float32).eval()
    rep['load_seconds'] = round(time.time() - t0, 1)
    mod = sys.modules[type(model).__module__]
    build = mod._uocr_build_inputs

    def cast(x):
        if isinstance(x, torch.Tensor): return x.float() if x.dtype == torch.bfloat16 else x
        if isinstance(x, (list, tuple)): return type(x)(cast(v) for v in x)
        return x
    mod._uocr_build_inputs = lambda *a, **k: cast(build(*a, **k))
    for p in pages:
        if time.time() > deadline:
            rep['cases'].append(dict(id=p['id'], skipped='time budget')); save(); continue
        t0 = time.time()
        try:
            with torch.inference_mode():
                res = model.infer(tok, image_file=p['path'], eval_mode=True, return_dict=True,
                                  max_new_tokens=int(os.environ.get('TRIAL_REF_MAX_TOKENS', '1024')), temperature=0.0)
            row = dict(id=p['id'], seconds=round(time.time() - t0, 1), text=res.get('text'))
            for k in ('ids', 'token_ids', 'output_ids'):
                if k in res and res[k] is not None:
                    v = res[k]; row['ids'] = v.tolist() if hasattr(v, 'tolist') else list(v); break
        except Exception as e:  # keep going; the error is part of the receipt
            row = dict(id=p['id'], seconds=round(time.time() - t0, 1), error=repr(e)[:2000])
        rep['cases'].append(row); save(); print('reference', p['id'], row.get('seconds'), flush=True)
    rep['completed'] = True; save()


if __name__ == '__main__':
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        sys.exit('hosted-only (Kaggle / GitHub Actions)')
    cmd = sys.argv[1]
    if cmd == 'render':
        render(sys.argv[2])
    elif cmd == 'reference':
        reference(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5:])
    else:
        sys.exit(__doc__)
