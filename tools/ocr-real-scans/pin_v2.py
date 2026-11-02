#!/usr/bin/env python3
"""Derive docs/ocr-real-scans-v2.json from v1: pin every Wikimedia image to its exact file revision.

No inference. Network only (Commons API + one download per page). v1 pinned the bytes by sha256 but
fetched the *current* thumbnail URL, and Wikimedia served the census-form page (en-form-economic-journal-67)
as two different JPEG byte streams across sessions. v2 adds, per Wikimedia page:

  image.revision   {timestamp, file_sha1, api}  - the Commons file revision the bytes were rendered from
                                                  (checked against the live API by the kernel)
  image.url        revision-qualified URL: the v1 URL + '?<YYYYMMDDhhmmss>' (the MediaWiki convention for a
                                           revision-specific file/thumbnail link; it also bypasses stale
                                           edge-cache objects keyed on the bare URL)
  image.fallback_urls [v1 URL]             tried only if the pinned URL's bytes fail the sha256 check

The sha256 stays the gate: a page whose bytes do not match is reported as failed and is not scored.
Each page is downloaded here and must hash to the v1 sha256, otherwise this script refuses to write v2.

usage: pin_v2.py [IN_JSON] [OUT_JSON]
"""
import hashlib, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}
API = 'https://commons.wikimedia.org/w/api.php'


def get(url):
    for i in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read()
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(5 * (i + 1))
    raise err


def revision(file_title):
    q = API + '?' + urllib.parse.urlencode(dict(action='query', titles=file_title, prop='imageinfo',
                                                 iiprop='timestamp|sha1', iilimit=1, format='json'))
    pg = next(iter(json.loads(get(q))['query']['pages'].values()))
    ii = pg['imageinfo'][0]
    return dict(timestamp=ii['timestamp'], file_sha1=ii['sha1'], api=q)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs/ocr-real-scans-v1.json'
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'docs/ocr-real-scans-v2.json'
    m = json.loads(src.read_text())
    m['schema'] = 'crispembed-ocr-real-scans-v2'
    m['description'] += (' v2: every Wikimedia image is pinned to its Commons file revision (timestamp + file sha1, '
                         'checked against the live API) and fetched from a revision-qualified URL; the kernel fails a '
                         'page loudly on any sha256 mismatch instead of running it.')
    m['builder'] = 'tools/ocr-real-scans/build_manifest.py (v1) + tools/ocr-real-scans/pin_v2.py'
    m['derived_from'] = src.name
    for p in m['pages']:
        im = p['image']
        if 'url' not in im:
            continue
        df = im['derived_from']
        rev = revision(df['file'])
        if rev['file_sha1'] != df['file_sha1']:
            raise SystemExit(f"{p['id']}: Commons file changed since v1 ({rev['file_sha1']} != {df['file_sha1']})")
        ts = rev['timestamp'].replace('-', '').replace(':', '').replace('T', '').replace('Z', '')
        bare = im['url'].split('?', 1)[0]
        pinned = f'{bare}?{ts}'
        h = hashlib.sha256(get(pinned)).hexdigest()
        if h != im['sha256']:
            raise SystemExit(f"{p['id']}: {pinned} -> {h}, manifest sha256 {im['sha256']}")
        im['url'] = pinned
        im['fallback_urls'] = [bare]
        im['revision'] = rev
        print(p['id'], ts, 'ok', flush=True)
    dst.write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')
    print('wrote', dst)


if __name__ == '__main__':
    main()
