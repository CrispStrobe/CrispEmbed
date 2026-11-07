#!/usr/bin/env python3
"""Verify the Wikimedia image pins of an OCR gold corpus (network only, no inference).

For each Wikisource page: the Commons file revision sha1 must equal image.revision.file_sha1, and the
bytes from image.url (then image.fallback_urls) must hash to image.sha256. Matching bytes are saved to
IMG_DIR/<id><ext> for downstream steps. Writes {id: {ok, status, path, tried}} and exits 1 if any page
drifted (the drifted page is reported, never silently repinned).

usage: verify_pins.py CORPUS_JSON IMG_DIR OUT_JSON
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}


def get(url):
    err = None
    for i in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(4 * (i + 1))
    raise err


def main(corpus, img_dir, out):
    m = json.load(open(corpus))
    os.makedirs(img_dir, exist_ok=True)
    res, bad = {}, 0
    for p in m['pages']:
        if p['source'] != 'wikisource':
            continue
        im = p['image']
        r = dict(ok=False, tried=[])
        try:
            info = json.loads(get(im['revision']['api']))
            pg = next(iter(info['query']['pages'].values()))
            sha1 = pg['imageinfo'][0]['sha1']
            if sha1 != im['revision']['file_sha1']:
                r['status'] = f'commons file changed ({sha1})'
            else:
                for url in [im['url']] + im.get('fallback_urls', []):
                    b = get(url)
                    h = hashlib.sha256(b).hexdigest()
                    r['tried'].append(dict(url=url, sha256=h, bytes=len(b)))
                    if h == im['sha256']:
                        ext = os.path.splitext(url.split('?')[0])[1] or '.jpg'
                        r['path'] = os.path.join(img_dir, p['id'] + ext)
                        open(r['path'], 'wb').write(b)
                        r['ok'], r['status'] = True, 'ok' if url == im['url'] else 'ok (fallback url)'
                        break
                else:
                    r['status'] = 'sha256 drift (rendered bytes differ from the pin)'
        except Exception as e:  # noqa: BLE001
            r['status'] = f'error {e!r}'
        res[p['id']] = r
        bad += not r['ok']
        print(p['id'], r['status'], flush=True)
        time.sleep(0.3)
    json.dump(res, open(out, 'w'), indent=1)
    print(f'{len(res) - bad}/{len(res)} Wikimedia pins verified')
    return 1 if bad else 0


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[1:]))
