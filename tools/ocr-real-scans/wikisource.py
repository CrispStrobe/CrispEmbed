"""Wikisource Page:/Seite: transcription + Commons scan fetcher used by build_manifest.py.
As a script: wikisource.py OUTDIR WIKI TITLE [WIKI TITLE ...] -> JSON lines."""
import sys, json, re, hashlib, urllib.request, urllib.parse, os, time
from html.parser import HTMLParser
UA = {'User-Agent': 'CrispEmbed-OCR-eval/1.0 (https://github.com/CrispStrobe/CrispEmbed)'}


def get(url):
    for i in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
        except Exception as e:
            err = e; time.sleep(3 * (i + 1))
    raise err


def api(host, **p):
    p['format'] = 'json'; p['formatversion'] = '2'
    data = urllib.parse.urlencode(p).encode()
    for i in range(4):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(f'https://{host}/w/api.php', data=data, headers=UA), timeout=90).read())
        except Exception as e:
            err = e; time.sleep(3 * (i + 1))
    raise err


class T(HTMLParser):
    BLOCK = {'p', 'div', 'br', 'tr', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'dd', 'dt', 'center'}
    VOID = {'br', 'img', 'hr', 'wbr', 'meta', 'link', 'input'}

    def __init__(s): super().__init__(); s.out = []; s.skip = 0; s.cell = 0; s.de = False

    def handle_starttag(s, t, a):
        if t in s.VOID:
            if not s.skip and t == 'br': s.out.append(' ' if s.cell else '\n')
            return
        a = dict(a); cls = (a.get('class') or '').split(); st = (a.get('style') or '').replace(' ', '').lower()
        if 'visibility:hidden' in st or 'display:none' in st or (s.de and t in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')):
            s.skip += 1; return
        if s.skip or t in ('style', 'script') or 'mw-editsection' in cls or 'noprint' in cls or 'mw-cite-backlink' in cls or any(c.startswith('prp-page-quality') or c.startswith('pagequality') for c in cls):
            s.skip += 1; return
        if t in s.BLOCK: s.out.append(' ' if s.cell else '\n')
        if t in ('td', 'th'): s.out.append(' ')
        if t == 'tr': s.cell += 1

    def handle_endtag(s, t):
        if t in s.VOID: return
        if s.skip:
            s.skip -= 1; return
        if t == 'tr': s.cell = max(0, s.cell - 1)
        if t in s.BLOCK: s.out.append(' ' if s.cell else '\n')

    def handle_data(s, d):
        if not s.skip: s.out.append(d.replace('\n', ' ') if s.cell else d)


def text_of(h, de=False):
    p = T(); p.de = de; p.feed(h); t = ''.join(p.out)
    lines = [' '.join(l.split()) for l in t.split('\n')]
    return '\n'.join(l for l in lines if l)


def one(out, wiki, title):
    host = f'{wiki}.wikisource.org'
    q = api(host, action='query', titles=title, prop='proofread|revisions', rvprop='ids|content|timestamp', rvslots='main')
    pg = q['query']['pages'][0]
    rev = pg['revisions'][0]
    # Rendered page incl. header/footer (running heads / page numbers are on the scan too).
    src = rev['slots']['main']['content']
    src = re.sub(r'<pagequality[^>]*/>', '', src)
    if wiki == 'de':  # Seitenstatus2 = wiki metadata header (work/author), not text on the scan
        src = re.sub(r'\{\{Seitenstatus2[^{}]*\}\}', '', src)
    src = src.replace('</noinclude>', '</noinclude>\n')
    par = api(host, action='parse', title=title, text=src, prop='text', contentmodel='wikitext')
    body = text_of(par['parse']['text'], wiki == 'de')
    body = body.split('\nEmpfohlene Zitierweise:')[0]
    m = re.match(r'^[^:]+:(.+?)(?:/(\d+))?$', title)
    fname, n = m.group(1), m.group(2)
    ii = dict(action='query', titles='File:' + fname, prop='imageinfo', iiprop='url|sha1|size|extmetadata|mime', iiurlwidth='1600')
    if n: ii['iiurlparam'] = f'page{n}-1600px'
    fh = 'commons.wikimedia.org'
    fp = api(fh, **ii)['query']['pages'][0]
    if fp.get('missing') or 'imageinfo' not in fp:
        fh = host; fp = api(fh, **ii)['query']['pages'][0]
    inf = fp['imageinfo'][0]; em = inf.get('extmetadata', {})
    url = (inf['thumburl'] if inf.get('thumburl') else inf['url']).split('?')[0]
    img = get(url)
    os.makedirs(out, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9]+', '_', f'{wiki}_{title}')[-90:]
    ext = os.path.splitext(url)[1] or '.jpg'
    open(f'{out}/{safe}{ext}', 'wb').write(img)
    clean = lambda k: re.sub(r'\s+', ' ', re.sub('<[^>]+>', '', em.get(k, {}).get('value', ''))).strip()[:300]
    return dict(wiki=wiki, title=title, quality=pg.get('proofread', {}).get('quality'),
                quality_text=pg.get('proofread', {}).get('quality_text'), revid=rev['revid'], rev_ts=rev['timestamp'],
                file='File:' + fname, file_host=fh, page=n, file_url=inf['url'].split('?')[0], file_sha1=inf['sha1'], image_url=url,
                w=inf.get('thumbwidth') or inf.get('width'), h=inf.get('thumbheight') or inf.get('height'),
                img_bytes=len(img), img_sha256=hashlib.sha256(img).hexdigest(), license=clean('LicenseShortName'),
                usage=clean('UsageTerms'), artist=clean('Artist'), credit=clean('Credit'), copyrighted=clean('Copyrighted'),
                local=f'{out}/{safe}{ext}', gt=body, wikitext=rev['slots']['main']['content'])


if __name__ == '__main__':
    out = sys.argv[1]; a = sys.argv[2:]
    for wiki, title in zip(a[::2], a[1::2]):
        try:
            print(json.dumps(one(out, wiki, title), ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps(dict(wiki=wiki, title=title, error=repr(e))), flush=True)
