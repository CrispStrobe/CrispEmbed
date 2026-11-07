"""Derive layout blocks (type, text, reading order, table structure) from a Wikisource
ProofreadPage revision. No inference, no network: input is the page wikitext and the
MediaWiki-rendered HTML of that wikitext with section sentinels inserted.

A ProofreadPage revision is ``<noinclude>HEADER</noinclude>BODY<noinclude>FOOTER</noinclude>``.
``sectioned_wikitext`` inserts a sentinel paragraph between the three sections so a single
``action=parse`` call keeps footnote numbering intact; ``blocks_from_html`` then walks the
rendered top-level elements in document order:

  section header                -> header      (running heads, page numbers, {{rh}} rows)
  section footer, non-reference -> footer
  ol.references > li            -> footnote    (wherever <references/> rendered them)
  h1..h6                        -> title       (rule "heading-tag")
  short centred block (<=120 ch) -> title      (rule "centred-short"; heuristic)
  table with >=2 rows and >=2 cols -> table    (cells: row, col, rowspan, colspan, text)
  ul / ol / dl                  -> list        (items)
  img / figure / [typeof~=mw:File] -> figure   (caption text, if any)
  div.poem / p / other text     -> text

Reading order is document order of the proofread transcription (header, body, footnotes,
footer), which is how Wikisource proofreaders transcribe a page. Every block records the
rule that typed it, so derived annotations stay distinguishable from manual ones.
Inline footnote markers (``[1]`` rendered by <ref>) are removed from block text and kept in
``markers``: the printed glyph (``¹``, ``(1)``, ``*``) is not recoverable from wikitext.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

SENTINEL = 'CRISPGOLDSECTIONBREAK'
VOID = {'br', 'img', 'hr', 'wbr', 'meta', 'link', 'input', 'source', 'col'}


def split_sections(wikitext: str) -> tuple[str, str, str]:
    """Return (header, body, footer) of a ProofreadPage revision."""
    m = re.match(r'^\s*<noinclude>(.*?)</noinclude>(.*)$', wikitext, re.S)
    header, rest = (m.group(1), m.group(2)) if m else ('', wikitext)
    m = re.match(r'^(.*)<noinclude>(.*?)</noinclude>\s*$', rest, re.S)
    body, footer = (m.group(1), m.group(2)) if m else (rest, '')
    header = re.sub(r'<pagequality[^>]*/>', '', header)
    header = re.sub(r'\{\{Seitenstatus2[^{}]*\}\}', '', header)  # de: wiki metadata, not printed
    return header.strip(), body.strip(), footer.strip()


def sectioned_wikitext(wikitext: str) -> str:
    h, b, f = split_sections(wikitext)
    return f'{h}\n\n{SENTINEL}\n\n{b}\n\n{SENTINEL}\n\n{f}\n'


class Node:
    __slots__ = ('tag', 'attrs', 'children', 'parent')

    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.children, self.parent = tag, dict(attrs or {}), [], parent

    def cls(self):
        return (self.attrs.get('class') or '').split()

    def style(self):
        return (self.attrs.get('style') or '').replace(' ', '').lower()


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node('root')
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def _hidden(n: Node) -> bool:
    c, st = n.cls(), n.style()
    return (n.tag in ('style', 'script') or 'display:none' in st or 'visibility:hidden' in st
            or 'mw-editsection' in c or 'noprint' in c or 'mw-cite-backlink' in c
            or any(x.startswith(('prp-page-quality', 'pagequality')) for x in c))


def _is_marker(n: Node) -> bool:
    return n.tag == 'sup' and 'reference' in n.cls()


BLOCKISH = {'p', 'div', 'table', 'ul', 'ol', 'dl', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'center', 'blockquote',
            'figure', 'pre', 'hr'}


def text_of(n, markers=None) -> str:
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
            return
        if _hidden(x):
            return
        if _is_marker(x):
            if markers is not None:
                markers.append(' '.join(''.join(_raw(x)).split()).strip('[]'))
            return
        if x.tag == 'br':
            out.append('\n')
            return
        if x.tag in ('td', 'th'):
            out.append(' ')
        if x.tag in BLOCKISH or x.tag in ('tr', 'li', 'dd', 'dt'):
            out.append('\n')
        for c in x.children:
            walk(c)
        if x.tag in BLOCKISH or x.tag in ('tr', 'li', 'dd', 'dt'):
            out.append('\n')

    walk(n)
    lines = [' '.join(l.replace('\xa0', ' ').split()) for l in ''.join(out).split('\n')]
    return '\n'.join(l for l in lines if l)


def _raw(n):
    for c in n.children:
        if isinstance(c, str):
            yield c
        else:
            yield from _raw(c)


def _has_block_child(n: Node) -> bool:
    return any(isinstance(c, Node) and not _hidden(c) and (c.tag in BLOCKISH) for c in n.children)


def _centred(n: Node) -> bool:
    st, c = n.style(), n.cls()
    return n.tag == 'center' or 'text-align:center' in st or any('center' in x for x in c)


def _figure(n: Node) -> bool:
    if n.tag in ('img', 'figure') or 'mw:File' in (n.attrs.get('typeof') or '') or 'thumb' in n.cls():
        return True
    return False


def _contains_figure(n: Node) -> bool:
    if _figure(n):
        return True
    return any(isinstance(c, Node) and _contains_figure(c) for c in n.children)


def table_cells(t: Node) -> tuple[list[dict], int, int]:
    """Cells of an HTML table on a grid (rowspan/colspan honoured)."""
    rows = []

    def collect(x):
        for c in x.children:
            if isinstance(c, Node):
                if c.tag == 'tr':
                    rows.append(c)
                elif c.tag in ('thead', 'tbody', 'tfoot'):
                    collect(c)

    collect(t)
    taken, cells = set(), []
    for r, tr in enumerate(rows):
        col = 0
        for td in (c for c in tr.children if isinstance(c, Node) and c.tag in ('td', 'th') and not _hidden(c)):
            while (r, col) in taken:
                col += 1
            rs = max(1, int(re.sub(r'\D', '', td.attrs.get('rowspan') or '1') or 1))
            cs = max(1, int(re.sub(r'\D', '', td.attrs.get('colspan') or '1') or 1))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, col + dc))
            cells.append(dict(row=r, col=col, rowspan=rs, colspan=cs, header=td.tag == 'th',
                              text=' '.join(text_of(td).split())))
            col += cs
    ncols = 1 + max((c['col'] + c['colspan'] - 1 for c in cells), default=-1)
    return cells, len(rows), ncols


def blocks_from_html(html: str, wiki: str = 'en') -> list[dict]:
    tree = _Tree()
    tree.feed(html)
    root = tree.root
    # descend into mw-parser-output
    def find(n):
        for c in n.children:
            if isinstance(c, Node):
                if 'mw-parser-output' in c.cls():
                    return c
                f = find(c)
                if f:
                    return f
        return None
    root = find(root) or root
    blocks: list[dict] = []
    state = {'section': 'header'}

    def emit(btype, rule, text, **extra):
        text = text.strip()
        if not text and btype not in ('figure', 'table'):
            return
        sec = state['section']
        if btype != 'footnote':
            if sec == 'header':
                btype, rule = 'header', 'section-header'
            elif sec == 'footer':
                btype, rule = 'footer', 'section-footer'
        b = dict(type=btype, text=text, type_provenance=f'derived:wikisource-html:{rule}', section=sec)
        b.update(extra)
        blocks.append(b)

    def para(n, rule_default='paragraph'):
        markers = []
        t = text_of(n, markers)
        if not t:
            return
        if t == SENTINEL:
            state['section'] = 'body' if state['section'] == 'header' else 'footer'
            return
        if SENTINEL in t:  # sentinel merged into a paragraph: split around it
            parts = t.split(SENTINEL)
            for i, p in enumerate(parts):
                if i:
                    state['section'] = 'body' if state['section'] == 'header' else 'footer'
                if p.strip():
                    emit('text', rule_default, p)
            return
        kids = [c for c in n.children if isinstance(c, Node) and c.tag != 'br']
        loose = ''.join(c for c in n.children if isinstance(c, str)).strip()
        centred = _centred(n) or (len(kids) == 1 and _centred(kids[0]) and not loose)
        extra = dict(markers=markers) if markers else {}
        if centred and len(t) <= 120:
            emit('title', 'centred-short', t, **extra)
        else:
            emit('text', rule_default, t, **extra)

    def walk(n):
        for c in n.children:
            if isinstance(c, str):
                if c.strip():
                    if c.strip() == SENTINEL:
                        state['section'] = 'body' if state['section'] == 'header' else 'footer'
                    else:
                        emit('text', 'bare-text', c)
                continue
            if _hidden(c) or c.tag == 'hr':
                continue
            if c.tag == 'ol' and 'references' in c.cls():
                for li in (x for x in c.children if isinstance(x, Node) and x.tag == 'li'):
                    emit('footnote', 'references-li', text_of(li), note_id=li.attrs.get('id', ''))
                continue
            if c.tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                if wiki == 'de':  # de.wikisource renders navigation headings; printed titles use templates
                    continue
                emit('title', 'heading-tag', text_of(c))
                continue
            if c.tag == 'table':
                cells, nr, nc = table_cells(c)
                if nr >= 2 and nc >= 2:
                    emit('table', 'table-grid', text_of(c), table=dict(rows=nr, cols=nc, cells=cells))
                elif _has_block_child(c) or nr >= 1:
                    for cell in cells:
                        if cell['text']:
                            emit('text', 'single-axis-table', cell['text'])
                continue
            if c.tag in ('ul', 'ol', 'dl'):
                items = [text_of(li) for li in c.children if isinstance(li, Node) and li.tag in ('li', 'dd', 'dt')]
                items = [i for i in items if i]
                if items:
                    emit('list', 'list-tag', '\n'.join(items), items=items)
                continue
            if _figure(c) or (_contains_figure(c) and len(text_of(c)) < 200 and c.tag != 'p'):
                emit('figure', 'file-embed', text_of(c))
                continue
            if 'poem' in c.cls():
                emit('text', 'poem', text_of(c))
                continue
            if c.tag in ('div', 'center', 'blockquote', 'span') and _has_block_child(c):
                if _centred(c):
                    # centred container: children inherit centring
                    for g in c.children:
                        if isinstance(g, Node) and g.tag in ('p', 'div') and not _has_block_child(g):
                            g.attrs['style'] = (g.attrs.get('style') or '') + ';text-align:center'
                walk(c)
                continue
            if c.tag == 'p' and _contains_figure(c) and not text_of(c):
                emit('figure', 'file-embed', '')
                continue
            para(c)

    walk(root)
    # de.wikisource footer templates append a citation box that is not printed on the scan
    blocks[:] = [b for b in blocks if not b['text'].startswith('Empfohlene Zitierweise')]
    for b in blocks:  # a long note typed into the footer section is a footnote, not a running foot
        if b['type'] == 'footer' and len(b['text']) > 40:
            b['type'], b['type_provenance'] = 'footnote', 'derived:wikisource-html:section-footer-long-note'
    body = [b for b in blocks if b['section'] == 'body' and b['type'] != 'footnote']
    if body and re.fullmatch(r'[a-zäöüßàâçéèêëîïôûœſ][\w’\'-]{0,14}', body[-1]['text']) and len(blocks) > 1:
        body[-1]['type'] = 'footer'  # catchword: first word of the next page printed at the foot
        body[-1]['type_provenance'] = 'derived:wikisource-html:catchword'
    for i, b in enumerate(blocks):
        b['order'] = i
        b['id'] = f'b{i}'
    return blocks
