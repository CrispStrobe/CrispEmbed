"""Versioned, model-free OCR quality views. Raw text is never rewritten in place."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'ci-heavy'))
from qwen35_quant_calibration import errors

POLICY = 'crispembed-ocr-quality-v1'


def words(text):
    """Whitespace only: preserve case, accents, punctuation and token order."""
    return ' '.join(text.split())


class Document(HTMLParser):
    """Small, fail-closed HTML fragment grammar; no browser repair or hidden text."""
    containers = {'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    inline = {'b', 'strong', 'i', 'em'}
    table_tags = {'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag not in self.containers | self.inline | self.table_tags | {'br'}:
            raise ValueError('Unsupported HTML tag: ' + tag)
        if attrs and (tag not in {'th', 'td'} or any(k not in {'colspan', 'rowspan'} or v != '1' for k, v in attrs)):
            raise ValueError('Unsupported HTML attributes/spans')
        parent = self.stack[-1] if self.stack else None
        if tag == 'table':
            if self.table is not None or parent is not None:
                raise ValueError('Nested/wrapped table is outside v1 grammar')
            self.table = []
        elif tag in {'thead', 'tbody', 'tfoot'}:
            if parent != 'table':
                raise ValueError('Misplaced table section')
        elif tag == 'tr':
            if parent not in {'table', 'thead', 'tbody', 'tfoot'} or self.row is not None:
                raise ValueError('Misplaced table row')
            self.row = []
        elif tag in {'td', 'th'}:
            if parent != 'tr' or self.cell is not None:
                raise ValueError('Misplaced table cell')
            self.cell = []
        elif self.table is not None and self.cell is None:
            raise ValueError('Formatting outside table cell')
        if tag == 'br':
            self.handle_data('\n')
            return
        if tag in self.containers or tag == 'table':
            self.parts.append('\n')
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if tag != 'br':
            raise ValueError('Unsupported self-closing tag')
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ValueError('Unbalanced HTML')
        if tag in {'td', 'th'}:
            self.row.append(words(''.join(self.cell)))
            self.cell = None
            self.parts.append(' ')
        elif tag == 'tr':
            if not self.row:
                raise ValueError('Empty table row')
            self.table.append(self.row)
            self.row = None
            self.parts.append('\n')
        elif tag == 'table':
            if not self.table or any(len(r) != len(self.table[0]) for r in self.table):
                raise ValueError('Empty/ragged table')
            self.tables.append(self.table)
            self.table = None
            self.parts.append('\n')
        elif tag in self.containers:
            self.parts.append('\n')

    def handle_data(self, data):
        if self.table is not None and self.cell is None:
            if data.strip():
                raise ValueError('Unassigned text inside table')
            return
        self.parts.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_comment(self, data):
        raise ValueError('Comments are outside v1 grammar')

    def handle_decl(self, decl):
        raise ValueError('Declarations are outside v1 grammar')

    def handle_pi(self, data):
        raise ValueError('Processing instructions are outside v1 grammar')

    def unknown_decl(self, data):
        raise ValueError('Unknown declaration')

    def finish(self, text):
        self.feed(text)
        self.close()
        if self.stack:
            raise ValueError('Unclosed HTML')
        return words(''.join(self.parts)), self.tables


def parse_document(text):
    """Recognize ATX headings and simple pipe tables before strict HTML parsing."""
    lines = text.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('```') or line.lstrip().startswith('~~~'):
            raise ValueError('Fenced blocks are outside v1 grammar')
        # Markdown table needs a header, a full separator row, and optional body.
        if i + 1 < len(lines) and line.strip().startswith('|') and lines[i+1].strip().startswith('|'):
            def cells(row):
                if not row.strip().endswith('|') or '\\' in row or '<' in row or '&' in row:
                    raise ValueError('Unsupported pipe table escaping/markup')
                return [c.strip() for c in row.strip()[1:-1].split('|')]
            header, separator = cells(line), cells(lines[i+1])
            if not header or len(header) != len(separator) or any(not re.fullmatch(r':?-{3,}:?', c) for c in separator):
                raise ValueError('Malformed pipe table separator')
            rows = [header]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = cells(lines[i])
                if len(row) != len(header):
                    raise ValueError('Ragged pipe table')
                rows.append(row)
                i += 1
            from html import escape
            output.append('<table>' + ''.join('<tr>' + ''.join('<td>' + escape(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</table>')
            continue
        if line.strip().startswith('|'):
            raise ValueError('Unrecognized pipe table row')
        output.append(re.sub(r'^ {0,3}#{1,6} +', '', line))
        i += 1
    return Document().finish('\n'.join(output))


def score(gold, text, *, complete):
    """Keep incomplete outputs and parsing failures visible; never score a prefix as complete."""
    if type(complete) is not bool or not isinstance(text, str):
        raise ValueError('Explicit completion boolean and actual UTF-8 text required')
    result = dict(policy=POLICY, complete=complete, raw_text=text,
        raw=errors(gold['text'], text), characters=len(text), lines=len(text.splitlines()))
    try:
        content, tables = parse_document(text)
    except (ValueError, AssertionError) as exc:
        result.update(parse_ok=False, parse_error=str(exc), content=None, table=None, exact_quality=False)
        return result
    expected_tables = gold['tables']
    # Position-indexed grids: insertion/deletion cannot be hidden by sorting or matching to gold.
    cell_errors = 0
    cell_total = sum(len(row) for table in expected_tables for row in table)
    for ti in range(max(len(tables), len(expected_tables))):
        want = expected_tables[ti] if ti < len(expected_tables) else []
        got = tables[ti] if ti < len(tables) else []
        for ri in range(max(len(want), len(got))):
            wr = want[ri] if ri < len(want) else []
            gr = got[ri] if ri < len(got) else []
            cell_errors += sum(ci >= len(wr) or ci >= len(gr) or words(wr[ci]) != words(gr[ci])
                for ci in range(max(len(wr), len(gr))))
    expected_content = words(gold['text'])
    result.update(parse_ok=True, parsed_text=content, parsed_tables=tables,
        content=errors(expected_content, content),
        table=dict(expected=len(expected_tables), actual=len(tables), expected_cells=cell_total,
                   cell_errors=cell_errors, exact=tables == expected_tables),
        exact_quality=complete and content == expected_content and tables == expected_tables)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--complete', required=True, choices=('eos', 'cap'))
    args = parser.parse_args()
    result = score(json.loads(args.gold.read_text()), args.output.read_text(), complete=args.complete == 'eos')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['exact_quality'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
