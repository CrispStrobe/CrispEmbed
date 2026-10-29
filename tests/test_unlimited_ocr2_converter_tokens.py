"""Stdlib-only converter metadata controls; no frameworks/tokenizer/model payloads."""
import ast
from copy import deepcopy
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models/convert-unlimited-ocr-to-gguf.py'
tree = ast.parse(SOURCE.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('tokenizer_metadata', 'write_tokenizer_metadata')]
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
metadata, write = namespace['tokenizer_metadata'], namespace['write_tokenizer_metadata']

def added(index, content, **flags):
    return dict(id=index, content=content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True, **flags)

class Writer:
    def __init__(self): self.rows = {}
    def add_array(self, key, value):
        if key in self.rows: raise AssertionError('duplicate writer key')
        self.rows[key] = list(value)

class Controls(unittest.TestCase):
    def test_base_ids_preserved_added_pieces_and_flags_written(self):
        raw = {'model': {'vocab': {'a': 0, 'b': 1}}, 'added_tokens': [added(3, '<PAGE>'), added(2, '<td>')]}
        frozen = deepcopy(raw); writer = Writer()
        self.assertEqual(write(writer, raw, 4), (4, 2)); self.assertEqual(raw, frozen)
        self.assertEqual(writer.rows['tokenizer.ggml.tokens'], ['a', 'b', '<td>', '<PAGE>'])
        self.assertEqual(writer.rows['unlimited_ocr.tokenizer.added_ids'], [2, 3])
        self.assertEqual(writer.rows['unlimited_ocr.tokenizer.added_content'], ['<td>', '<PAGE>'])
        self.assertEqual(writer.rows['unlimited_ocr.tokenizer.added_flags'], [16, 16])
    def test_existing_base_special_id_same_content_and_no_added_legacy(self):
        raw = {'model': {'vocab': {'<s>': 0, 'ä': 1}}, 'added_tokens': [added(0, '<s>')]}
        self.assertEqual(metadata(raw, 2)[0], ['<s>', 'ä'])
        writer = Writer(); self.assertEqual(write(writer, {'model': raw['model']}, 2), (2, 0))
        self.assertEqual(set(writer.rows), {'tokenizer.ggml.tokens'})
    def test_all_831_pinned_fixture_added_ids_retained(self):
        # Source fixture contains only authenticated original pieces, not tokenizer payload.
        text = (ROOT / 'tests/unlimited_ocr2_text_fixture.h').read_text()
        block = text.split('static const piece kAdded[] = {', 1)[1].split('};', 1)[0]
        rows = [(int(i), ast.literal_eval(value).encode('latin1').decode('utf-8')) for i, value in
                re.findall(r'\{\s*(\d+),\s*("(?:\\.|[^"\\])*")\s*\}', block)]
        self.assertEqual([i for i, _ in rows], list(range(128000, 128831)))
        raw = {'model': {'vocab': {'owned-base-placeholder': 0}}, 'added_tokens': [added(i, value) for i, value in rows]}
        tokens, records = metadata(raw, 129280)
        self.assertEqual(len(tokens), 128831); self.assertEqual(len(records), 831)
        for i, value in rows: self.assertEqual(tokens[i], value)
        self.assertEqual(tokens[128821:128825], ['<td>', '</td>', '<tr>', '</tr>'])
        self.assertEqual(tokens[128827:128831], ['<PAGE>', '<BLOCK>', '<FORK>', '<CHILD>'])
    def test_reject_conflicts_duplicates_bounds_and_nonboolean_flags(self):
        good = {'model': {'vocab': {'a': 0}}, 'added_tokens': [added(1, '<PAGE>')]}
        bads = []
        v = deepcopy(good); v['added_tokens'][0]['id'] = 0; bads.append(v)
        v = deepcopy(good); v['added_tokens'].append(v['added_tokens'][0]); bads.append(v)
        for value in (-1, 3, True, 1.0):
            v = deepcopy(good); v['added_tokens'][0]['id'] = value; bads.append(v)
        for value in ('', '\0', '\ud800'):
            v = deepcopy(good); v['added_tokens'][0]['content'] = value; bads.append(v)
        v = deepcopy(good); v['added_tokens'][0]['special'] = 1; bads.append(v)
        v = deepcopy(good); del v['added_tokens'][0]['normalized']; bads.append(v)
        for bad in bads:
            with self.subTest(bad=repr(bad)), self.assertRaises((ValueError, UnicodeEncodeError)): metadata(bad, 2)
    def test_actual_converter_uses_helper_and_preserves_tensor_mapping(self):
        source = SOURCE.read_text(); self.assertIn('write_tokenizer_metadata(writer, tok_data, vocab_size)', source)
        self.assertNotIn('tokens = [""] * len(vocab)', source)
        self.assertIn('writer.add_array("tokenizer.ggml.merges", merges_str)', source)
        self.assertNotIn('numpy', repr([ast.dump(n) for n in functions]))

if __name__ == '__main__': unittest.main()
