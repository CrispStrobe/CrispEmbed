"""Source/independent schema controls only. Native compilation stays hosted."""
from pathlib import Path
import json
import re
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Controls(unittest.TestCase):
    def test_all122_fixture_shapes_come_from_frozen_conversion(self):
        records=json.loads((ROOT/'tests/results/josev2b/head-f32-conversion.json').read_text())['tensors']
        expected={r['target']:list(reversed(r['gguf_shape_numpy']))+[1]*(4-len(r['gguf_shape_numpy'])) for r in records}
        text=(ROOT/'tests/josev-packed/head-shapes.h').read_text()
        rows=re.findall(r'\{\s*"([^"]+)"\s*,\s*\{([^}]+)\}',text)
        actual={name:[int(x.strip()) for x in values.split(',')] for name,values in rows}
        self.assertEqual(len(rows),122);self.assertEqual(actual,expected)
    def test_unchanged_frontend_inference_and_head_graph(self):
        cfg=json.loads((ROOT/'docs/josev2b-packed-runtime.json').read_text())
        old=lambda name:subprocess.check_output(['git','show',cfg['base']+':'+name],cwd=ROOT).decode()
        before=old('src/josev2b_text_frontend.cpp');after=(ROOT/'src/josev2b_text_frontend.cpp').read_text()
        marker='std::vector<int32_t> text_frontend::tokens('
        self.assertEqual(before[before.index(marker):],after[after.index(marker):])
        before=old('src/josev2b_head.cpp');after=(ROOT/'src/josev2b_head.cpp').read_text()
        a,b='struct graph_builder {','bool validate_input('
        self.assertEqual(before[before.index(a):before.index(b)],after[after.index(a):after.index(b)])
    def test_old_composed_guard_remains_closed(self):
        sys.path.insert(0,str(ROOT/'tools'))
        import josev2b_composed_contract
        with self.assertRaisesRegex(ValueError,'native source|owned source'):
            josev2b_composed_contract.verify()
    def test_old_loader_tensor_counts_stay_distinct(self):
        text=(ROOT/'src/josev2b_text_frontend.cpp').read_text()
        self.assertIn('gguf_get_n_tensors(g.get()) == 0',text)
        self.assertIn('josev2b_packed::file_bounds(path, error)',text)
        head=(ROOT/'src/josev2b_head.cpp').read_text()
        self.assertIn('gguf_get_n_tensors(meta) != 122',head)
        self.assertIn('Borrowed head requires host storage',head)
if __name__=='__main__':unittest.main()
