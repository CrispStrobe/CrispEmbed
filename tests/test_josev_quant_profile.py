"""Frozen metadata inventory and critical protection; actual C++ stays hosted."""
import json
from pathlib import Path
import re
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Controls(unittest.TestCase):
    def test_frozen_320_shapes_independent_from_mapper(self):
        rows=json.loads((ROOT/'tests/results/josev2b/backbone-f32.json').read_text())['conversion']['tensors']
        expected={r['target']:list(reversed(r['shape']))+[1]*(4-len(r['shape']))for r in rows}
        parsed=re.findall(r'\{\s*"([^"]+)"\s*,\s*\{([^}]+)\}',(ROOT/'tests/josev-quant/backbone-shapes.h').read_text())
        actual={name:[int(v.strip())for v in ne.split(',')]for name,ne in parsed}
        self.assertEqual(len(parsed),320);self.assertEqual(actual,expected)
    def test_no_head_or_token_algorithm_change(self):
        self.assertEqual((ROOT/'src/josev2b_head.cpp').read_bytes(),subprocess.check_output(['git','show','6b5868cef:src/josev2b_head.cpp'],cwd=ROOT))
        old=subprocess.check_output(['git','show','6b5868cef:src/josev2b_text_frontend.cpp'],cwd=ROOT,text=True)
        new=(ROOT/'src/josev2b_text_frontend.cpp').read_text();marker='std::vector<int32_t> text_frontend::tokens('
        self.assertEqual(old[old.index(marker):],new[new.index(marker):])
    def test_f32_component_entrypoints_stay_explicit(self):
        s=(ROOT/'src/josev2b_backbone.cpp').read_text()
        self.assertIn('return map_impl(tensors, weights, false);',s)
        self.assertIn('return run_impl(weights, ids, threads, output, false);',s)
        self.assertIn('core_env::on("CRISPEMBED_JOSEV_QUANT")',s)
        self.assertIn('core_env::on("CRISPEMBED_QWEN35_QUANT")',s)
    def test_metadata_requires_real_q4_and_logical_parameter_inventory(self):
        s=(ROOT/'src/josev2b_packed_profile.h').read_text()
        self.assertIn('allow_quantized = false',s);self.assertIn('mixed && !actual_q4',s)
        self.assertIn('total += elements;',s);self.assertIn('1881825088 + 30472708',s)
        self.assertIn('crispembed-josev-packed-q4k-v1',s)
if __name__=='__main__':unittest.main()
