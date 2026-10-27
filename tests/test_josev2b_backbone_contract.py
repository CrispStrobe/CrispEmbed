"""Source boundary controls; actual C++/model checks run hosted."""
import json
from pathlib import Path
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
BASE='906a9abb1c71d912bc20472f469456de4bee08b2'


def function(text,name):
    start=text.index('bool '+name+'(');body=text.index('{',start);depth=1;i=body+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[start:i]


class Controls(unittest.TestCase):
    def test_existing_small_profile_mapper_is_byte_identical(self):
        original=subprocess.check_output(['git','show',BASE+':src/qwen35_decoder.cpp'],cwd=ROOT,text=True)
        current=(ROOT/'src/qwen35_decoder.cpp').read_text()
        self.assertEqual(function(current,'map_small_decoder'),function(original,'map_small_decoder'))
        self.assertIn('shape(weights.embedding, 1024, 248320)',function(current,'map_small_decoder'))

    def test_profile_dimensions_match_original_source_contract(self):
        cfg=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text())['backbone']
        self.assertEqual((cfg['hidden_size'],cfg['intermediate_size'],cfg['layers']),(2048,6144,24))
        self.assertEqual(cfg['full_attention_layers'],list(range(3,24,4)))
        self.assertEqual((cfg['linear_key_heads'],cfg['linear_value_heads'],cfg['linear_key_dim'],cfg['linear_value_dim']),(16,16,128,128))
        source=(ROOT/'src/josev2b_backbone.cpp').read_text()
        self.assertIn('tensors.size() != 320',source)
        self.assertIn('t->type == GGML_TYPE_F32',source)
        self.assertIn('core_env::on("CRISPEMBED_JOSEV_BACKBONE")',source)
        self.assertIn('w.head ||',source)

    def test_hidden_route_returns_all_rows_before_vocabulary_projection(self):
        text=(ROOT/'src/qwen35_decoder.cpp').read_text()
        self.assertLess(text.index('if (hidden_only) {'),text.index('std::vector<float> last('))
        self.assertIn('return finite(output.norm) && std::all_of(output.hidden.begin(), output.hidden.end(), finite);',text)
        hidden=function(text,'decoder_hidden_cpu')
        self.assertIn('output = {};',hidden)
        self.assertIn('decoder_outputs temporary;',hidden)
        self.assertIn('output = std::move(temporary)',hidden)
        self.assertIn('nullptr, temporary, nullptr, true',hidden)
        ordinary=function(text,'decoder_cpu')
        self.assertIn('weights_cache, false',ordinary)

    def test_new_component_not_in_public_build_or_registry(self):
        for name in ('CMakeLists.txt','src/crispembed.cpp'):
            self.assertNotIn('josev2b_backbone.cpp',(ROOT/name).read_text())
        cmake=(ROOT/'tests/josev-backbone/CMakeLists.txt').read_text()
        self.assertIn('set(GGML_CUDA OFF',cmake)
        self.assertIn('set(GGML_NATIVE OFF',cmake)
        self.assertIn('metadata.cpp',cmake)


if __name__=='__main__':unittest.main()
