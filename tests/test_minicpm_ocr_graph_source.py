"""Light source contracts; actual C++ component and model execution stay hosted."""
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def compact(s):
    return re.sub(r'\s+', '', s)


class Controls(unittest.TestCase):
    def test_norm_rope_and_attention_math_preserved(self):
        old = (ROOT/'src/qwen35_attention.cpp').read_text()
        new = (ROOT/'src/qwen35_graph_decoder.cpp').read_text()
        # Exact helper bodies preserve source-level rounding/order, not just
        # a transcription of the mathematical formula.
        for begin, end in [('static void norm_head(', 'static void rotate('),
                           ('static void rotate(', 'bool attention_cpu(')]:
            body = old[old.index(begin):old.index(end)]
            self.assertIn(compact(body), compact(new))
        begin = '    auto & cache = output.cache;'
        end = '    projected.clear();'
        self.assertIn(compact(old[old.index(begin):old.index(end)]), compact(new))

    def test_small_linear_core_called_without_reimplementation(self):
        s = (ROOT/'src/qwen35_graph_decoder.cpp').read_text()
        self.assertIn('linear_core_cpu(input, output.core)', s)
        self.assertNotIn('ggml_gated_delta_net(', s)
        self.assertNotIn('project_cpu(', s)
        self.assertIn('ggml_mul_mat(ctx.get(), weight, input)', s)
        self.assertNotIn('ggml_cast(', s)
        self.assertIn('ggml_backend_buffer_is_host(weight->buffer)', s)
        self.assertIn('ggml_set_output(result)', s)

    def test_only_minicpm_routes_and_request_snapshot(self):
        s = (ROOT/'src/minicpm_ocr.cpp').read_text()
        self.assertEqual(s.count('CRISPEMBED_MINICPM_OCR_GRAPH'), 1)
        self.assertIn('if (graph) return c.graph_decoder.run', s)
        self.assertIn('return qwen35_ocr::decoder_cpu(', s)
        self.assertLess(s.index('graph_decoder.clear();'), s.index('core_gguf::free_weights(loaded)'))
        self.assertIn('decode(hidden.data(), positions.data(), int(ids.size()), nullptr)', s)
        self.assertIn('decode(hidden.data(), positions.data(), 1, &cache)', s)

    def test_release_component_checks_and_owned_storage(self):
        s = (ROOT/'tests/minicpm-graph/controls.cpp').read_text()
        self.assertNotIn('assert(', s)
        self.assertIn('core_util::clean_exit(', s)
        self.assertIn('ggml_backend_alloc_ctx_tensors', s)
        for term in ('GGML_TYPE_F32', 'GGML_TYPE_F16', 'GGML_TYPE_Q8_0', 'GGML_TYPE_Q4_K',
                     'borrowed weights remain byte exact', 'token bound before input read',
                     'same-context repeat exact', 'separate backend context', 'disable after success'):
            self.assertIn(term, s)

    def test_q8_publisher_preserves_4d_patch(self):
        s = json.loads((ROOT/'docs/minicpm-ocr-publication-v1.json').read_text())
        q = s['recipes']['Q8_0']
        self.assertEqual(q['tensor_type'], ['v.emb.patch_embedding.weight=f16'])
        self.assertEqual(q['precision_receipt'], 'tests/results/minicpm-ocr/bench.json')
        self.assertEqual(s['parity_cases'], ['line', 'para-en', 'table', 'tall', 'tiny'])


if __name__ == '__main__':
    unittest.main()
