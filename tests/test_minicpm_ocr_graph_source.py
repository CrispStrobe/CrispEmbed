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
        body = old[old.index('static void rotate('):old.index('bool attention_cpu(')]
        self.assertIn(compact(body), compact(new))
        # The qwen35 head norm gained a default-off LightOn4B physical branch
        # (lighton4b_physical_native_component / lighton4b_upstream_gamma). Its
        # default path must stay the exact arithmetic MiniCPM's graph copy keeps.
        norm = old[old.index('static bool norm_head('):old.index('static void rotate(')]
        default = norm[norm.index('const float scale = rms_scale_cpu(x, width, eps);'):]
        self.assertIn(compact('const float scale = rms_scale_cpu(x, width, eps);'
                              'for (int i = 0; i < width; ++i) y[i] = (x[i] * scale) * '
                              'lighton4b_upstream_gamma(weights[i], physical);'), compact(default))
        gamma = (ROOT/'src/lighton4b_upstream_physical_semantics.h').read_text()
        self.assertIn(compact('return physical ? value : (1.0f + value);'), compact(gamma))
        self.assertIn(compact('static void norm_head(const float * x, const float * weights, int width, float eps, float * y) {'
                              'const float scale = rms_scale_cpu(x, width, eps);'
                              'for (int i = 0; i < width; ++i) y[i] = (x[i] * scale) * (1.0f + weights[i]);}'),
                      compact(new))
        begin = '    auto & cache = output.cache;'
        end = '    projected.clear();'
        block = old[old.index(begin):old.index(end)]
        # Drop only the default-off LightOn4B physical-profile cache fields.
        # Default-off physical F16 attention wraps the original pooled loop as
        # `if (physical_f16) {...} else {<original>}`; compare the else branch.
        if '    if (physical_f16) {' in block:
            head, rest = block.split('    if (physical_f16) {', 1)
            original = rest.split('    } else {', 1)[1].rsplit('    }', 1)[0]
            block = head + original
        block = '\n'.join(l for l in block.splitlines() if 'physical' not in l and 'repack' not in l)
        self.assertIn(compact(block), compact(new))

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
        # Current-tree contract (not a frozen guard). Since the 2026-11-02 default flip
        # (48fcd6f5) the stateless gate is read twice: once to route, once so the
        # default-on CRISPEMBED_QWEN35_GRAPH_DECODER yields to it (exclusive gates).
        # Still only these two env reads, both in this file.
        self.assertEqual(s.count('core_env::on("CRISPEMBED_MINICPM_OCR_GRAPH")'), 2)
        self.assertEqual(s.count('getenv("CRISPEMBED_MINICPM_OCR_GRAPH")'), 0)
        self.assertIn('are exclusive', s)
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
        # Main's qualified recipe quantizes from the F16 GGUF, whose 4-D patch
        # kernel is not flattened (only F32 4-D tensors are), so no override is needed.
        self.assertEqual(q['from'], 'F16')
        self.assertEqual(q['tensor_type'], [])
        self.assertEqual(q['precision_receipt'], 'tests/results/minicpm-ocr/bench.json')
        self.assertEqual(s['parity_cases'], ['line', 'para-en', 'table', 'tall', 'tiny'])


if __name__ == '__main__':
    unittest.main()
