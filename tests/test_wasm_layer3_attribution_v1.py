#!/usr/bin/env python3
"""Small source/parser controls only. No build, framework or model access."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from wasm_layer3_attribution_v1 import *


def negative():
    failures=[c+'/1 final gates' for c in ('original','empty','punctuation-currency','mixed-scripts','whitespace','numbers','near-limit','truncating')]
    for thread in ('1','2'):
        failures+=['original/1/stage/'+thread+'/'+s for s in ('layer_3','layer_4','encoder_out')]
        failures+=['original/1/recomputed-mean/'+thread]
    return {'schema':'crispembed-wasm-q8-rounding-ab-result-v1','source':BASE,'state':'completed-rejected',
            'qualification':False,'passed':False,'failures':failures,
            'cases':{str(i):{g:{'ordinary_dump_bit_exact':True,'legacy_bit_exact':g!='1'} for g in ('unset','empty','0','1')} for i in range(10)},
            'first_sentence_stages':{g:{t:[{'stage':s,'old_wasm_bit_exact':True} for s in STAGES] for t in ('1','2')} for g in ('unset','empty','0')}}


class Layer3Tests(unittest.TestCase):
    def test_actual_frozen_encoder_overlay_and_owner_retention(self):
        text=subprocess.check_output(['git','show',BASE+':src/crispembed.cpp'],cwd=ROOT,text=True)
        self.assertIn('#include <cmath>',text)
        result=overlay_encoder(text)
        self.assertEqual(len(NAMES),23)
        for name in NAMES:self.assertIn('names.push_back("'+name+'")',result)
        self.assertIn('owner = owner->view_src',result)
        self.assertIn('ggml_build_forward_expand(gf, view)',result)
        self.assertIn('ggml_view_4d(gctx, tensor,',result)
        self.assertIn('view->op == GGML_OP_VIEW && view->src[0] == tensor',result)
        self.assertIn('view->nb[axis] == tensor->nb[axis]',result)
        self.assertNotIn('ggml_view_tensor(gctx, tensor)',result)
        self.assertIn('core_env::on("CRISPEMBED_LAYER3_TRACE")',result)
        self.assertNotIn('ggml_add_inplace',result[len(text):])
        with self.assertRaisesRegex(ValueError,'encoder bytes'):
            overlay_encoder(text+'\n')

    def test_negative_provenance_cannot_be_promoted_or_waived(self):
        report=negative();trusted_ab(report)
        for key,value in [('source','0'*40),('state','completed-supported'),('passed',True),('qualification',True),('failures',[])]:
            bad=copy.deepcopy(report);bad[key]=value
            with self.assertRaises(ValueError):trusted_ab(bad)
        bad=copy.deepcopy(report);bad['cases']['0']['0']['legacy_bit_exact']=False
        with self.assertRaisesRegex(ValueError,'legacy'):trusted_ab(bad)
        bad=copy.deepcopy(report);bad['first_sentence_stages']['0']['1'].pop()
        with self.assertRaisesRegex(ValueError,'stage'):trusted_ab(bad)

    def test_actual_workspace_q8_0_scales_and_signed_bytes(self):
        data=(struct.pack('<e',.5)+bytes([127,128]+[0]*30))*12
        rows=conversion_blocks(data,8,32)
        self.assertEqual(len(rows),12)
        self.assertEqual(rows[0][0]['scale'],.5)
        self.assertEqual((rows[0][0]['signed_min'],rows[0][0]['signed_max']),(-128,127))
        with self.assertRaises(ValueError):conversion_blocks(data[:-1],8,32)
        with self.assertRaises(ValueError):conversion_blocks(data,20,32)

    def test_actual_workspace_q8_k_preserves_sum_fields(self):
        block=struct.pack('<f',.125)+bytes([1]*256)+struct.pack('<16h',*([16]*16))
        rows=conversion_blocks(block*12,15,256)
        self.assertEqual(rows[0][0]['scale'],.125)
        self.assertEqual(rows[0][0]['sum_bytes_sha256'],digest(block[-32:]))
        bad=struct.pack('<f',float('nan'))+block[4:]
        with self.assertRaisesRegex(ValueError,'scale'):conversion_blocks(bad*12,15,256)

    def test_projection_row_gate_rejects_one_bad_row(self):
        self.assertTrue(row_metrics([1.,0.,0.,1.],[1.,0.,0.,1.],2)['gate_pass'])
        self.assertFalse(row_metrics([1.,0.,0.,1.],[1.,0.,1.,0.],2)['gate_pass'])
        with self.assertRaises(ValueError):row_metrics([1.],[1.],2)

    def test_actual_wasm_compile_flags_and_coverage(self):
        names=['src/crispembed.cpp','ggml/src/ggml-cpu/ggml-cpu.c',
               'ggml/src/ggml-cpu/arch/wasm/quants.c','wasm/embed_wrapper.c']
        rows=[{'file':'/source/'+name,'arguments':['/emsdk/em++' if name.endswith('.cpp') else '/emsdk/emcc','-pthread','-c','/source/'+name]+(['-msimd128'] if name.startswith('ggml/') else [])} for name in names]
        self.assertTrue(check_wasm_commands(rows)['passed'])
        for flag in ('-msimd128','-pthread'):
            bad=copy.deepcopy(rows);bad[2]['arguments'].remove(flag)
            with self.assertRaisesRegex(ValueError,'compile flags'):check_wasm_commands(bad)
        bad=copy.deepcopy(rows);bad[1]['arguments'].append('-mno-simd128')
        with self.assertRaisesRegex(ValueError,'compile flags'):check_wasm_commands(bad)
        with self.assertRaisesRegex(ValueError,'coverage'):check_wasm_commands(rows[:-1])
        bad=copy.deepcopy(rows);bad[0]['arguments'].append('-msimd128')
        with self.assertRaisesRegex(ValueError,'compile flags'):check_wasm_commands(bad)

    def test_browser_threaded_module_requires_isolated_origin(self):
        browser=(ROOT/'tests/test_wasm_layer3_attribution_v1.cjs').read_text()
        self.assertIn("res.setHeader('Cross-Origin-Opener-Policy','same-origin')",browser)
        self.assertIn("res.setHeader('Cross-Origin-Embedder-Policy','require-corp')",browser)
        self.assertLess(browser.index("globalThis.crossOriginIsolated!==true"),browser.index('await window.CrispEmbedText'))
        self.assertIn("typeof SharedArrayBuffer!=='function'",browser)

    def test_native_replay_export_requires_unmangled_dynamic_symbol(self):
        self.assertTrue(check_native_export('00001234 T crispembed_layer3_replay\n')['passed'])
        for symbols in ('00001234 T _Z_crispembed_layer3_replay\n','00001234 t crispembed_layer3_replay\n',''):
            with self.assertRaisesRegex(ValueError,'export'):check_native_export(symbols)
        replay=(ROOT/'tests/wasm-layer3/replay.inc').read_text()
        self.assertIn('extern "C" CRISPEMBED_API int crispembed_layer3_replay',replay)

    def test_source_probes_are_overlay_only(self):
        kernel=(ROOT/'tests/wasm-layer3/kernel_capture.inc').read_text()
        self.assertIn('const void * converted = params->wdata;',kernel)
        self.assertNotIn('quantize_row_q8_0_wasm_nearest(',kernel)
        replay=(ROOT/'tests/wasm-layer3/replay.inc').read_text()
        self.assertIn('ggml_mul_mat(graph_context, weight, input)',replay)
        self.assertIn('sched_graph_compute(ctx->sched',replay)
        self.assertIn('ggml_backend_sched_reset(ctx->sched)',replay)
        self.assertEqual(len(WEIGHTS),6)
        self.assertEqual(WEIGHTS['blk.3.ffn_down.weight'],('gelu','fc2_raw'))

    def test_protected_diagnostic_closure(self):
        spec=json.loads((ROOT/'docs/wasm-layer3-attribution-v1.json').read_text())
        self.assertEqual(spec['ab']['review'],AB_REVIEW)
        self.assertEqual(spec['ab']['raw'],AB_RAW)
        self.assertEqual(spec['gates'],{'max_abs':.002,'min_row_cosine':.99999})
        required={'tools/wasm_layer3_attribution_v1.py','tools/ci-heavy/wasm_layer3_attribution_v1.py',
                  'tests/wasm-layer3/kernel_capture.inc','tests/wasm-layer3/replay.inc',
                  'tests/test_wasm_layer3_attribution_v1.cjs','tests/test_wasm_layer3_diff.cpp',
                  'tools/successor_wasm_trace_contract.py','tools/ci-heavy/successor_assets.py',
                  'tools/successor_asset_manifest.py'}
        self.assertEqual(set(spec['protected_sources']),required)
        for name,value in spec['protected_sources'].items():self.assertEqual(digest((ROOT/name).read_bytes()),value)
        workflow=(ROOT/'.github/workflows/wasm-layer3-attribution-v1.yml').read_text()
        self.assertIn('fetch-depth: 0',workflow)
        self.assertIn('timeout-minutes: 60',workflow)
        self.assertIn('if: always()',workflow)
        driver=(ROOT/'tools/ci-heavy/wasm_layer3_attribution_v1.py').read_text()
        self.assertIn("'--clean','--no-simd'",driver)


if __name__=='__main__':unittest.main()
