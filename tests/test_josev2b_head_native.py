#!/usr/bin/env python3
"""Small frozen-metadata/source controls only; no arrays, builds or frameworks."""
import copy
import json
import re
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_head_native as n
from josev2b_head_schema import tensor_schema
class Controls(unittest.TestCase):
    def setUp(self):self.contract=json.loads(n.CONTRACT.read_text())
    def test_exact_original_shapes_and_order_for_all_four_cases(self):
        self.assertEqual(len(self.contract['cases']),4)
        for case in self.contract['cases']:
            self.assertEqual(n.stage_plan(n.validate_manifest(case['input'])),case['original_stages'])
    def events(self):
        case=self.contract['cases'][0];events=[]
        for index,row in enumerate(case['original_stages']):
            events.append(dict(**row,file=f'{index:05d}.f32',dtype='float32',elements=__import__('math').prod(row['shape'])))
        return case['input'],events
    def test_duplicate_occurrences_cannot_be_overwritten_or_reordered(self):
        manifest,events=self.events();n.validate_events(manifest,events)
        matches=[i for i,r in enumerate(events) if r['stage']=='head/option_context_projection']
        self.assertEqual([events[i]['occurrence'] for i in matches],[0,1,2])
        for bad in ('dictionary','reorder','shape','occurrence'):
            rows=copy.deepcopy(events)
            if bad=='dictionary':rows=list({r['stage']:r for r in rows}.values())
            elif bad=='reorder':rows[matches[0]],rows[matches[1]]=rows[matches[1]],rows[matches[0]]
            elif bad=='shape':rows[0]['shape']=[319,2048]
            else:rows[matches[1]]['occurrence']=0
            with self.assertRaises(ValueError):n.validate_events(manifest,rows)
    def test_empty_span_wrong_type_shape_and_extra_properties_reject(self):
        original=self.contract['cases'][0]['input']
        for kind in ('empty','type','tokens','fields','options','path','extra'):
            m=copy.deepcopy(original)
            if kind=='empty':m['fields'][0]['question_end']=m['fields'][0]['question_start']
            elif kind=='type':m['fields'][0]['type']=3
            elif kind=='tokens':m['tokens']=513
            elif kind=='fields':m['fields']*=6
            elif kind=='options':m['fields'][0]['options']*=65
            elif kind=='path':m['hidden_f32']='../hidden.f32'
            else:m['temperature']=0.5
            with self.assertRaises(ValueError):n.validate_manifest(m)
    def test_pretty_integer_fixture_and_json_whitespace_only_source(self):
        source=(ROOT/'tests/test_josev2b_head_probe.cpp').read_text()
        fixture=re.search(r'fields\(R"JSON\((.*?)\)JSON"\)',source,re.S).group(1)
        value=json.loads(fixture);n.validate_manifest(value)
        self.assertEqual(value['fields'][0]['options'][0]['end'],466)
        self.assertIn('find_first_not_of(" \\t\\r\\n")',source)
        self.assertIn('find_last_not_of(" \\t\\r\\n")',source)
        self.assertNotIn('std::isspace',source)
        self.assertIn('Malformed/out-of-bound integer accepted',source)
        self.assertIn('integer(pretty_option.at("end")) == 466',source)

    def test_schema_inventory_and_fixed_original_source(self):
        schema=tensor_schema();self.assertEqual(len(schema),122);self.assertEqual(sum(__import__('math').prod(s) for s in schema.values()),30472708)
        c=self.contract;self.assertEqual(c['original_receipt_sha256'],'51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090');self.assertEqual(c['ggml'],'890278a8342c620197c90e702e1188bcab94f510')
        self.assertFalse(c['qualification']);self.assertFalse(c['full_model'])
    def test_observer_only_in_testing_and_connected_views_keep_owners(self):
        src=(ROOT/'src/josev2b_head.cpp').read_text();header=(ROOT/'src/josev2b_head.h').read_text()
        self.assertIn('#ifdef CRISPEMBED_JOSEV_HEAD_TESTING',src);self.assertNotIn('run_observed',header)
        self.assertIn('ggml_view_4d(ctx,tensor',re.sub(r'\s+','',src));self.assertIn('owner=owner->view_src',re.sub(r'\s+','',src))
        self.assertNotIn('ggml_view_tensor(ctx',src);self.assertIn('occurrences[label]++',src)
    def test_prenorm_erf_f32_and_no_fallback_source(self):
        src=re.sub(r'\s+','',(ROOT/'src/josev2b_head.cpp').read_text());self.assertIn('ggml_mul_mat_set_prec(out,GGML_PREC_F32)',src)
        self.assertIn('ggml_gelu_erf',src);self.assertIn('norm(name+".norm1",fields)',src);self.assertIn('norm(name+".norm2",x)',src);self.assertIn('norm(name+".norm3",y)',src)
        self.assertIn('ggml_dup(ctx,w(name))',src);self.assertIn('1e-12f',src);self.assertIn('1e-8f',src)
        self.assertIn('core_env::on("CRISPEMBED_JOSEV_HEAD")',src);self.assertNotIn('ggml_backend_init_best',src)
    def test_failure_outputs_clear_before_gates_and_observer_publishes_last(self):
        src=(ROOT/'src/josev2b_head.cpp').read_text()
        ordinary=src[src.index('bool run(context *'):src.index('#ifdef CRISPEMBED_JOSEV_HEAD_TESTING',src.index('bool run(context *'))]
        observed=src[src.index('bool run_observed('):src.index('bool metadata_controls(')]
        for body in (ordinary,observed):self.assertLess(body.index('result = {};'),body.index('if (!gate(error))'))
        self.assertIn('graph.compute(pending, error)',observed)
        self.assertLess(observed.index('callback(out);'),observed.index('result = std::move(pending);'))
        probe=(ROOT/'tests/test_josev2b_head_probe.cpp').read_text()
        self.assertIn('Observer exception published a computed result',probe)
        self.assertIn('exact(ordinary, recovered)',probe)
        self.assertIn('gate_failure_output_empty',probe)

    def test_probe_requires_exact_neutrality_and_clean_exit(self):
        src=(ROOT/'tests/test_josev2b_head_probe.cpp').read_text()
        self.assertIn('exact(ordinary,observed)&&exact(ordinary,repeat)',re.sub(r'\s+','',src));self.assertIn('core_util::clean_exit',src)
        self.assertIn('Duplicate JSON member',src);self.assertIn('"--contracts"',src);self.assertIn('"--gates"',src)
if __name__=='__main__':unittest.main()
