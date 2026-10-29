#!/usr/bin/env python3
"""Authored stdlib/source controls, without NumPy/arrays/models/builds."""
import ast,importlib.util,json,pathlib,tempfile,unittest,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=load('new_compare_protocol','tools/wasm-layer3-f32-reanalysis/protocol.py')
class Controls(unittest.TestCase):
 def test_closed_compare_only_source(self):self.assertEqual(P.source()['expected_fail'],18)
 def test_original_e3_execution_guard_remains_strict(self):
  old=load('immutable_e3_protocol','tools/wasm-layer3-f32/protocol.py')
  with self.assertRaisesRegex(ValueError,'Undeclared source delta'):old.source()
 def test_exact25_pairs_include_all12_unchanged_oldnative_targets(self):
  outputs={b:{k:{'file':'raw/'+b+'/'+k+'.f32'}for k in ['quant_native','quant_candidate','cast_native','cast_candidate']}for b in ['native1','native2','wasm']}
  rows=P.pairs(outputs);self.assertEqual(len(rows),25)
  self.assertEqual([r[0]for r in rows[:4]],['backend.native','threads.native','backend.candidate','threads.candidate'])
  old=[r for r in rows if'.old_native'in r[0]];self.assertEqual(len(old),12)
  self.assertTrue(all(r[1]in ['browser/cast-targets/native1-native.f32','browser/cast-targets/native2-native.f32']for r in old))
  self.assertEqual(sum('.same_input.'in r[0]for r in rows),6)
 def test_zip_positive_and_unsafe_duplicate_symlink_negative(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'authored.zip'
   with zipfile.ZipFile(p,'w')as z:z.writestr('raw/fixture',b'authored')
   self.assertEqual(P.archive(p,100),{'raw/fixture':b'authored'})
   with self.assertRaises(ValueError):P.archive(p,2)
   with zipfile.ZipFile(p,'w')as z:
    z.writestr('raw/duplicate',b'a')
    with self.assertWarns(UserWarning):z.writestr('raw/duplicate',b'b')
   with self.assertRaises(ValueError):P.archive(p,100)
   for name in ['../escape','/absolute','raw\\bad']:
    with zipfile.ZipFile(p,'w')as z:z.writestr(name,b'x')
    with self.assertRaises(ValueError):P.archive(p,100)
   with zipfile.ZipFile(p,'w')as z:
    v=zipfile.ZipInfo('raw/link');v.external_attr=(0o120777<<16);z.writestr(v,b'target')
   with self.assertRaises(ValueError):P.archive(p,100)
 def test_actual_controls_must_all_pass(self):
  keys=['positive','wrong_mode','wrong_extents','nonfinite_input','nonfinite_weight','output_preserved','coefficient_output_preserved','recovery']
  n=dict(threads=[1,2],**{k:True for k in keys});b=dict(completed=True,baseline_completed=True,isolated=True,pthreads=True,page_errors=[],controls={k:True for k in keys})
  def packet():return {'review/native-controls.json':json.dumps(n).encode(),'browser/cast-report.json':json.dumps(b).encode()}
  self.assertTrue(P.controls(packet())['isolated'])
  b['controls']['recovery']=False
  with self.assertRaises(ValueError):P.controls(packet())
 def test_no_operator_or_build_or_f64_reference_coercion(self):
  s=(ROOT/'tools/wasm-layer3-f32-reanalysis/replay.py').read_text();tree=ast.parse(s)
  attrs={n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
  self.assertFalse(attrs&{'CDLL','matmul','dot','command','graph_compute'});self.assertNotIn('np.float64',s);self.assertNotIn('astype(np.float32)',s)
  self.assertIn("dtype='<f4'",s);self.assertIn('actual=old.collect(1,ref,independent)',s);self.assertIn("actual['status']=='numeric_rejection'",s)
 def test_every_raw_member_accounted_and_adapters_lossless(self):
  s=(ROOT/'tools/wasm-layer3-f32-reanalysis/replay.py').read_text()
  self.assertIn("set(raw)==set(expected)|{'control.gguf','replay.gguf'}",s)
  self.assertIn("(out/'control.gguf').read_bytes()==raw['control.gguf']",s)
  self.assertIn("all_raw_members_authenticated=",s);self.assertIn('operator_supported=False',s)
 def test_profile_disambiguates_generic_and_wasm_quants(self):
  s=(ROOT/'tools/wasm-layer3-f32-reanalysis/replay.py').read_text();self.assertIn("split('/ggml-cpu/',1)[1]",s);self.assertIn("endswith('/ggml-cpu/'+name)",s)
 def test_workflow_never_requests_model_browser_compiler(self):
  s=(ROOT/'.github/workflows/wasm-layer3-f32-reanalysis-v1.yml').read_text()
  self.assertIn('timeout-minutes: 15',s);self.assertIn('submodules: false',s);self.assertIn('if: always()',s)
  for token in ['setup-emsdk','setup-node','cmake --build','torch==','playwright']:self.assertNotIn(token,s)
 def test_initial_negative_is_retained_and_unqualified(self):
  x=json.loads(P.INITIAL.read_text());self.assertEqual(x['actual_ref']['pass_count'],7);self.assertEqual(x['actual_ref']['fail_count'],18)
  self.assertFalse(x['operator_supported']or x['qualification']);self.assertGreater(x['worst_saved_ref_row']['max_absolute'],.07)
if __name__=='__main__':unittest.main()
