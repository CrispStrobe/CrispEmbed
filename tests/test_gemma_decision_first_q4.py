#!/usr/bin/env python3
"""Source/stdlib controls; --numeric explicitly enables tiny hosted locked-NumPy controls."""
import ast,copy,hashlib,importlib.util,json,pathlib,struct,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
NUMERIC='--numeric'in sys.argv
if NUMERIC:sys.argv.remove('--numeric')
sys.path[:0]=[str(ROOT/'tools/gemma-decision-q4-v1'),str(ROOT/'tools')]
import gemma_contract as K
import gemma_q4_recipe as R
import gemma_q4_quantizer_fixture as F
spec=importlib.util.spec_from_file_location('gemma_q4_driver',ROOT/'tools/ci-heavy/gemma_decision_first_q4.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def rows():
 result={}
 def add(name,shape):result[name]=dict(type=0,stored_shape=shape,storage_rank=len(shape))
 for name,shape in {'enc.embed_tokens.weight':[512,262144],'enc.ple.per_layer_model_projection.weight':[512,12288],
                   'enc.ple.per_layer_projection_norm.weight':[512],'enc.norm.weight':[512],'enc.embedding_projection.weight':[512,768]}.items():add(name,shape)
 for i in range(24):
  prefix='enc.layers.'+str(i)+'.'
  for name in ['input_layernorm.weight','post_attention_layernorm.weight','pre_feedforward_layernorm.weight','post_feedforward_layernorm.weight','ple_block.post_per_layer_input_norm.weight']:add(prefix+name,[512])
  add(prefix+'layer_scalar',[1])
  for name in ['q','k']:add(prefix+'self_attn.'+name+'_norm.weight',[256])
  for name,shape in [('q_proj',[512,1024]),('k_proj',[512,256]),('v_proj',[512,256]),('o_proj',[1024,512])]:add(prefix+'self_attn.'+name+'.weight',shape)
  for name,shape in [('gate_proj',[512,2048]),('up_proj',[512,2048]),('down_proj',[2048,512])]:add(prefix+'mlp.'+name+'.weight',shape)
  for name in ['per_layer_input_gate','per_layer_projection']:add(prefix+'ple_block.'+name+'.weight',[512,512])
 for name in ['state_proj','question_proj','option_proj']:add('head.'+name+'.weight',[768,256]);add('head.'+name+'.bias',[256])
 for name,cols,n in [('field.0',768,256),('field.3',256,256),('residual.0',1024,256),('residual.3',256,1)]:add('head.'+name+'.weight',[cols,n]);add('head.'+name+'.bias',[n])
 for name in ['field_norm','option_norm']:
  for suffix in ['weight','bias']:add('head.'+name+'.'+suffix,[256])
 add('head.type_emb.weight',[256,3]);add('temperatures',[3])
 for name in ['gate','log_joint_scale','log_prior_scale']:add('head.'+name,[1])
 return result

def fixture():
 result=[]
 for scenario,n in enumerate([13,11,6]):
  nq=3 if scenario<2 else 2;answers={str(q):dict(kind='choice',probabilities={'a':1.},label='a',value='a',expected_level=None)for q in range(nq)}
  result.append(dict(texts=[str(scenario)+' text '+str(i)for i in range(n)],text_embeds=[[0.]*768 for _ in range(n)],answers=answers,
   indices=dict(state_index=[0]*nq,question_index=[1]*nq,option_index=[[2]]*nq,question_type=[0]*nq),logits=[[0.]]*nq,probabilities=[[1.]]*nq))
 return result

class SourceControls(unittest.TestCase):
 def test_source_closure_and_owned_files(self):
  p=K.read(K.PROFILE);K.validate(p)
  self.assertEqual(len(p['protected_git_objects']),len(K.PROTECTED))
  for name,pin in p['protected_git_objects'].items():self.assertEqual(K.git('rev-parse',K.BASE+':'+name),pin)
  for name,pin in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/name),pin,name)
 def test_no_runtime_or_historical_changes(self):
  for name in ['src','CMakeLists.txt','ggml','tools/quantize.cpp','tests/gemma_decision_parity.py','tests/test_gemma_decision_diff.cpp','models/convert-gemma-decision-to-gguf.py']:
   self.assertEqual(K.git('rev-parse',K.BASE+':'+name),K.git('rev-parse','HEAD:'+name))
 def test_immutable_targets_scope_and_no_universal_absolute_gate(self):
  p=K.read(K.PROFILE)
  for key,value in [('stage_max_relative_l2',.004),('absolute_stage_budget',.002),('typed_questions',7),('original_forwards',1),('publication_qualified',True),('encoder_forwards_per_model',89)]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):K.validate(q)
  q=copy.deepcopy(p);q['inputs']['fixtures']['revision']='0'*40
  with self.assertRaises(ValueError):K.validate(q)
 def test_actual_mixed_role_plans_and_critical_groups(self):
  r=rows();self.assertEqual(len(r),436)
  broad=R.plan(r,'broad-mixed');protected=R.plan(r,'protected-input')
  self.assertEqual({k:sum(v==k for v in broad.values())for k in [0,8,12]},{0:218,8:1,12:217})
  self.assertEqual({k:sum(v==k for v in protected.values())for k in [0,12]},{0:268,12:168})
  self.assertEqual(broad['enc.embed_tokens.weight'],8);self.assertEqual(protected['enc.embed_tokens.weight'],0)
  for name in r:
   if 'ple'in name or name.startswith('head.')or 'norm'in name or name=='temperatures'or 'embedding_projection'in name:self.assertEqual(protected[name],0)
   if name.startswith('head.')or 'norm'in name or name=='temperatures'or 'embedding_projection'in name:self.assertEqual(broad[name],0)
 def test_no_fallback_missing_matrix_or_unreviewed_recipe(self):
  for recipe in R.RECIPES:
   r=rows();r['enc.layers.0.mlp.down_proj.weight']['stored_shape'][0]=896
   with self.assertRaises(ValueError):R.plan(r,recipe)
   r=rows();del r['enc.layers.0.mlp.up_proj.weight']
   with self.assertRaises(ValueError):R.plan(r,recipe)
  with self.assertRaises(ValueError):R.plan(rows(),'adaptive-new-recipe')
 def test_complete_inventory_shape_type_and_extra_tensor_rejection(self):
  meta={'general.architecture':dict(value='gemma_decision')}
  for name,value in [('hidden_size',512),('embedding_dim',768),('n_layer',24),('n_head',4),('intermediate_size',2048),('vocab_size',262144),('sliding_window',512),('head_width',256)]:meta['gemma_decision.'+name]=dict(kind=4,value=value)
  p=dict(metadata=meta,tensors=rows())
  with patch.object(R,'parse',return_value=p),patch.object(R,'array',side_effect=[[256]*24,[1]*24]):self.assertEqual(len(R.inventory('synthetic-no-payload',False)['tensors']),436)
  for change in ['shape','type','extra','missing']:
   q=copy.deepcopy(p)
   if change=='shape':q['tensors']['head.state_proj.weight']['stored_shape']=[256,768]
   elif change=='type':q['tensors']['enc.norm.weight']['type']=1
   elif change=='extra':q['tensors']['unreviewed.weight']=dict(type=0,stored_shape=[512,512],storage_rank=2)
   else:del q['tensors']['enc.layers.23.layer_scalar']
   with patch.object(R,'parse',return_value=q),patch.object(R,'array',side_effect=[[256]*24,[1]*24]):
    with self.assertRaises(ValueError):R.inventory('synthetic-no-payload',False)
 def test_fixture_mapping_keeps_all_scenarios_and_rejects_mutation(self):
  original=fixture();self.assertEqual(D.validate_fixture(original),original)
  for change in ['duplicate','index','type','head']:
   f=copy.deepcopy(original)
   if change=='duplicate':f[0]['texts'][0]=f[0]['texts'][1]
   elif change=='index':f[1]['indices']['option_index'][0]=[999]
   elif change=='type':f[2]['indices']['question_type'][0]=2
   else:f[0]['logits'].pop()
   with self.assertRaises(ValueError):D.validate_fixture(f)
 def test_f32_barrier_rejects_false_partial_or_changed_proof(self):
  p=dict(passed=True,source='a'*40,actual_ref_stages=145,typed_questions=8,panel={'sha256':'p'},binaries={'sha256':'b'},source_inputs={'sha256':'i'})
  record=dict(completed=True,strict_f32_passed=True,actual_ref_stages=145,probes=[dict(completed=True,strict_f32_passed=True,stages=[dict(strict_f32_passed=True)for _ in range(29)])for _ in range(5)],typed=dict(completed=True,strict_f32_passed=True,question_count=8,questions=[dict(passed=True)for _ in range(8)]))
  D.admit_candidates(p,'a'*40,{'sha256':'p'},{'sha256':'b'},{'sha256':'i'},record)
  for key,value in [('passed',False),('actual_ref_stages',144),('typed_questions',7),('source','b'*40),('panel',{})]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):D.admit_candidates(q,'a'*40,{'sha256':'p'},{'sha256':'b'},{'sha256':'i'},record)
  broken=copy.deepcopy(record);broken['probes'][0]['stages'].pop()
  with self.assertRaises(ValueError):D.admit_candidates(p,'a'*40,{'sha256':'p'},{'sha256':'b'},{'sha256':'i'},broken)
 def test_small_real_gguf_fixture_parser_without_nn(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'fixture.gguf';names=F.write(p);parsed=R.parse(p)
   self.assertEqual(set(parsed['tensors']),set(names));self.assertEqual(parsed['metadata']['general.architecture']['value'],'gemma_decision')
   self.assertTrue(all(r['type']==0 for r in parsed['tensors'].values()))
 def test_workflow_source_only_then_fixed_f32_barrier_and_remote_retention(self):
  w=(ROOT/'.github/workflows/gemma-decision-first-q4-v1.yml').read_text()
  order=['.py preflight ','.py build ','.py prepare ','.py baseline ','.py candidates ']
  indices=[w.index(s)for s in order];self.assertEqual(indices,sorted(indices))
  self.assertEqual(w.count('if: always()'),2);self.assertIn('retention-days: 90',w);self.assertIn('--require-hashes --no-deps',w)
  c=(ROOT/'tests/gemma-decision-q4/CMakeLists.txt').read_text();self.assertIn('GGML_LLAMAFILE ON',c);self.assertIn('ROOT_ARCH_HELPER',c);self.assertNotIn('src/crispembed.cpp',c)
  cpp=(ROOT/'tests/gemma-decision-q4/probe.cpp').read_text();self.assertIn('reference.compare(name, x, n, 0)',cpp);self.assertNotIn('is_pass(',cpp)
 def test_python_syntax(self):
  for p in (ROOT/'tools/gemma-decision-q4-v1').glob('*.py'):ast.parse(p.read_text())
  ast.parse((ROOT/'tools/ci-heavy/gemma_decision_first_q4.py').read_text())

@unittest.skipUnless(NUMERIC,'Tiny NumPy controls run only after hosted hash-locked install')
class NumericControls(unittest.TestCase):
 def test_independent_zero_rows_fixed_gates_and_magnitudes(self):
  import numpy as np
  self.assertEqual(np.__version__,'2.2.6')
  a=np.array([[0.,0.],[1.,2.]])
  self.assertTrue(D.independent(a,a,[2,2])['passed'])
  b=a.copy();b[1,0]+=.1;self.assertFalse(D.independent(a,b,[2,2])['passed'])
  self.assertEqual(D.independent(a,a,[2,2])['max_absolute'],0.)
 def test_f32_and_actual_quantized_scale_finite_guards(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'span'
   for kind,width in [(0,4),(8,34),(12,144)]:
    good=struct.pack('<f',1.)if kind==0 else struct.pack('<e',1.)+(struct.pack('<e',.5)if kind==12 else b'')+b'\0'*(width-(4 if kind==12 else 2))
    p.write_bytes(good);row=dict(type=kind,data_offset=0,bytes=width);self.assertEqual(R.finite_and_hash(p,row),hashlib.sha256(good).hexdigest())
    bad=bytearray(good);bad[:4 if kind==0 else 2]=struct.pack('<f',float('nan'))if kind==0 else struct.pack('<H',0x7c00);p.write_bytes(bad)
    with self.assertRaises(ValueError):R.finite_and_hash(p,row)

if __name__=='__main__':unittest.main()
