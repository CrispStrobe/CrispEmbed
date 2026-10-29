"""CPU-provider/source/barrier controls only; no model/package/compiler."""
import ast,copy,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/gemma-decision-q4-cpu-v2'))
import cpu_contract as K
import kernel
spec=importlib.util.spec_from_file_location('new_cpu_driver',ROOT/'tools/ci-heavy/gemma_decision_q4_cpu_v2.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
class Controls(unittest.TestCase):
 def test_exact_predecessor_protected_objects_and_owned_hashes(self):
  p=K.declaration()
  for name,obj in p['protected_git_objects'].items():
   self.assertEqual(K.git('rev-parse',K.PREVIOUS+':'+name),obj,name)
   self.assertEqual(K.git('rev-parse','HEAD:'+name),obj,name)
  for name,pin in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/name),pin,name)
 def test_complete_numerical_body_equivalence(self):
  class MetadataOnly(ast.NodeTransformer):
   def visit_Subscript(self,node):
    if isinstance(node.value,ast.Attribute)and node.value.attr=='environ'and isinstance(node.value.value,ast.Name)and node.value.value.id=='os'and isinstance(node.slice,ast.Constant)and node.slice.value=='GITHUB_SHA':return ast.Name(id='SOURCE',ctx=ast.Load())
    return self.generic_visit(node)
  def functions(path):
   tree=MetadataOnly().visit(ast.parse(path.read_text()))
   return {n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name not in ['source','preflight','main']}
  self.assertEqual(functions(ROOT/'tools/ci-heavy/gemma_decision_first_q4.py'),functions(ROOT/'tools/ci-heavy/gemma_decision_q4_cpu_v2.py'))
 def test_prior_driver_protocol_recipe_probe_are_byte_preserved(self):
  for name in ['tools/ci-heavy/gemma_decision_first_q4.py','docs/gemma-decision-first-q4-v1.json','tools/gemma-decision-q4-v1/gemma_q4_recipe.py','tests/gemma-decision-q4/probe.cpp','tests/gemma-decision-q4/CMakeLists.txt']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',K.PREVIOUS+':'+name],cwd=ROOT))
 def test_only_genuine_cpu_provider_with_explicit_source(self):
  source='a'*40
  with patch.object(Path,'is_dir',return_value=True),patch.dict(os.environ,dict(KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source),clear=True):K.provider(source)
  for values in [dict(GITHUB_ACTIONS='true',KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source),dict(GITHUB_SHA=source,KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source),dict(KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE='b'*40),dict(CRISPEMBED_CAPTURE_SOURCE=source)]:
   with patch.object(Path,'is_dir',return_value=True),patch.dict(os.environ,values,clear=True):
    with self.assertRaises(ValueError):K.provider(source)
  with patch.object(Path,'is_dir',return_value=False),patch.dict(os.environ,dict(KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source),clear=True):
   with self.assertRaises(ValueError):K.provider(source)
 def test_complete_numeric_negative_is_not_structural_success(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);K.write(root/'review/f32-panel.json',dict(completed=True));K.write(root/'review/f32-barrier.json',dict(passed=False))
   result=kernel.result_scope(root,'baseline',1);self.assertTrue(result['completed']);self.assertFalse(result['numerical_passed']);self.assertFalse(result['candidates_executed']);self.assertEqual(result['encoder_forward_maximum'],88)
   K.write(root/'review/f32-panel.json',dict(completed=False));self.assertIsNone(kernel.result_scope(root,'baseline',1))
 def test_both_fixed_candidate_numeric_negatives_remain_rejections(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);r=dict(completed=True,selected_recipe=None,publication_qualified=False,recipes=[dict(recipe=k,panel=dict(strict_f32_passed=False))for k in ['broad-mixed','protected-input']]);K.write(root/'review/result.json',r)
   result=kernel.result_scope(root,'candidates',1);self.assertTrue(result['completed']);self.assertFalse(result['numerical_passed']);self.assertEqual(result['encoder_forward_maximum'],264)
   with self.assertRaises(ValueError):kernel.result_scope(root,'candidates',0)
   r['recipes'].pop();K.write(root/'review/result.json',r)
   with self.assertRaises(ValueError):kernel.result_scope(root,'candidates',1)
 def test_actual_isolated_help_has_no_optional_imports(self):
  for path in ['tools/ci-heavy/gemma_decision_q4_cpu_v2.py','tools/gemma-decision-q4-cpu-v2/kernel.py']:
   result=subprocess.run([sys.executable,'-I',str(ROOT/path),'--help'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--reviewed-source',result.stdout)
  for name in ['numpy','torch','transformers','gguf','safetensors']:self.assertNotIn(name,sys.modules)
 def test_same_resource_bounds_and_honest_explicit_metadata(self):
  p=K.declaration();self.assertEqual(p['resources']['minimum_disk_bytes'],16*2**30);self.assertEqual(p['resources']['minimum_ram_bytes'],8*2**30)
  text=(ROOT/'tools/ci-heavy/gemma_decision_q4_cpu_v2.py').read_text();self.assertNotIn('GITHUB_',text);self.assertNotIn('old.source(',text)
  for forbidden in ["os.environ['GITHUB_ACTIONS']=",'old.PROFILE=','old.ROOT=']:
   self.assertNotIn(forbidden,(ROOT/'tools/gemma-decision-q4-cpu-v2/cpu_contract.py').read_text())
if __name__=='__main__':unittest.main()
