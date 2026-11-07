#!/usr/bin/env python3
"""One Kaggle CPU model-free union: stats, closed recipe, old policies, Q6 head, no ship."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_q4km_recipe as recipe
import lighton4b_q6_recipe as old
import lighton4b_gdn_out_q6_recipe as out6

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 if not(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir())or list(Path('/dev').glob('nvidia[0-9]*')):raise ValueError('Kaggle CPU only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);report={'schema':'lighton4b-imatrix-q4km-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model_payload_bytes':0,'weight_quantization_executed':False,'model_forward_executed':False,'quality_qualified':False,'current_q4_shipping_approved':False}
 def phase(label):report['stage']=label;print('PHASE '+label,flush=True)
 def run(label,args,expect=0):
  phase(label);r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=1200);(out/(label+'.log')).write_text(r.stdout+r.stderr)
  if r.returncode!=expect:raise ValueError('control failed: '+label)
  return r.stdout
 saved=dict(os.environ)
 try:
  phase('adapter-actual-means-and-legacy-controls');adapter=load('q4km_adapter',ROOT/'tools/ci-heavy/lighton4b_imatrix_adapter_controls.py');aout=out/'adapter';aout.mkdir();os.environ['HEAVY_OUT']=str(aout);adapter.main();os.environ['HEAVY_OUT']=str(out);build=scratch/'lighton-imatrix/build'
  for module in('test_lighton4b_imatrix_q4km_recipe.py','test_lighton4b_q6_recipe.py','test_lighton4b_gdn_out_q6_recipe.py'):run(module,[sys.executable,ROOT/'tests'/module])
  # Verify old frozen receipt blobs and the explicit new-source amendment chain.
  pins=[json.loads((ROOT/('tests/results/qwen35-ocr/quality-quant-'+p+'-inventory-table.json')).read_text())['source_commit']for p in('q8','q8-q6','q4')]
  run('fetch-frozen-receipt-commits',['git','fetch','--depth','1','origin',*sorted(set(pins))]);run('source-pin-check',[sys.executable,ROOT/'tools/qwen35_source_pins.py','--check','--verify-git'])
  targets=['test-lighton4b-imatrix-q4km-recipe','test-lighton4b-q6-policy','test-lighton4b-gdn-out-q6-recipe','test-lighton4b-gdn-out-recipe']
  run('build-policy-controls',['cmake','--build',build,'--target',*targets,'--parallel','2'])
  for label,p,target in [('new-q4km',recipe.plan(),targets[0]),('old32-q8head',old.plan(),targets[1]),('old24-q6out',out6.plan(),targets[2])]:
   lines=[]
   for row in p['all_tensors']:
    if row['name'].startswith('l.'):
     shape=list(reversed(row['source_shape']));shape+=[1]*(4-len(shape));lines.append(' '.join([row['name'],str(row['type']),*map(str,shape)]))
   schema=out/(label+'.tsv');schema.write_text('\n'.join(lines)+'\n');run(label,[build/target,schema])
  run('old24-q8out',[build/targets[3]])
  # Closed ordinary production CLI must reject all paths without weight work.
  binary=build/'crispembed-quantize';header=scratch/'lighton-imatrix/target-header-only.gguf';adapted=aout/'adapted-imatrix.gguf';forbidden=scratch/'must-not-publish-q4km.gguf'
  tests=[('no-stats',[header,forbidden,'q4_k','--lighton4b-imatrix-q4km-style']),('wrong-method',[header,forbidden,'q8_0','--lighton4b-imatrix-q4km-style','--imatrix',adapted]),('override',[header,forbidden,'q4_k','--lighton4b-imatrix-q4km-style','--imatrix',adapted,'--tensor-type','l.embed_tokens.weight=q8_0']),('legacy-header',[header,forbidden,'q4_k','--lighton4b-imatrix-q4km-style','--imatrix',adapted,'--legacy-file-type-header']),('wrong-original-source',recipe.quantizer_arguments(header,forbidden,adapted))]
  for label,args in tests:
   run('closed-cli-'+label,[binary,*args],1)
   if forbidden.exists()or Path(str(forbidden)+'.partial').exists():raise ValueError('rejected profile published file')
  # Same build/cache, independent actual Q6 operators and public no-ship fence.
  phase('q6-tied-head-and-public-fence');release=load('q4km_release',ROOT/'tools/kaggle/lighton4b_release_q6_head_controls.py');rout=out/'release';rout.mkdir();os.environ['HEAVY_OUT']=str(rout);os.environ['HEAVY_EXISTING_BUILD']=str(build);release.main();os.environ['HEAVY_OUT']=str(out)
  report.update(status='passed',all248_actual_means_verified=True,new_recipe=recipe.plan()['recipe'],q6_names_sha256=recipe.plan()['q6_names_sha256'],old_recipes_preserved=True,closed_source_and_override_rejections=5,q6_embedding_tied_head_and_public_fence_passed=True)
 except Exception as e:report['error_type']=type(e).__name__;report['error']=str(e)if isinstance(e,ValueError)else'bounded model-free controls failed';raise
 finally:
  os.environ.clear();os.environ.update(saved);(out/'imatrix-q4km-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
