#!/usr/bin/env python3
"""Model-free separate24-Q6 policy plus preserved32/Q8 controls."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_gdn_out_q6_recipe as recipe

def main():
 spec=importlib.util.spec_from_file_location('old_controls',ROOT/'tools/ci-heavy/lighton4b_q6_controls.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 out=Path(os.environ['HEAVY_OUT']);report={'status':'failed','model_payload_bytes':0,'model_inference_executed':False}
 try:
  old.main();build=Path(os.environ['HEAVY_SCRATCH'])/'q6-controls'
  def run(name,args):
   with(out/(name+'.log')).open('w')as f:subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  for name in ('test_lighton4b_gdn_out_q6_recipe.py','test_lighton4b_q6_convert.py'):
   run(name,[sys.executable,ROOT/'tests'/name])
  targets=['test-lighton4b-gdn-out-q6-recipe','test-lighton4b-gdn-out-recipe'];run('build-out-profiles',['cmake','--build',build,'--target',*targets,'--parallel','2'])
  rows=[]
  for r in recipe.plan()['all_tensors']:
   if r['name'].startswith('l.'):
    ne=list(reversed(r['source_shape']));ne += [1]*(4-len(ne));rows.append(' '.join([r['name'],str(r['type']),*map(str,ne)]))
  fixture=out/'out24-schema.tsv';fixture.write_text('\n'.join(rows)+'\n')
  for name in targets:run(name,[build/name]+([fixture]if name.endswith('q6-recipe')else[]))
  report.update(status='passed',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),recipe=recipe.RECIPE,stdout={name:(out/(name+'.log')).read_text()for name in targets})
 finally:(out/'gdn-out-q6-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
