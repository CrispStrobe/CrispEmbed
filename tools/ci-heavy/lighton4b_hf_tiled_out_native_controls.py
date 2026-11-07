#!/usr/bin/env python3
"""No-checkpoint native owner/layout/cache controls and immutable old-path checks."""
import json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_q4km_recipe as recipe

def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_BUILD']);started=time.monotonic();r={'schema':'lighton4b-hf-tiled-out-native-controls/v1','status':'failed','model_payload_bytes':0,'model_forward_executed':False,'checks':{}}
 def run(label,args):
  print('PHASE '+label,flush=True)
  with(out/(label+'.log')).open('w')as log:child=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=240)
  if child.returncode:raise ValueError('native layout control failed: '+label)
  text=(out/(label+'.log')).read_text();m=re.search(r'checks=(\d+)',text);r['checks'][label]=int(m.group(1))if m else None
  if ('pass=1'not in text and 'passed=1'not in text)or(not m and label!='old-stream-projection'):raise ValueError('native layout proof marker absent: '+label)
 try:
  lines=[]
  for row in recipe.plan()['all_tensors']:
   if row['name'].startswith('l.'):
    shape=list(reversed(row['source_shape']));shape+=[1]*(4-len(shape));lines.append(' '.join([row['name'],str(row['type']),*map(str,shape)]))
  schema=out/'closed426.tsv';schema.write_text('\n'.join(lines)+'\n')
  run('hf-tiled-recipe',[build/'test-lighton4b-hf-tiled-out-recipe',schema]);run('hf-tiled-fullwidth-operator',[build/'test-lighton4b-hf-tiled-out-operator'])
  run('old-hf-imatrix-recipe',[build/'test-lighton4b-imatrix-q4km-recipe',schema]);run('old-physical-modes',[build/'test-lighton4b-physical-native-ops']);run('old-q6-tied-head',[build/'test-lighton4b-q6-tied-head']);run('old-stream-projection',[build/'test-qwen35-stream-projection']);run('public-release-fence',[build/'test-lighton4b-release-fence',out/'public-header-only.gguf']);r['status']='passed'
 finally:r['seconds']=time.monotonic()-started;(out/'native-controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
