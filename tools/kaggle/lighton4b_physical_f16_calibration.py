#!/usr/bin/env python3
"""Two frozen source-feature cases, exact completed form binary; no qualification."""
import hashlib,importlib.util,io,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load('full_form',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');require=f.require;sha=f.sha
FORM='0a0c2084baa4e46aa80f09f7ae94fd4b1c56ccae2bbf2f9ef6210e81e8cf1d01'
PHYSICAL='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'
CASE_IDS=['simple_table','arabic_printed_line']
def admit(cfg):
 require(cfg['schema']=='lighton4b-physical-f16-calibration/v1'and cfg['case_ids']==CASE_IDS,'closed two-case calibration')
 require(cfg['form_receipt_sha256']==FORM and cfg['physical_receipt_sha256']==PHYSICAL,'immutable receipts')
 require(cfg['cap']==256 and cfg['threads']==4 and cfg['per_case_timeout_seconds']==1800 and cfg['total_timeout_seconds']==3600,'bounded execution')

def main():
 import numpy as np
 from PIL import Image
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
 cfgpath=ROOT/'docs/lighton4b-physical-f16-calibration-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg)
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'calibration';work.mkdir();started=time.monotonic();deadline=Deadline(3600);deadline.__enter__()
 r={'schema':cfg['schema'],'status':'failed','source_sha':None,'cases':{},'commands':{},'quality_qualified':False,'source_feature_fed':True,'model_f32_instantiated':False}
 def save():(out/'physical-f16-calibration.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,timeout=600,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(timeout,3600-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'exact mounted receipt '+name);return rows[0]
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'artifact '+str(p.name))
 try:
  run('source-pin',['git','rev-parse','HEAD']);r['source_sha']=(out/'source-pin.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  base=load('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(work).free>=25<<30 and shutil.disk_usage(out).free>=4<<30,'guarded RAM/disk before acquisition')
  fp=locate('physical-f16-full-form.json',FORM);form=json.loads(fp.read_text());require(form['status']=='measured_full_form'and form['source_sha']=='4a63d73bee7289deab41fb20f67765d3c6800182'and form['arms']['f16-full']['teacher_positions']==177 and form['arms']['f16-full']['eos'],'complete form receipt')
  pp=locate('physical-decode.json',PHYSICAL);physical=json.loads(pp.read_text());require(physical['status']=='measured'and physical['source_sha']=='ac34f72ca89823a177c538ad598dfa00fc69f124','physical baseline receipt')
  source_bundle=fp.parent/'runtime'
  for rel,row in form['runtime_bundle'].items():verify(source_bundle/rel,row)
  bundle=work/'runtime';shutil.copytree(source_bundle,bundle)
  for rel,row in form['runtime_bundle'].items():verify(bundle/rel,row)
  (bundle/'bin/lighton4b-native-document').chmod(0o755)
  native=bundle/'bin/lighton4b-native-document';require(sha(native)==form['binary_sha256'],'same actual form executable');r['binary_sha256']=sha(native);r['runtime_bundle']=form['runtime_bundle'];r['form_receipt_sha256']=FORM;r['physical_receipt_sha256']=PHYSICAL
  protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());cases=[x for x in protocol['cases']if x['id']in CASE_IDS];require([x['id']for x in cases]==CASE_IDS,'fixed case order')
  # Authenticate every case before any new model forward.
  prepared=[]
  for case in cases:
   name=case['id'];folder=pp.parent/name;art=physical['cases'][name]['artifacts']
   for rel in ['input.rgb','original.json','retained-upstream-response.json','upstream_features/prompt.ids','upstream_features/modalities.ids','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32']:verify(folder/rel,art[rel])
   require(sha(folder/'original.json')==case['reference_sha256']and sha(folder/'retained-upstream-response.json')==case['upstream_response_sha256'],'frozen original/upstream references')
   image=Image.open(ROOT/case['image']).convert('RGB');require(sha(ROOT/case['image'])==case['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'exact frozen image RGB')
   ids=list(map(int,(folder/'upstream_features/prompt.ids').read_text().split()));b=io.BytesIO();np.save(b,np.array([ids],dtype='<i8'),allow_pickle=False);require(len(ids)==case['expected_prompt_tokens']and hashlib.sha256(b.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'complete actual prompt IDs')
   original=json.loads((folder/'original.json').read_text());require(len(original['generated_ids'])==case['reference_generated_ids']and original['generated_ids'][-1]in[248044,248046],'complete EOS teacher reference');teacher=work/(name+'.ids');teacher.write_text(''.join(str(x)+'\n'for x in original['generated_ids']))
   prepared.append((case,folder,art,image,original,teacher));r['cases'][name]={'inputs_authenticated':True,'original_sha256':sha(folder/'original.json'),'prompt_sha256':sha(folder/'upstream_features/prompt.ids')}
  save()
  row=protocol['models']['Q4_K_M'];up=work/row['file'];run('host-cpu',['lscpu']);base.download('https://huggingface.co/'+protocol['model_repo']+'/resolve/'+protocol['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256'])
  row=protocol['base_model'];protected=work/row['file'];base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'physical-native.gguf';audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==form['candidate_sha256'],'identical immutable candidate');r['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(form['arms']['f16-full']['gates']);require(env['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION']=='1','same attention configuration');env['LD_LIBRARY_PATH']=str(bundle/'lib');r['gates']=form['arms']['f16-full']['gates']
  for case,folder,art,image,original,teacher in prepared:
   name=case['id'];dest=out/name;dest.mkdir();require(base.memory()['MemAvailable']>=10<<30,'per-case RAM')
   def tick():
    r['active_progress']={'case':name,'teacher_rows':len(list(dest.glob('teacher.*.logits.f32'))),'greedy_rows':len(list(dest.glob('greedy.*.logits.f32')))};save()
   run(name,[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,256,4,'full','--upstream-vision',folder/'observer-on/source.vision.0.f32'],env,1800,tick)
   summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8');require(summary['teacher_positions']==len(original['generated_ids'])and summary['eos']==(ids[-1]in[248044,248046])and(len(ids)==256 or summary['eos']),'full score/decode closure')
   for j in range(len(original['generated_ids'])):
    x=np.fromfile(dest/f'teacher.{j}.logits.f32',dtype='<f4');require(x.size==248320 and np.isfinite(x).all(),'finite complete teacher row')
   for j,token in enumerate(ids):
    x=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(x.size==248320 and np.isfinite(x).all()and int(x.argmax())==token,'actual independent greedy row')
   require((dest/'prompt.ids').read_bytes()==(folder/'upstream_features/prompt.ids').read_bytes()and sha(dest/'vision.merged.f32')==art['observer-on/source.vision.0.f32']['sha256']and sha(dest/'prefill.input.f32')==art['upstream_features/prefill.input.f32']['sha256']and sha(dest/'prefill.positions.f32')==art['upstream_features/prefill.positions.f32']['sha256'],'same actual input/feature/positions')
   source=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');native_row=np.fromfile(dest/'greedy.0.logits.f32',dtype='<f4');require(source.size==248320 and np.isfinite(source).all(),'finite source firstrow');d=native_row.astype('f8')-source;retained=json.loads((folder/'retained-upstream-response.json').read_text())
   require(sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32')==sha(dest/'prefill.logits.f32'),'same firstrow');r['cases'][name].update(summary=summary,generated_ids=ids,decoded=text,vs_original_bf16=f.metric(base,ids,text,original['generated_ids'],original['decoded']),vs_retained_upstream=f.metric(base,ids,text,retained['tokens'],retained['content']),retained_upstream_vs_original_bf16=f.metric(base,retained['tokens'],retained['content'],original['generated_ids'],original['decoded']),first_score_vs_source={'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'reference_max_abs':float(abs(source).max())},resources=r['commands'][name]);save()
  require(sha(candidate)==form['candidate_sha256'],'unchanged weights');candidate.unlink();r['status']='measured_two_case_calibration'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for name in CASE_IDS for p in(out/name).rglob('*')if p.is_file()};save()
if __name__=='__main__':main()
