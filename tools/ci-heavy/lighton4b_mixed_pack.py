#!/usr/bin/env python3
"""Hosted authenticated stream composition of a predeclared mixed Q4/Q8 candidate."""
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_canonicalize import sha,require,parse
from lighton4b_mixed_pack import compose,RECIPE

def download(url,path,size,expected):
    h=hashlib.sha256();count=0
    try:
        with urllib.request.urlopen(url,timeout=120)as response,path.open('xb')as stream:
            while block:=response.read(8*2**20):
                count+=len(block);require(count<=size,'source transfer oversized');stream.write(block);h.update(block)
    except Exception:raise ValueError('bounded immutable source transfer failed')from None
    require(count==size and h.hexdigest()==expected,'immutable source payload differs')

def main():
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable packaging required')
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);p=json.loads((ROOT/'docs/lighton4b-mixed-calibration-v1.json').read_text());policy=json.loads((ROOT/'docs/lighton4b-mixed-packing-v1.json').read_text())
    r={'schema':'lighton4b-mixed-candidate/v1','status':'blocked','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'recipe':RECIPE,'model_forward_executed':False,'quality_qualified':False,'native_qualified':False,'stage':'controls-and-capacity'}
    try:
        with(out/'controls.log').open('w')as log:subprocess.run([sys.executable,ROOT/'tests/test_lighton4b_mixed_pack.py','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        r['preflight']={'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free,'shared_filesystem':scratch.stat().st_dev==out.stat().st_dev}
        memory={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
        r['preflight'].update(memory);r['preflight']['load_average']=list(os.getloadavg());require(memory['MemAvailable']>=512*1024**2,'insufficient streaming packer RAM')
        required=p['resource_guards']['packer_min_free_disk_bytes'];require(r['preflight']['scratch_free_bytes']>=required,'insufficient hosted scratch capacity')
        if not r['preflight']['shared_filesystem']:require(r['preflight']['output_free_bytes']>=12*1024**3,'insufficient separate output capacity')
        require(sha(ROOT/'docs/lighton4b-mixed-calibration-v1.json')==policy['calibration_protocol_sha256'],'frozen calibration protocol differs')
        require(p['recipe_frozen_before_all_panel_outputs']and p['candidate_recipe']==RECIPE,'predeclared recipe differs')
        inputs=p['candidate_inputs'];audit=policy['tensor_audit'];path=scratch/'tensor-audit.json'
        # Small public report, pinned by both source and full bytes hash.
        with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'+audit['source']+'/'+audit['path'],timeout=90)as response:data=response.read(2**20)
        require(hashlib.sha256(data).hexdigest()==audit['sha256'],'source tensor audit differs');path.write_bytes(data);census=json.loads(data)
        models={};r['stage']='retrieve-authenticated-input-models'
        for recipe in('q4_k','q8_0'):
            pin=inputs[recipe];path=scratch/pin['file'];download('https://huggingface.co/'+inputs['repository']+'/resolve/'+inputs['revision']+'/'+pin['file'],path,pin['bytes'],pin['sha256']);models[recipe]=path
            require(census['artifacts'][recipe]['sha256']==pin['sha256'],'source audit candidate identity differs')
        expected={key:census['artifacts'][key]['tensors']for key in models};r['stage']='stream-compose-and-readback'
        model=out/'lightonocr3-4b-q4k-mixers-q8-v1.gguf';receipt=compose(models['q4_k'],models['q8_0'],model,expected)
        counts=dict(collections.Counter({0:'F32',8:'Q8_0',12:'Q4_K'}[record['type']]for record in receipt['records']));upgrades=sum(record['source_recipe']=='q8_0'for record in receipt['records'])
        require(receipt['tensor_count']==723 and counts==policy['expected_types']and upgrades==152 and receipt['payload_bytes']==policy['expected_payload_bytes']and receipt['metadata_file_type']==14,'mixed physical storage policy differs')
        r.update(status='packed-and-readback-verified',stage='complete',candidate={'file':model.name,'sha256':sha(model),'bytes':model.stat().st_size},types=counts,upgraded_attention_tensors=upgrades,ffn_q4_tensors=96,protocol_sha256=policy['calibration_protocol_sha256'],tensor_audit_sha256=audit['sha256'],source_candidates=inputs,packing=receipt)
    except Exception as error:
        message=str(error)if isinstance(error,ValueError)else'hosted composition failed'
        for value,label in((ROOT,'[repository]'),(scratch,'[scratch]'),(out,'[report]')):message=message.replace(str(value),label)
        r.update(error_type=type(error).__name__,error=message);return 1
    finally:(out/'mixed-candidate.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
