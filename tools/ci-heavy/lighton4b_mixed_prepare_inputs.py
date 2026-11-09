#!/usr/bin/env python3
"""Freeze four predeclared image inputs; no model/framework forward/download."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise RuntimeError('hosted preparation required')
    out=Path(os.environ['HEAVY_OUT']);protocol=ROOT/'docs/lighton4b-mixed-calibration-v1.json';p=json.loads(protocol.read_text())
    report={'schema':'lighton4b-mixed-frozen-inputs/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256':sha(protocol),'model_downloaded':False,'model_forward_executed':False,'cases':[]}
    try:
        subprocess.run([sys.executable,'-m','pip','install','Pillow==11.3.0'],check=True)
        from PIL import Image
        for case in p['cases']:
            source=ROOT/case['source']
            if sha(source)!=case['source_sha256']:raise ValueError('frozen source image differs')
            image=Image.open(source).convert('RGB');resize=case['transform']['resize_width_height']
            if resize:image=image.resize(tuple(resize),Image.Resampling.BICUBIC)
            image=Image.frombytes('RGB',image.size,image.tobytes())
            folder=out/case['id'];folder.mkdir();path=folder/'input.png';image.save(path,format='PNG',compress_level=6,optimize=False)
            rgb=image.tobytes();(folder/'input.rgb').write_bytes(rgb)
            report['cases'].append({'id':case['id'],'split':case['split'],'source_sha256':case['source_sha256'],'png_file':case['id']+'/input.png','png_sha256':sha(path),'png_bytes':path.stat().st_size,'rgb_file':case['id']+'/input.rgb','rgb_sha256':hashlib.sha256(rgb).hexdigest(),'rgb_bytes':len(rgb),'width':image.width,'height':image.height,'expected_grid':case['expected_grid'],'expected_input_tokens':case['expected_input_tokens']})
        report['status']='prepared-before-model-outputs'
    finally:(out/'frozen-inputs.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
