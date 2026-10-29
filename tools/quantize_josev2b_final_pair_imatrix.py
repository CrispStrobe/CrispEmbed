#!/usr/bin/env python3
"""Two fixed paired-Q4 development controls; exact existing C quantizer only."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/josev-ffn-imatrix-v1'),str(ROOT/'tools')]
import josev_ffn_statistics as imatrix
import quantize_josev2b_protected as previous
from ci_process import run
IDENTITIES=('uncalibrated-pair','calibrated-pair')
ROLE='q4-last-f16-down'
SOURCE=previous.SOURCE
COUNTS={0:292,1:148,8:0,12:2}
hosted=previous.hosted
require=previous.require
def plan(rows):
 result=previous.recipe(rows,ROLE)
 require({t:sum(v==t for v in result.values())for t in COUNTS}==COUNTS,'Exact paired-Q4 role census')
 require({n for n,t in result.items()if t==12}==set(imatrix.NAMES),'Only final gate/up Q4')
 return result

def quantize(source,output,executable,identity,importance=None):
 source,output,executable=map(Path,(source,output,executable))
 require(identity in IDENTITIES and source.is_file()and source.stat().st_size==SOURCE['bytes']and previous.digest(source)==SOURCE['sha256']
  and executable.is_file()and not output.exists(),'Fresh immutable source/target/executable required')
 require((importance is not None)==(identity=='calibrated-pair'),'Calibration is explicit, no Q8/imatrix fallback')
 proof=imatrix.verify(importance,1193)if importance is not None else None
 wanted=plan(previous.schema());args=[str(executable),str(source),str(output),'q4_k','--metadata-string','josev2b.packed.format='+previous.FORMAT]
 for name,kind in wanted.items():
  if kind!=12:args+=['--tensor-type',name+'='+previous.TYPES[kind].lower()]
 if importance is not None:args+=['--imatrix',str(importance)]
 env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')};env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 output.parent.mkdir(parents=True,exist_ok=True);log=output.with_suffix('.quantize.log');result=run(args,log,env=env,timeout=2400)
 log.with_name(log.name+'.process.json').write_text(json.dumps(result,indent=2)+'\n')
 require(not result['timed_out']and result['exit']==0,'Actual C quantizer failed')
 text=log.read_text();require('(fallback 'not in text,'No fallback encoding')
 used=imatrix.usage(text)if importance is not None else dict(loaded=0,used=0)
 if importance is None:require('(imatrix)'not in text and 'with imatrix'not in text,'Uncalibrated control consumed unexpected importance')
 if importance is not None:require(imatrix.verify(importance,1193)==proof,'Importance changed during quantization')
 converted=previous.verify_conversion(source,output,previous.schema(),ROLE,vector=True)
 require(converted['output_bytes']==4879929824,'Exact unchanged mixed storage extent')
 if importance is None:require(previous.digest(output)=='289798eb4066585e59dec70c1c5447b02eece199699fce1b6a6e38933d61d31d','Uncalibrated model differs from retained development control')
 return dict(schema='crispembed-josev-final-pair-imatrix-conversion-v1',completed=True,identity=identity,role=ROLE,
  source=SOURCE,output_sha256=previous.digest(output),quantizer_sha256=previous.digest(executable),quantizer_log_sha256=previous.digest(log),
  imatrix=proof,imatrix_usage=used,quantizer_process=result,**converted)
