#!/usr/bin/env python3
"""Read-only distribution diagnostics for authenticated BF16 logit rows stored as F32.

This reads existing scores, never a model. KLD is source||candidate at one causal
position, not a decoded-quality measure or the publisher's corpus KLD benchmark.
"""
import argparse,array,hashlib,json,math,struct,sys
from pathlib import Path

def compare_values(reference,candidate):
 if not reference or len(reference)!=len(candidate)or not all(math.isfinite(v)for v in list(reference)+list(candidate)):raise ValueError('finite equal nonempty rows required')
 n=len(reference);error=[b-a for a,b in zip(reference,candidate)];mean=math.fsum(error)/n
 def distribution(values):
  peak=max(values);normalizer=math.fsum(math.exp(v-peak)for v in values);logz=peak+math.log(normalizer)
  return [math.exp(v-peak)/normalizer for v in values],logz
 p,logp=distribution(reference);q,logq=distribution(candidate);gold=max(range(n),key=reference.__getitem__);selected=max(range(n),key=candidate.__getitem__)
 return {'selected_id':selected,'original_selected_id':gold,'probability_of_original_id':q[gold],'original_probability':p[gold],'mean_logit_error':mean,'rms_error':math.sqrt(math.fsum(v*v for v in error)/n),'centered_rms_error':math.sqrt(math.fsum((v-mean)**2 for v in error)/n),'max_abs_error':max(map(abs,error)),'score_rms':math.sqrt(math.fsum(v*v for v in candidate)/n),'max_abs_score':max(map(abs,candidate)),'source_to_native_next_token_kld':math.fsum(weight*((a-logp)-(b-logq))for weight,a,b in zip(p,reference,candidate))}

def read_row(path,digest):
 raw=Path(path).read_bytes()
 if len(raw)!=993280 or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('authenticated248320-vocabulary row required')
 if any(word&65535 for(word,)in struct.iter_unpack('<I',raw)):raise ValueError('BF16-expanded F32 row required')
 values=array.array('f');values.frombytes(raw)
 if sys.byteorder!='little':values.byteswap()
 return values

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--reference-sha256',required=True);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--candidate-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=compare_values(read_row(args.reference,args.reference_sha256),read_row(args.candidate,args.candidate_sha256));r.update(schema='lighton4b-one-position-logit-diagnostics/v1',reference_sha256=args.reference_sha256,candidate_sha256=args.candidate_sha256,reference_model_forward_executed=False,f32_model_instantiated=False,decoded_quality_qualified=False,kld_direction='original BF16 distribution || candidate distribution');args.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
