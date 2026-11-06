#!/usr/bin/env python3
"""Bounded genuine BF16 mixed free generation; existing form, no new gold/model upload."""
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import types
import urllib.request
import lighton4b_bf16_quant_first_score as base
base.PROTOCOL=base.ROOT/'docs/lighton4b-bf16-mixed-freegen-v1.json'
require=base.require

def validate_generation(ids,scores,counts,cap=256):
    length=len(ids)
    require(cap in(34,256)and 1<=length<=cap and scores==length and counts=={'root':length,'text':length,'vision':1},'bounded independent generation/cache forward counts differ')
    require(all(type(i)is int and 0<=i<248320 for i in ids)and 248046 not in ids[:-1]and(ids[-1]==248046 or length==cap),'actual effective EOS/cap closure differs')

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def decoded_agreement(reference_ids,reference_text,ids,text):
    common=next((i for i,(a,b)in enumerate(zip(reference_ids,ids))if a!=b),min(len(reference_ids),len(ids)))
    words=reference_text.split();native_words=text.split()
    return {'ids_exact':reference_ids==ids,'text_exact':reference_text==text,'matching_prefix_ids':common,
            'first_differing_id':None if reference_ids==ids else common,'reference_tokens':len(reference_ids),'candidate_tokens':len(ids),
            'cer':{'edits':edit_distance(reference_text,text),'reference_codepoints':len(reference_text),'rate':edit_distance(reference_text,text)/len(reference_text)},
            'wer':{'edits':edit_distance(words,native_words),'reference_words':len(words),'rate':edit_distance(words,native_words)/len(words)},
            'normalization':'none; Unicode codepoints; WER Unicode whitespace split, case/punctuation retained; agreement, not human OCR accuracy'}

def reference_asset(protocol,path,sha,limit):
    url='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'+protocol['evidence_source']+'/'+path
    with urllib.request.urlopen(url,timeout=120)as response:raw=response.read(limit+1)
    require(len(raw)<=limit and hashlib.sha256(raw).hexdigest()==sha,'immutable small source evidence differs')
    return json.loads(raw)

class FullOracle(base.Oracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.report.pop('max_new_tokens_per_arm',None)
        cap=self.p['max_new_tokens']
        self.report.update(generation_caps={'original_anchor':1,'mixed':cap},teacher_forced_replay=False,intermediate_capture=False)
        try:
            full=self.p['full_free_generation']is True and cap==256
            trace=self.p.get('activation_trace')is True and not self.p['full_free_generation']and cap==34 and self.p['trace_position']==33
            require((full or trace)and self.p['maximum_score_bytes']==cap*993280,'fixed full-generation/closed trace bound differs')
            self.stage('authenticate-independent-source-evidence')
            self.previous=reference_asset(self.p,self.p['previous_mixed_path'],self.p['previous_mixed_sha256'],2**20)
            self.native=reference_asset(self.p,self.p['native_report_path'],self.p['native_report_sha256'],2**20)
            require(self.native['status']=='measured'and self.native['full_completion_executed']and self.native['source_sha']==self.p['native_source']and self.native['model']['sha256']=='d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474','native comparison source differs')
            require(self.previous['comparison_completed']and self.previous['source_sha']=='50307be7fdb5ec17f367b9b632f506f3d2307126'and self.previous['mixed_policy']==self.p['mixed_policy'],'previous same-weight BF16 source differs')
            full=base.ROOT/'docs/lighton4b-bf16-quant-v1.json';require(base.digest(full)==self.p['original_full_protocol_sha256'],'frozen official full reference differs')
            self.original_full=json.loads(full.read_text())
        except Exception:
            self.report.update(status='blocked',error='small immutable full-generation evidence authentication failed');self.save();raise

    def forward(self,label,model,inputs,processor,torch,np,baseline=None):
        if label=='original':return super().forward(label,model,inputs,processor,torch,np,baseline)
        require(label=='mixed'and self.report['mixed_final_bf16_parameter_sha256']==self.previous['mixed_final_bf16_parameter_sha256'],'all249 mixed effective BF16 weights must match previous oracle')
        require(self.report['original_bf16_tensor_sha256']==self.previous['original_bf16_tensor_sha256'],'all723 original BF16 parameter hashes differ')
        for name,value in inputs.items():
            stream=io.BytesIO();np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
            require(hashlib.sha256(stream.getbuffer()).hexdigest()==self.p['input_sha256'][name],'prepared input changed before independent full generation')
        cap=self.p['max_new_tokens'];self.stage('mixed-independent-greedy-cap'+str(cap));counts={'root':0,'vision':0,'text':0}
        def hook(key):
            def count(module,args):counts[key]+=1
            return count
        handles=[model.register_forward_pre_hook(hook('root')),model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('text'))]
        started=time.monotonic()
        try:
            with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
        finally:
            for h in handles:h.remove()
        ids=result.sequences[:,inputs['input_ids'].shape[1]:].reshape(-1).tolist();length=len(ids)
        validate_generation(ids,len(result.scores),counts,cap)
        text=processor.batch_decode(result.sequences[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        records=[];first=None
        self.report['full_generation_partial']={'ids':ids,'forward_counts':counts,'score_records':records}
        self.save()
        for i,score in enumerate(result.scores):
            value=score.detach().cpu().float().contiguous().numpy();require(value.shape==(1,248320)and np.isfinite(value).all(),'complete finite free-generation scores required')
            raw=value.astype('<f4').tobytes();name=f"{self.p.get('output_prefix','mixed')}.scores-{i:03d}.f32";(self.out/name).write_bytes(raw)
            records.append({'position':i,'file':name,'shape':[1,248320],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            if i==0:first=value.copy()
            if i%16==0:self.save()
        require(sum(r['bytes']for r in records)<=self.p['maximum_score_bytes']and records[0]['sha256']==self.p['mixed_first_score_sha256'],'full score budget or frozen mixed first-score anchor differs')
        id_bytes=(''.join(str(i)+'\n'for i in ids)).encode();text_bytes=text.encode()
        (self.out/(self.p.get('output_prefix','mixed')+'.ids')).write_bytes(id_bytes);(self.out/(self.p.get('output_prefix','mixed')+'.txt')).write_bytes(text_bytes)
        a=baseline.reshape(-1).astype(np.float64);b=first.reshape(-1).astype(np.float64);e=b-a;cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        self.report['arms']['mixed']={'ids':ids,'decoded':text,'scores':records,'parameter_dtype':'bfloat16','all_scores_finite':True,
            'forward_counts':counts,'seconds':time.monotonic()-started,'generation_cap':cap,'eos_completed':ids[-1]==248046,'cap_hit':ids[-1]!=248046 and length==cap,
            'stop':'eos'if ids[-1]==248046 else'max_tokens','decoded_utf8_sha256':hashlib.sha256(text_bytes).hexdigest(),
            'decoded_file':{'file':self.p.get('output_prefix','mixed')+'.txt','bytes':len(text_bytes),'sha256':hashlib.sha256(text_bytes).hexdigest()},
            'ids_file':{'file':self.p.get('output_prefix','mixed')+'.ids','bytes':len(id_bytes),'sha256':hashlib.sha256(id_bytes).hexdigest()},
            'vs_original_bf16':{'max_abs':float(np.abs(e).max()),'rms':float(np.sqrt(np.mean(e*e))),'cosine':cos,'strict_pass':bool(np.abs(e).max()<=self.p['max_abs']and cos>=self.p['min_cos'])},
            'vs_original_full_decoded':decoded_agreement(self.original_full['original_generated_ids'],self.original_full['original_decoded'],ids,text),
            'vs_native_full_decoded':decoded_agreement(self.native['decoded']['ids'],self.native['decoded']['text'],ids,text)if self.native is not None else None}
        self.report.update(mixed_first_score_bit_exact=True,full_generation_completed=True,native_numerical_score_comparison='deferred to read-only consumer; no artifact credential in original worker')
        self.report.pop('full_generation_partial',None)
        self.save();minimal=types.SimpleNamespace(sequences=result.sequences.detach().clone(),scores=(result.scores[0].detach().clone(),))
        del result;gc.collect();return minimal,first

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.report.pop('total_one_position_forwards',None)
        self.report['total_model_forward_calls']=1+len(self.report['arms']['mixed']['ids'])
        self.save();return result

def main():
    base.Oracle=FullOracle
    return base.main()
if __name__=='__main__':raise SystemExit(main())
