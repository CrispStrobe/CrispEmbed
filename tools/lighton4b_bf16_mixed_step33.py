#!/usr/bin/env python3
"""Passive original mixed-BF16 step33 capture, all34 score hashes neutral."""
from pathlib import Path
import hashlib
import json
import lighton4b_bf16_mixed_freegen as full
base=full.base
base.PROTOCOL=base.ROOT/'docs/lighton4b-bf16-mixed-step33-v1.json'
require=base.require

def stage_names():
    prefix='greedy.33.'
    return {prefix+n for n in ['input','positions','positions.full','norm','head.input','head.output']+[f'layer.{i}'for i in range(32)]+['layer0.'+n for n in ['input_norm','qkv','z','a','b','attention','attention_residual','post_norm','gate','up','activation','down']]}

def validate_neutrality(arm,reference):
    require(len(reference['scores'])>=34 and arm['ids']==reference['ids'][:34]and len(arm['scores'])==34,'actual source first34 IDs differ')
    require(all(a['sha256']==b['sha256']and a['position']==b['position']==i for i,(a,b)in enumerate(zip(arm['scores'],reference['scores'][:34]))),'observer neutrality/all34 immutable source scores differ')

class TraceOracle(full.FullOracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        require(self.p['activation_trace']and self.p['trace_position']==33 and self.p['max_new_tokens']==34 and self.p['expected_trace_stage_count']==50,'closed step33 trace contract differs')
        self.capture=full.reference_asset(self.p,self.p['full_capture_path'],self.p['full_capture_sha256'],2*2**20)
        require(self.capture['full_generation_completed']and len(self.capture['arms']['mixed']['ids'])==178,'retained178-row source capture required')
        self.report.update(intermediate_capture=True,activation_trace=True,trace_position=33,trace_stages={})

    def forward(self,label,model,inputs,processor,torch,np,baseline=None):
        if label=='original':return super().forward(label,model,inputs,processor,torch,np,baseline)
        records=self.report['trace_stages'];self.step=-1;total=0;handles=[]
        self.report['operator_runtime']={'cpu_capability':torch._C._get_cpu_capability(),'torch_git_version':torch.version.git_version,'threads':torch.get_num_threads(),'mkldnn_enabled':torch.backends.mkldnn.enabled}
        def capture(name,value):
            nonlocal total
            if self.step!=33:return
            require(name in stage_names()and name not in records and isinstance(value,torch.Tensor),'unique actual source stage required')
            value=value.detach().cpu().contiguous();array=value.float().numpy();require(np.isfinite(array).all(),'finite source stage required')
            raw=array.astype('<f4').tobytes();total+=len(raw);require(total<=self.p['maximum_trace_bytes'],'bounded source activation capture exceeded')
            filename=name+'.f32';(self.out/filename).write_bytes(raw)
            records[name]={'file':filename,'shape':list(value.shape),'original_dtype':str(value.dtype),'f32_elements':value.numel(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'abs_max':float(np.abs(array).max())}
        def root_pre(module,args,kwargs):
            self.step+=1
            if self.step==33:
                ids=kwargs.get('input_ids');require(ids is not None and ids.numel()==1,'actual cached single-token source input required')
                got=ids.reshape(-1).tolist();require(got==[self.capture['arms']['mixed']['ids'][32]],'actual original shared-prefix input token differs')
                self.report['trace_input_ids']=got
        handles.append(model.register_forward_pre_hook(root_pre,with_kwargs=True))
        language=model.model.language_model;layer=language.layers[0]
        require(len(language.layers)==32 and hasattr(layer,'linear_attn'),'actual32-layer source blueprint differs')
        def language_pre(module,args,kwargs):
            if self.step!=33:return
            capture('greedy.33.input',kwargs['inputs_embeds'])
            positions=kwargs['position_ids'];capture('greedy.33.positions.full',positions)
            require(positions.ndim==3 and positions.shape[0]in(3,4),'actual mrope source positions differ')
            capture('greedy.33.positions',positions[-3:])
            self.report['trace_positions']=positions[-3:].reshape(-1).tolist()
        handles.append(language.register_forward_pre_hook(language_pre,with_kwargs=True))
        def output_hook(name):
            def hook(module,args,value):capture('greedy.33.'+name,value)
            return hook
        def input_hook(name):
            def hook(module,args):
                if self.step==33:require(len(args)>0,'actual positional source stage input required');capture('greedy.33.'+name,args[0])
            return hook
        for i,block in enumerate(language.layers):handles.append(block.register_forward_hook(output_hook('layer.'+str(i))))
        handles.append(language.norm.register_forward_hook(output_hook('norm')))
        handles.append(model.lm_head.register_forward_pre_hook(input_hook('head.input')))
        handles.append(model.lm_head.register_forward_hook(output_hook('head.output')))
        modules={'input_norm':layer.input_layernorm,'attention':layer.linear_attn,'post_norm':layer.post_attention_layernorm,'gate':layer.mlp.gate_proj,'up':layer.mlp.up_proj,'down':layer.mlp.down_proj}
        for name,module in modules.items():handles.append(module.register_forward_hook(output_hook('layer0.'+name)))
        for name in ('qkv','z','a','b'):handles.append(getattr(layer.linear_attn,'in_proj_'+name).register_forward_hook(output_hook('layer0.'+name)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(input_hook('layer0.attention_residual')))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(input_hook('layer0.activation')))
        try:result=super().forward(label,model,inputs,processor,torch,np,baseline)
        finally:
            for handle in handles:handle.remove()
        require(self.step==33 and set(records)==stage_names(),'all50 actual target stages must be captured once')
        arm=self.report['arms']['mixed'];reference=self.capture['arms']['mixed']
        validate_neutrality(arm,reference)
        require(records['greedy.33.head.input']['sha256']==records['greedy.33.norm']['sha256'],'actual source finalnorm/head boundary differs')
        self.report.update(trace_observer_all34_scores_bit_exact=True,trace_observer_all34_ids_exact=True,trace_bytes=total,
                           head_output_matches_generated_score=records['greedy.33.head.output']['sha256']==reference['scores'][33]['sha256'])
        self.save();return result

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.report['full_generation_completed']=False
        self.report['activation_trace_completed']=True
        for key in ('vs_original_full_decoded','vs_native_full_decoded'):self.report['arms']['mixed'].pop(key,None)
        self.report['scope']='actual mixed source step33,34-token partial replay; no EOS/full decode claim'
        self.save();return result

def main():
    base.Oracle=TraceOracle
    return base.main()
if __name__=='__main__':raise SystemExit(main())
