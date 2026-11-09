#!/usr/bin/env python3
"""Unchanged-file Q4 oracle in the genuine BF16 original, bounded independent decode."""
import json
import os
from pathlib import Path
import lighton4b_bf16_mixed_freegen as full
base=full.base
base.PROTOCOL=base.ROOT/'docs/lighton4b-bf16-plain-q4-freegen-v1.json'
OriginalOracle=base.Oracle

class PlainOracle(full.FullOracle):
    def __init__(self,*args,**kwargs):
        OriginalOracle.__init__(self,*args,**kwargs)
        base.require(self.p['plain_q4_full']is True and self.p['full_free_generation']is True and self.p['max_new_tokens']==256 and self.p['maximum_score_bytes']==256*993280,'fixed plain Q4 full bound differs')
        self.report.pop('max_new_tokens_per_arm',None)
        self.report.update(generation_caps={'original_anchor':1,'q4_k':256},teacher_forced_replay=False,intermediate_capture=False,precision_overrides=False)
        self.stage('authenticate-plain-q4-first-score')
        previous=full.reference_asset(self.p,self.p['previous_plain_q4_path'],self.p['previous_plain_q4_sha256'],2**20)
        base.require(previous['comparison_completed']and previous['original_first_score_bit_exact']and previous['arms']['q4_k']['sha256']==self.p['mixed_first_score_sha256'],'previous genuine Q4 source anchor differs')
        weights={r['gguf']:r['bf16_sha256']for r in previous['q4_k-weights']['tensors']if r['quantized']}
        base.require(sorted(weights)==self.p['plain_q4_names']and len(weights)==249 and len(previous['original_bf16_tensor_sha256'])==723,'previous complete Q4 parameter map differs')
        # Compatibility keys remain internal to the shared streaming implementation.
        self.previous={'mixed_final_bf16_parameter_sha256':weights,'original_bf16_tensor_sha256':previous['original_bf16_tensor_sha256']}
        self.native=None
        path=base.ROOT/'docs/lighton4b-bf16-quant-v1.json'
        base.require(base.digest(path)==self.p['original_full_protocol_sha256'],'official full reference differs')
        self.original_full=json.loads(path.read_text())

    def run(self,*args,**kwargs):
        import torch
        self.report['operator_runtime']={'cpu_capability':torch._C._get_cpu_capability(),'torch_git_version':torch.version.git_version,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'mkldnn_enabled':torch.backends.mkldnn.enabled,'torch_config':torch.__config__.show(),'isa_environment':{k:os.environ.get(k)for k in ('ATEN_CPU_CAPABILITY','MKL_ENABLE_INSTRUCTIONS','MKL_CBWR','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA')}}
        cpu={}
        for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines():
            if ':'in line:
                key,value=line.split(':',1)
                if key.strip()in ('vendor_id','cpu family','model','model name','stepping','flags'):cpu[key.strip()]=value.strip()
        self.report['operator_runtime']['cpu_metadata']=cpu
        self.save()
        base.require(all(os.environ.get(k)==v for k,v in self.p['backend_dispatch'].items())and torch._C._get_cpu_capability()=='AVX2','explicit AVX2 backend dispatch profile differs')
        result=super().run(*args,**kwargs)
        self.report['arms']['q4_k']=self.report['arms'].pop('mixed')
        self.report['plain_q4_final_bf16_parameter_sha256']=self.report.pop('mixed_final_bf16_parameter_sha256')
        self.report['plain_q4_first_score_bit_exact']=self.report.pop('mixed_first_score_bit_exact')
        self.report.pop('mixed_policy',None)
        self.report['precision_policy']='unchanged immutable Q4_K file:248 Q4_K matrices and Q8_0 embedding; no overrides'
        self.save()
        return result

def main():
    base.Oracle=PlainOracle
    return base.main()
if __name__=='__main__':raise SystemExit(main())
