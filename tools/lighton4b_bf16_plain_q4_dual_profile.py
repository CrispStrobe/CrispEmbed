#!/usr/bin/env python3
"""Two closed CPU profiles selected before forward solely by measured CPU identity."""
from pathlib import Path
import lighton4b_bf16_plain_q4_freegen as plain
base=plain.base
base.PROTOCOL=base.ROOT/'docs/lighton4b-bf16-plain-q4-dual-profile-v1.json'
def select_profile(protocol,identity):
    key='/'.join(identity.get(k,'')for k in ('vendor_id','cpu family','model'))
    base.require(key in protocol['cpu_profiles'],'unadmitted CPU vendor/family/model')
    return key,protocol['cpu_profiles'][key]
class DualOracle(plain.PlainOracle):
    def __init__(self,*args,**kwargs):
        identity={}
        for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines():
            if ':'in line:
                key,value=line.split(':',1);identity[key.strip()]=value.strip()
        protocol=args[3]
        key,profile=select_profile(protocol,identity)
        # The caller and observer share this resolved in-memory protocol; immutable
        # JSON contains both closed anchors and remains hashed, never rewritten.
        protocol['original_first_score_sha256']=profile['original_first_score_sha256']
        super().__init__(*args,**kwargs)
        self.profile=profile
        self.report.update(reference_profile=profile['profile'],cpu_profile_key=key,resolved_original_anchor_sha256=profile['original_first_score_sha256'],frozen_reference_substituted=False,frozen_official_anchor_matched=profile['require_frozen_q4_anchor'],backend_quality_qualified=False)
        self.save()
    def check_first_score(self,record):
        if self.profile['require_frozen_q4_anchor']:super().check_first_score(record)
        self.report['first_q4_vs_frozen_profile']={'actual_sha256':record['sha256'],'frozen_sha256':self.p['mixed_first_score_sha256'],'bit_exact':record['sha256']==self.p['mixed_first_score_sha256'],'numeric_comparison':'separate read-only consumer on raw retained vectors; CPU profile explicit'}
        self.save()
def main():
    base.Oracle=DualOracle
    return base.main()
if __name__=='__main__':raise SystemExit(main())
