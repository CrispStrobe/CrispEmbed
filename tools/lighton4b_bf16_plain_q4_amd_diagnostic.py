#!/usr/bin/env python3
"""Distinct AMD BF16 source diagnostic; never replaces the frozen original."""
import lighton4b_bf16_plain_q4_freegen as plain
base=plain.base
base.PROTOCOL=base.ROOT/'docs/lighton4b-bf16-plain-q4-amd-diagnostic-v1.json'
class AMDOracle(plain.PlainOracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        base.require(self.p['distinct_backend_diagnostic']is True and self.p['original_first_score_sha256']=='02ca259be95fc699f079f07d7ca35078728507fef18b4fa82fc630ff23b50e2c','distinct repeated AMD baseline differs')
        self.report.update(reference_profile=self.p['reference_profile'],frozen_reference_substituted=False,frozen_official_anchor_matched=False,backend_quality_qualified=False)
    def check_first_score(self,record):
        self.report['first_q4_vs_frozen_profile']={'actual_sha256':record['sha256'],'frozen_sha256':self.p['mixed_first_score_sha256'],'bit_exact':record['sha256']==self.p['mixed_first_score_sha256'],'numeric_comparison':'raw retained vectors compared by separate read-only consumer; backend difference explicit'}
        self.save()
def main():
    base.Oracle=AMDOracle
    return base.main()
if __name__=='__main__':raise SystemExit(main())
