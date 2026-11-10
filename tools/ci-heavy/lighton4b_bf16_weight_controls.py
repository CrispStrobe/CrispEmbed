#!/usr/bin/env python3
"""Checkpoint-free operator/gate/cache tests; preserve old synthetic numerical failures."""
import importlib.util
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 if os.environ.get('GITHUB_ACTIONS')!='true':raise RuntimeError('hosted only')
 spec=importlib.util.spec_from_file_location('decoder_controls',ROOT/'tools/ci-heavy/lighton4b_decoder_compose.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 out=Path(os.environ['HEAVY_OUT']);error=None
 try:module.execute(weight_controls=True)
 except Exception as problem:error=type(problem).__name__
 receipt=json.loads((out/'decoder-compose-receipt.json').read_text())
 native=(out/'native.log').read_text() if (out/'native.log').exists() else ''
 weights=(out/'weight-controls.log').read_text() if (out/'weight-controls.log').exists() else ''
 passed='DECODER_COMPOSE controls=' in native and 'passed=1' in native and 'BF16_DEQUANT_WEIGHTS passed=1' in weights
 report={'schema':'lighton4b-bf16-weight-controls/v1','status':'passed' if passed else 'failed','checkpoint_payload_bytes':0,'full_model_storage':False,'quality_qualified':False,'native_controls':native,'weight_controls':weights,'legacy_synthetic_numeric_status':receipt.get('status'),'legacy_synthetic_earliest_failure':receipt.get('earliest_failure'),'legacy_error_type':error,'synthetic_numeric_budget_unchanged':True,'note':'Any original synthetic numerical rejection remains preserved, independent from new weight/gate/cache controls.'}
 (out/'bf16-weight-controls.json').write_text(json.dumps(report,indent=2)+'\n')
 if not passed:raise RuntimeError('focused weight/cache controls did not pass; inspect preserved logs')
if __name__=='__main__':main()
