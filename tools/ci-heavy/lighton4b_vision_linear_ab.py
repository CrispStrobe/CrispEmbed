#!/usr/bin/env python3
"""Incremental norm-only vs norm+linear ordinary vision/decoded A/B."""
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONV_GATE=False
ATTENTION_GATE=False
def main():
 spec=importlib.util.spec_from_file_location('replay',ROOT/'tools/ci-heavy/lighton4b_vision_layer0_replay.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.BASELINE_NORM_GATES=('CRISPEMBED_LIGHTON4B_VISION_BF16_NORM','CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM')
 module.NORM_GATES=module.BASELINE_NORM_GATES+('CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR',)+(('CRISPEMBED_LIGHTON4B_VISION_BF16_CONV',)if CONV_GATE else ())+((
  'CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION','CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2')if ATTENTION_GATE else ())
 frozen=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-baseline-stages.json').read_text())
 if frozen['run_id']!=42028668863 or frozen['native_report_sha256']!='7a1a74ddbeaf1eaa1e916f091ab38a5895e59f7eb04165972a36ae8d4cb1ae14':raise ValueError('baseline receipt differs')
 module.COMPOSED_STAGE_NAMES=tuple(frozen['stages'])
 module.BASELINE_STAGE_OVERRIDES={name:{'native_sha256':row['sha256']}for name,row in frozen['stages'].items()}
 module.main()
if __name__=='__main__':main()
