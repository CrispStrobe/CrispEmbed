#!/usr/bin/env python3
"""Explicit contracted-norm candidate; preserves earlier norm-only A/B protocol."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 spec=importlib.util.spec_from_file_location('replay',ROOT/'tools/ci-heavy/lighton4b_vision_layer0_replay.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.NORM_GATES=('CRISPEMBED_LIGHTON4B_VISION_BF16_NORM','CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM')
 module.main()
if __name__=='__main__':main()
