#!/usr/bin/env python3
"""All verified vision math including merger; rejected rotary and decoder off."""
import importlib.util
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 os.environ['CRISPEMBED_LIGHTON4B_BF16_LINEAR']='0';os.environ['CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2']='0'
 spec=importlib.util.spec_from_file_location('ab',ROOT/'tools/ci-heavy/lighton4b_vision_linear_ab.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.CONV_GATE=True;module.ATTENTION_GATE=True;module.GELU_GATE=True;module.MERGER_GATE=True;module.main()
if __name__=='__main__':main()
