#!/usr/bin/env python3
"""Explicit verified norm+nn.Linear+Conv3D candidate, no default promotion."""
import importlib.util
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 # Isolate vision changes: decoder candidate remains explicitly disabled.
 os.environ['CRISPEMBED_LIGHTON4B_BF16_LINEAR']='0'
 spec=importlib.util.spec_from_file_location('ab',ROOT/'tools/ci-heavy/lighton4b_vision_linear_ab.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.CONV_GATE=True;module.main()
if __name__=='__main__':main()
