#!/usr/bin/env python3
"""Explicit BF16-vision candidate using unchanged immutable quantized models."""
import importlib.util
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise RuntimeError('hosted-only candidate')
    spec=importlib.util.spec_from_file_location('native_document',ROOT/'tools/ci-heavy/lighton4b_native_document.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.GATES=tuple(module.GATES)+('CRISPEMBED_LIGHTON4B_VISION_BF16',)
    module.main()
if __name__=='__main__':main()
