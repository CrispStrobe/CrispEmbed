"""Isolated entry point for byte-frozen, stdlib-only actual Ref controls."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    frozen = Path(__file__).resolve().parents[1] / 'unlimited-ocr2-q4k'
    sys.path.insert(0, str(frozen))
    runpy.run_path(str(frozen / 'controls.py'), run_name='__main__')
