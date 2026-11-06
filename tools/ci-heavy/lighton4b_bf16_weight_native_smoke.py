#!/usr/bin/env python3
"""One same-Q4-image OFF/ON BF16 text weight first-score/partial16 diagnostic."""
import importlib.util
from pathlib import Path
path=Path(__file__).with_name('lighton4b_corrected_mixed_smoke.py')
spec=importlib.util.spec_from_file_location('corrected_native',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':module.main(weight_ab=True)
