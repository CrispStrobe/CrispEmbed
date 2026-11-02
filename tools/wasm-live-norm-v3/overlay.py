"""v3 disposable overlay: the unchanged v1 overlay, then the v3 observer header.

Only src/live_norm_observer.h differs from v1; the crispembed.cpp edit is the
v1 edit byte for byte (asserted by the v3 tests).
"""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('live_norm_v1_overlay_frozen',ROOT/'tools/wasm-live-norm-v1/overlay.py')
V1_OVERLAY=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(V1_OVERLAY)
OBSERVER=ROOT/'tests/wasm-live-norm-v3/observer.h'

def apply(source):
 changes=V1_OVERLAY.apply(source)
 target=Path(source)/'src/live_norm_observer.h';target.write_bytes(OBSERVER.read_bytes())
 changes['src/live_norm_observer.h']={'before':None,'after':hashlib.sha256(target.read_bytes()).hexdigest()}
 return changes
