"""v3 disposable overlay: the unchanged v1 overlay, then two v3 edits.

1. src/live_norm_observer.h is the v3 observer (diagnostics + `measuring`).
2. The three unevaluated scheduler-reservation builds
   (`measure_gf = build_encoder_graph(...)`) are bracketed with
   `ctx->live_norm.measuring`. Run 41421952964 showed the native first encode's
   reservation graph at bucket_seq_len(12)=16 rows reaching the observer.
Nothing else in the v1 crispembed.cpp edit changes (asserted by the v3 tests).
"""
import hashlib,importlib.util,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('live_norm_v1_overlay_frozen',ROOT/'tools/wasm-live-norm-v1/overlay.py')
V1_OVERLAY=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(V1_OVERLAY)
OBSERVER=ROOT/'tests/wasm-live-norm-v3/observer.h'
MEASURE=re.compile(r'^(?P<i>[ ]+)ggml_cgraph \* measure_gf = build_encoder_graph\(ctx, [^\n]*\);$',re.M)
RESERVATION_BUILDS=3

def bracket(text):
 found=MEASURE.findall(text)
 if len(found)!=RESERVATION_BUILDS:raise ValueError('Expected exactly %d reservation builds, found %d'%(RESERVATION_BUILDS,len(found)))
 return MEASURE.sub(lambda m:f"{m['i']}ctx->live_norm.measuring = true;\n{m[0]}\n{m['i']}ctx->live_norm.measuring = false;",text)

def apply(source):
 changes=V1_OVERLAY.apply(source)
 p=Path(source)/'src/crispembed.cpp';p.write_text(bracket(p.read_text()))
 changes['src/crispembed.cpp']['after']=hashlib.sha256(p.read_bytes()).hexdigest()
 target=Path(source)/'src/live_norm_observer.h';target.write_bytes(OBSERVER.read_bytes())
 changes['src/live_norm_observer.h']={'before':None,'after':hashlib.sha256(target.read_bytes()).hexdigest()}
 return changes
