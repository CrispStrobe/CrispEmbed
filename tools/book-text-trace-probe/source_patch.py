"""Versioned neutral book observer; reuses pinned, reviewed live capture mechanism.

Only an upstream quantized diagnostic build is patched. No production runtime,
weights, sampling algorithm or request parameters are changed.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("form_trace_patch", ROOT / "tools/form-text-trace-probe/source_patch.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = base.SOURCE
ENV = "CRISPEMBED_TEST_UPSTREAM_TEXT_BOOK_TRACE_DIR"
DONE = "CRISPEMBED_TEST_UPSTREAM_TEXT_BOOK_TRACE_SAMPLES"
POSITION = 405


def configured():
    # Clone module attributes without mutating the shared imported form patch.
    spec = importlib.util.spec_from_file_location("book_form_trace_patch", ROOT / "tools/form-text-trace-probe/source_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENV, module.DONE = ENV, DONE
    helper = module.HELPER
    if helper.count("n==31") != 1 or helper.count("8u<<20") != 1 or helper.count("128u<<20") != 1:
        raise ValueError("reviewed helper substitution anchors drifted")
    module.HELPER = helper.replace("n==31", "n==405").replace("8u<<20", "16u<<20").replace("128u<<20", "256u<<20")
    return module


def patch(root):
    root = Path(root)
    result = configured().patch(root)
    sampling = root / "common/sampling.cpp"
    text = sampling.read_text()
    if text.count("if(samples>=512)") != 1:
        raise ValueError("reviewed sampling bound anchor drifted")
    sampling.write_text(text.replace("if(samples>=512)", "if(samples>=1536)"))
    import hashlib
    result["common/sampling.cpp"]["after_sha256"] = hashlib.sha256(sampling.read_bytes()).hexdigest()
    return result
