import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantize", ROOT / "models" / "quantize.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_keep_pattern_preserves_critical_tensor():
    assert not MODULE.should_quantize(
        "lstm.0.weight_hh", (64, 256), "q8_0", ["lstm.0.weight_hh"])


def test_default_quantization_policy_is_unchanged():
    assert MODULE.should_quantize("lstm.0.weight_hh", (64, 256), "q8_0")
    assert not MODULE.should_quantize("lstm.0.bias", (256,), "q8_0")
