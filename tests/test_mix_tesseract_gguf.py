import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mix_tesseract_gguf", ROOT / "models" / "mix-tesseract-gguf.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_names_preserves_order_and_supports_patterns():
    names = ["conv.weight", "lstm.3.weight_ih", "lstm.3.weight_hh",
             "lstm.2.bias"]
    assert MODULE.selected_names(names, ["lstm.3.*"]) == names[1:3]


def test_selected_names_empty_is_explicit():
    assert MODULE.selected_names(["lstm.1.bias"], ["output.*"]) == []
