from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "models" / "convert-tesseract-to-gguf.py").read_text()


def test_converter_preserves_dawg_payloads_losslessly():
    assert "import base64" in SOURCE
    assert "DAWG_COMPONENT_NAMES" in SOURCE
    assert "tesseract_lstm.dawg_components" in SOURCE
    assert "base64.b64encode(payload)" in SOURCE
    assert "tesseract_lstm.dawg.{name}.sha256" in SOURCE


if __name__ == "__main__":
    test_converter_preserves_dawg_payloads_losslessly()
    print("tesseract converter contract: PASS")
