from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "src" / "tesseract_lstm.cpp").read_text()


def test_tesseract_runtime_keeps_int_mode_and_lut_path():
    """Protect the fast/quality path from an accidental old-file overwrite."""
    assert "bool int_mode" in SOURCE
    assert "int8_lstm_row_dot" in SOURCE
    assert "tesseract_tanh" in SOURCE
    assert "tesseract_logistic" in SOURCE


def test_tesseract_runtime_keeps_gated_scratch_reuse():
    assert "CRISPEMBED_TESSERACT_REUSE_SCRATCH" in SOURCE
    assert "ctx->reuse_scratch" in SOURCE


def test_tesseract_runtime_exposes_gated_dawg_queries():
    assert "CRISPEMBED_TESSERACT_DAWG_LOAD" in SOURCE
    assert "tesseract_lstm_dawg_matches" in SOURCE
    assert "tesseract_lstm_dawg_matches_utf8" in SOURCE


if __name__ == "__main__":
    test_tesseract_runtime_keeps_int_mode_and_lut_path()
    test_tesseract_runtime_keeps_gated_scratch_reuse()
    print("tesseract runtime contract: PASS")
