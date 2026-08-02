from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "src" / "tesseract_lstm.cpp").read_text()
ORCHESTRATOR_SOURCE = (Path(__file__).parents[1] / "src" / "ocr_orchestrator.cpp").read_text()
PAGESEG_SOURCE = (Path(__file__).parents[1] / "src" / "tesseract_pageseg.cpp").read_text()
DAWG_SOURCE = (Path(__file__).parents[1] / "src" / "tesseract_dawg.cpp").read_text()


def test_int_mode_cache_is_present():
    assert "prepare_lstm_int_weights" in SOURCE
    assert "int8_lstm_row_dot_cached" in SOURCE
    assert "std::vector<int8_t> input_q" in SOURCE
    assert "CRISPEMBED_TESSERACT_DISABLE_INT_CACHE" in SOURCE
    assert "CRISPEMBED_TESSERACT_CROP_PAD" in ORCHESTRATOR_SOURCE
    assert "CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD" in PAGESEG_SOURCE
    assert "CRISPEMBED_TESSERACT_RECODE_COMPOSE" in SOURCE
    assert "recode_classes_to_unichars" in SOURCE
    assert "tesseract_lstm_dawg_component_count" in SOURCE
    assert "tesseract_lstm.dawg_components" in SOURCE
    assert "kDawgMagicNumber" not in DAWG_SOURCE
    assert "unterminated forward edge run" in DAWG_SOURCE
    assert "tesseract_dawg_contains_base64" in DAWG_SOURCE
    assert "tesseract_dawg_has_prefix_base64" in DAWG_SOURCE
    assert "length % 4 != 0" in DAWG_SOURCE
    assert "tesseract_dawg_init_base64" in DAWG_SOURCE
    assert "context_lookup" in DAWG_SOURCE
    assert "dawg_contexts" in SOURCE
    assert "tesseract_lstm_dawg_contains" in SOURCE
    assert "tesseract_lstm_dawg_has_prefix" in SOURCE
    assert "tesseract_lstm_dawg_state" in SOURCE
    assert "TESSERACT_DAWG_COMPLETE_WORD" in DAWG_SOURCE
    assert "CRISPEMBED_TESSERACT_DAWG_PREFIX" in SOURCE


if __name__ == "__main__":
    test_int_mode_cache_is_present()
    print("tesseract kernel contract: PASS")
