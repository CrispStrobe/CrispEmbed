from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "src" / "tesseract_lstm.cpp").read_text()


def test_int_mode_cache_is_present():
    assert "prepare_lstm_int_weights" in SOURCE
    assert "int8_lstm_row_dot_cached" in SOURCE
    assert "std::vector<int8_t> input_q" in SOURCE
    assert "CRISPEMBED_TESSERACT_DISABLE_INT_CACHE" in SOURCE


if __name__ == "__main__":
    test_int_mode_cache_is_present()
    print("tesseract kernel contract: PASS")
