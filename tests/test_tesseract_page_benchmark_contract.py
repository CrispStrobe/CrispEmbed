from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE = (ROOT / "tools/compare_tesseract_page_metrics.py").read_text()
BENCHMARK = (ROOT / "tools/benchmark_tesseract_page.py").read_text()


def test_page_comparator_exposes_decoder_experiments():
    assert '"--recode-beam"' in COMPARE
    assert '"--dawg-score"' in COMPARE
    assert '"--compose"' in COMPARE
    assert '"CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH"' in COMPARE
    assert '"CRISPEMBED_TESSERACT_DAWG_SCORE"' in COMPARE
    assert '"CRISPEMBED_TESSERACT_RECODE_COMPOSE"' in COMPARE


def test_page_benchmark_preserves_exact_output_pair():
    assert '"official_text"' in BENCHMARK
    assert '"native_text"' in BENCHMARK
    assert '"official_lines"' in BENCHMARK
    assert '"native_regions"' in BENCHMARK
    assert '"identical"' in BENCHMARK

