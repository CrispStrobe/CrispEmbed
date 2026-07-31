#!/usr/bin/env python3
"""Guard the explicit OCR backend capability claims."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/ocr_backend_matrix.md"
PPOCR = ROOT / "docs/ppocrv6.md"


def main() -> int:
    text = MATRIX.read_text()
    required = (
        "PP-OCRv6 detector/recognizer", "PP-LCNet orientation", "DBNet + TrOCR",
        "Tesseract-LSTM", "PARSeq", "Surya", "GOT/GLM/Qwen/InternVL/DeepSeek VLMs",
        "Unlimited-OCR", "SmolDocling", "PP-FormulaNet / MixTeX",
        "HMER / BTTR / PosFormer", "SMT / SMT++ / Polyphonic-TrOMR / Transcoda",
    )
    for name in required:
        assert name in text, f"missing backend matrix row: {name}"
    assert "PP-OCRv6 remains explicitly CPU-only" in PPOCR.read_text()
    assert "GGML_METAL=OFF" in text and "GGML_CUDA=OFF" in text
    print(f"OCR backend matrix OK: {len(required)} required families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
