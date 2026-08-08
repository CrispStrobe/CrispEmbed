#!/usr/bin/env python3
"""Guard the explicit OCR backend capability claims."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/ocr_backend_matrix.md"
PPOCR = ROOT / "docs/ppocrv6.md"


def main() -> int:
    text = MATRIX.read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] != "Engine family":
            rows.append(cells)
    assert rows, "backend matrix has no data rows"
    for row in rows:
        assert len(row) == 4, f"malformed backend matrix row: {row!r}"
        assert all(row), f"blank backend matrix field: {row!r}"
        # "Yes on CUDA" is the per-backend-kind default the pix2struct decode
        # graph landed (2026-08-08, merged 69e39a62): CUDA gets the graph, Metal
        # and CPU keep the scalar path. The guard rejected it, so this test has
        # been red on main since that row was written.
        assert row[2] in {
            "No",
            "Partial",
            "Yes on CUDA",
            "Yes, when backend enabled",
        }, f"unsupported capability value: {row[2]}"
    required = (
        "PP-OCRv6 detector/recognizer", "PP-LCNet orientation", "DBNet + TrOCR",
        "Tesseract-LSTM", "PARSeq", "Surya", "GOT/GLM/Qwen/InternVL/DeepSeek VLMs",
        "Unlimited-OCR", "SmolDocling", "PP-FormulaNet / MixTeX",
        "HMER / BTTR / PosFormer", "SMT / SMT++ / Polyphonic-TrOMR / Transcoda",
    )
    for name in required:
        assert name in text, f"missing backend matrix row: {name}"
    ppocr_text = PPOCR.read_text()
    assert "CPU fallback" in ppocr_text and "diagnostic-only" in ppocr_text
    assert "GGML_METAL=OFF" in text and "GGML_CUDA=OFF" in text
    # Keep the partial VLM claims tied to concrete, source-audited CPU seams.
    # These guards are intentionally textual: changing a seam requires updating
    # the capability matrix in the same change instead of silently overstating
    # GPU residency.
    for marker in (
        "CPU window partition",
        "CPU spatial merge",
        "UOCR_SAM_CONV_CPU",
        "UOCR_MOE_CPU",
    ):
        assert marker in text, f"VLM residency boundary missing from matrix: {marker}"
    print(f"OCR backend matrix OK: {len(required)} required families, {len(rows)} schema-valid rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
