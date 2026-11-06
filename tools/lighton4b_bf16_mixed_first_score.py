#!/usr/bin/env python3
"""Exploratory fixed attention/embeddingQ8, FFNQ4 original BF16 first position."""
from pathlib import Path
import lighton4b_bf16_quant_first_score as original
original.PROTOCOL=Path(__file__).resolve().parents[1]/'docs/lighton4b-bf16-mixed-first-score-v1.json'
if __name__=='__main__':raise SystemExit(original.main())
