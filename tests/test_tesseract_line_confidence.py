#!/usr/bin/env python3
"""Model-free tests for Tesseract line-confidence acceptance semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_tesseract_line_confidence import confidence_acceptance_checks  # noqa: E402


class TesseractLineConfidenceTest(unittest.TestCase):
    def test_greedy_delta_and_beam_contract_pass(self) -> None:
        args = type("Args", (), {"max_greedy_word_confidence_delta": 0.01, "require_beam_sequence_only": True})()
        checks = confidence_acceptance_checks(
            args,
            {"mean_word_confidence": 0.96},
            {"word_confidence": 0.965},
            {"char_confidences": 0, "word_confidence": 0.0},
        )
        self.assertEqual(checks, {"max_greedy_word_confidence_delta": True, "beam_sequence_only": True})

    def test_confidence_gate_rejects_calibration_drift(self) -> None:
        args = type("Args", (), {"max_greedy_word_confidence_delta": 0.01, "require_beam_sequence_only": False})()
        checks = confidence_acceptance_checks(args, {"mean_word_confidence": 0.96}, {"word_confidence": 0.98}, None)
        self.assertEqual(checks, {"max_greedy_word_confidence_delta": False})

    def test_beam_contract_rejects_character_confidences_or_missing_beam(self) -> None:
        args = type("Args", (), {"max_greedy_word_confidence_delta": None, "require_beam_sequence_only": True})()
        self.assertEqual(confidence_acceptance_checks(args, {}, {}, {"char_confidences": 2}), {"beam_sequence_only": False})
        self.assertEqual(
            confidence_acceptance_checks(args, {}, {}, {"char_confidences": 0, "word_confidence": 0.1}),
            {"beam_sequence_only": False},
        )
        self.assertEqual(confidence_acceptance_checks(args, {}, {}, None), {"beam_sequence_only": False})

    def test_official_word_gate_rejects_empty_reference(self) -> None:
        args = type("Args", (), {
            "max_greedy_word_confidence_delta": None,
            "require_beam_sequence_only": False,
            "require_official_words": True,
        })()
        self.assertEqual(
            confidence_acceptance_checks(args, {"words": 0}, {}, None),
            {"official_words_present": False},
        )
        self.assertEqual(
            confidence_acceptance_checks(args, {"words": 1}, {}, None),
            {"official_words_present": True},
        )

    def test_text_gate_rejects_quality_mismatch(self) -> None:
        args = type("Args", (), {
            "max_greedy_word_confidence_delta": None,
            "require_beam_sequence_only": False,
            "require_official_words": False,
            "require_greedy_text_match": True,
        })()
        self.assertEqual(
            confidence_acceptance_checks(args, {"text": "Brighton"}, {"text": "Drighton"}, None),
            {"greedy_text_matches": False},
        )


if __name__ == "__main__":
    unittest.main()
