"""CrispEmbed — lightweight text embedding via ggml."""

from ._binding import CrispEmbed, CrispVit, CrispFace, CrispFacePipeline, CrispClipText, CrispMathOcr, CrispOcrPipeline, CrispLayout, CrispTextDetect, CrispNER, CrispScanCleanup

__all__ = ["CrispEmbed", "CrispVit", "CrispFace", "CrispFacePipeline", "CrispClipText", "CrispMathOcr", "CrispOcrPipeline", "CrispLayout", "CrispTextDetect", "CrispNER", "CrispScanCleanup"]
# Tracks /VERSION (the C library version). Wheel CI copies the freshly built
# libcrispembed.{so,dylib,dll} + ggml siblings alongside this file.
__version__ = "0.3.2"
