"""CrispEmbed — lightweight text embedding via ggml."""

from ._binding import (
    set_gpu_backend,
    accept_biometric_use,
    CrispEmbed, CrispVit, CrispFace, CrispFacePipeline, CrispClipText,
    CrispOcrModel, CrispMathOcr, CrispOcrPipeline, CrispOcrOrchestrator, CrispPreprocess,
    CrispLayout, CrispTextDetect, CrispNER, CrispKIE, CrispLiLT,
    CrispTextLID, CrispTruecaser, CrispScanCleanup, CrispTableParse,
    CrispTextSr, CrispTbsrnSr, CrispPanSr, CrispHatSr, CrispDatSr,
    CrispSafmnSr, CrispEsrganSr, CrispSwinirSr,
    CrispRestormer, CrispScunet, CrispAdaIR, CrispInstructIR,
    CrispPix2Struct, CrispGraniteVision, CrispLightOnOCR,
)

__all__ = [
    "set_gpu_backend",
    "accept_biometric_use",
    "CrispEmbed", "CrispVit", "CrispFace", "CrispFacePipeline", "CrispClipText",
    "CrispOcrModel", "CrispMathOcr", "CrispOcrPipeline", "CrispOcrOrchestrator", "CrispPreprocess",
    "CrispLayout", "CrispTextDetect", "CrispNER", "CrispKIE", "CrispLiLT",
    "CrispTextLID", "CrispTruecaser", "CrispScanCleanup", "CrispTableParse",
    "CrispTextSr", "CrispTbsrnSr", "CrispPanSr", "CrispHatSr", "CrispDatSr",
    "CrispSafmnSr", "CrispEsrganSr", "CrispSwinirSr",
    "CrispRestormer", "CrispScunet", "CrispAdaIR", "CrispInstructIR",
    "CrispPix2Struct", "CrispGraniteVision", "CrispLightOnOCR",
]
# Tracks /VERSION (the C library version). Wheel CI copies the freshly built
# libcrispembed.{so,dylib,dll} + ggml siblings alongside this file.
__version__ = "0.3.2"
