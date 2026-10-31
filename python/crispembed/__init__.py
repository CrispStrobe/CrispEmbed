"""CrispEmbed — lightweight text embedding via ggml."""

from ._binding import (
    set_gpu_backend,
    set_offline,
    is_offline,
    accept_biometric_use,
    CrispEmbed, CrispVit, CrispFace, CrispFacePipeline, CrispClipText,
    CrispOcrModel, CrispMathOcr, CrispOcrPipeline, CrispOcrOrchestrator, CrispPreprocess, OCR_ENGINE_TELEOCR,
    CrispLayout, CrispTextDetect, CrispNER, CrispKIE, CrispLiLT,
    CrispTextLID, CrispTruecaser, CrispScanCleanup, CrispTableParse,
    CrispTextSr, CrispTbsrnSr, CrispPanSr, CrispHatSr, CrispDatSr,
    CrispSafmnSr, CrispEsrganSr, CrispSwinirSr,
    CrispRestormer, CrispScunet, CrispAdaIR, CrispInstructIR,
    CrispPix2Struct, CrispGraniteVision, CrispLightOnOCR, CrispSam2,
)
from ._moedern_text import CrispMoEdern
from ._mineru import CrispMinerU
from ._oxr import CrispOXR

from ._systemone import CrispSystemOne

__all__ = [
    "CrispSystemOne",
    "CrispMoEdern",
    "CrispMinerU",
    "CrispOXR",
    "set_gpu_backend",
    "set_offline",
    "is_offline",
    "accept_biometric_use",
    "CrispEmbed", "CrispVit", "CrispFace", "CrispFacePipeline", "CrispClipText",
    "CrispOcrModel", "CrispMathOcr", "CrispOcrPipeline", "CrispOcrOrchestrator", "CrispPreprocess", "OCR_ENGINE_TELEOCR",
    "CrispLayout", "CrispTextDetect", "CrispNER", "CrispKIE", "CrispLiLT",
    "CrispTextLID", "CrispTruecaser", "CrispScanCleanup", "CrispTableParse",
    "CrispTextSr", "CrispTbsrnSr", "CrispPanSr", "CrispHatSr", "CrispDatSr",
    "CrispSafmnSr", "CrispEsrganSr", "CrispSwinirSr",
    "CrispRestormer", "CrispScunet", "CrispAdaIR", "CrispInstructIR",
    "CrispPix2Struct", "CrispGraniteVision", "CrispLightOnOCR", "CrispSam2",
]
# Tracks /VERSION (the C library version). Wheel CI copies the freshly built
# libcrispembed.{so,dylib,dll} + ggml siblings alongside this file.
__version__ = "0.3.2"
