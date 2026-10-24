#!/usr/bin/env python3
"""Hosted model-backed check that a second Ovis stage loads its own GGUF path."""
import argparse
import ctypes
import json
from pathlib import Path
import sys


class StageMetric(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_int), ("engine", ctypes.c_char_p),
        ("elapsed_ms", ctypes.c_float), ("cleanup_applied", ctypes.c_int),
        ("accepted", ctypes.c_int), ("text_chars", ctypes.c_int),
        ("mean_confidence", ctypes.c_float),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--first-model", type=Path, required=True)
    parser.add_argument("--truncated-second", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo/"python"))
    from crispembed import CrispOcrOrchestrator

    stages = [
        {"engine": 21, "model_a": str(args.first_model), "vlm_max_tokens": 12,
         "min_chars": 2147483647, "min_confidence": 0.0},
        {"engine": 21, "model_a": str(args.truncated_second), "vlm_max_tokens": 12,
         "min_chars": 1, "min_confidence": 0.0},
    ]
    pipe = CrispOcrOrchestrator(stages=stages, router=False, n_threads=2,
                               lib_path=str(args.lib))
    try:
        result = pipe.run(str(args.image))
        dll = pipe._lib
        dll.crispembed_ocr_pipeline_stage_metrics.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        dll.crispembed_ocr_pipeline_stage_metrics.restype = ctypes.POINTER(StageMetric)
        count = ctypes.c_int()
        ptr = dll.crispembed_ocr_pipeline_stage_metrics(pipe._ctx, ctypes.byref(count))
        metrics = [
            {"index": row.index, "engine": row.engine.decode() if row.engine else "",
             "accepted": bool(row.accepted), "text_chars": row.text_chars,
             "mean_confidence": float(row.mean_confidence)}
            for row in (ptr[index] for index in range(count.value))
        ] if ptr else []
        print("RESULT stage-cache " + json.dumps({
            "stages_tried": count.value, "metrics": metrics,
            "fallback_text": result["text"], "fallback_n_regions": result["n_regions"],
        }, ensure_ascii=True), flush=True)
    finally:
        pipe.__del__()


if __name__ == "__main__":
    main()
