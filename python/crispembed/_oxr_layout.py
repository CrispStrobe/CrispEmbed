"""Native OXR-1.0 D-FINE layout detector (layout_detect ``dfine`` architecture).

``CrispOXRLayout.analyze`` returns exactly the OXR layout service's
``/inference`` response body (``{"status", "params", "data": {"labels",
"boxes", "scores"}}``) for a batch of pages, so it can be passed to OXR's
``parse_layout_batch`` or ``CrispOXR.process_pages(layout_response=...)``.
Requires ``CRISPEMBED_OXR=1`` and ``CRISPEMBED_ACCEPT_OXR_LICENSE=1``; the
weights are under the OXR License (see licenses/oxr-LICENSE).
"""

from __future__ import annotations

import ctypes as C
import json
import math
import os
import threading
from typing import Any, Sequence

import numpy as np

from ._binding import _load_library

LAYOUT_CLASSES = (
    "Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
    "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure",
)


def _rgb(image) -> np.ndarray:
    from PIL import Image
    if isinstance(image, (str, os.PathLike)):
        with Image.open(os.fspath(image)) as opened:
            return np.ascontiguousarray(opened.convert("RGB"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise TypeError("layout images must be a path, PIL.Image or uint8 HxW/HxWx3/HxWx4 array")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("layout array channel count must be 3 or 4")
    return np.ascontiguousarray(image)


class CrispOXRLayout:
    """Resident native D-FINE layout model; calls are serialized per context."""

    def __init__(self, model_path: str | os.PathLike[str], n_threads: int = 4, lib_path: str | None = None):
        self._ctx = None
        self._lock = threading.Lock()
        path = os.fspath(model_path)
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("OXR layout requires a nonempty NUL-free local GGUF path")
        if type(n_threads) is not int or not 1 <= n_threads <= 256:
            raise ValueError("n_threads must be an integer in 1..256")
        self._lib = _load_library(lib_path)
        for suffix, args, result in (
            ("init", [C.c_char_p, C.c_int], C.c_void_p),
            ("free", [C.c_void_p], None),
            ("detect", [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int, C.c_int, C.c_float], C.c_char_p),
            ("timings", [C.c_void_p], C.c_char_p),
            ("last_error", [C.c_void_p], C.c_char_p),
        ):
            try:
                fn = getattr(self._lib, "crispembed_oxr_layout_" + suffix)
            except AttributeError as error:
                raise RuntimeError("Native library lacks the OXR layout API; build this source revision") from error
            fn.argtypes, fn.restype = args, result
        self._ctx = self._lib.crispembed_oxr_layout_init(path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(self._error())
        self.last_timings: list[dict[str, float]] = []

    def _error(self) -> str:
        raw = self._lib.crispembed_oxr_layout_last_error(self._ctx)
        return raw.decode("utf-8", errors="replace") if raw else "OXR layout operation failed"

    def detect(self, image, confidence: float | None = None) -> dict[str, Any]:
        """One page: the service response for a one-image batch."""
        return self.analyze([image], confidence)

    def analyze(self, images: Sequence[Any], confidence: float | None = None) -> dict[str, Any]:
        """Batch of RGB pages -> service response (labels/boxes/scores per page)."""
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be a finite number between 0 and 1")
        threshold = float("nan") if confidence is None else float(confidence)
        labels, boxes, scores, params = [], [], [], {}
        self.last_timings = []
        with self._lock:
            if not self._ctx:
                raise RuntimeError("OXR layout context is closed")
            for image in images:
                pixels = _rgb(image)
                height, width = pixels.shape[:2]
                channels = 1 if pixels.ndim == 2 else pixels.shape[2]
                raw = self._lib.crispembed_oxr_layout_detect(
                    self._ctx, pixels.ctypes.data_as(C.POINTER(C.c_uint8)), width, height, channels, threshold)
                if raw is None:
                    raise RuntimeError(self._error())
                page = json.loads(raw.decode("utf-8"))
                params = page["params"]
                labels.append(page["data"]["labels"][0])
                boxes.append(page["data"]["boxes"][0])
                scores.append(page["data"]["scores"][0])
                self.last_timings.append(json.loads(self._lib.crispembed_oxr_layout_timings(self._ctx)))
        return {"status": "success", "params": params, "data": {"labels": labels, "boxes": boxes, "scores": scores}}

    def regions(self, image, score_threshold: float = 0.5, nms_iou_threshold: float | None = 0.6) -> list[dict]:
        """OXR client-side selection (threshold, clip, class-wise NMS) as region dicts."""
        response = self.detect(image)
        pixels = _rgb(image)
        height, width = pixels.shape[:2]
        data = response["data"]
        candidates = []
        for index, (label, box, score) in enumerate(zip(data["labels"][0], data["boxes"][0], data["scores"][0])):
            if score < score_threshold or box[0] >= box[2] or box[1] >= box[3]:
                continue
            x1, y1 = min(max(box[0], 0.0), float(width)), min(max(box[1], 0.0), float(height))
            x2, y2 = min(max(box[2], 0.0), float(width)), min(max(box[3], 0.0), float(height))
            if x1 < x2 and y1 < y2:
                candidates.append((index, label, [x1, y1, x2, y2], score))
        if nms_iou_threshold is not None:
            def iou(a, b):
                w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = w * h
                if inter == 0.0:
                    return 0.0
                return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            kept = []
            for cand in sorted(candidates, key=lambda c: (-c[3], c[0])):
                if not any(k[1] == cand[1] and iou(cand[2], k[2]) > nms_iou_threshold for k in kept):
                    kept.append(cand)
            keep = {k[0] for k in kept}
            candidates = [c for c in candidates if c[0] in keep]
        return [{"type": LAYOUT_CLASSES[label], "bbox": box, "score": score} for _, label, box, score in candidates]

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.crispembed_oxr_layout_free(self._ctx)
                self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
