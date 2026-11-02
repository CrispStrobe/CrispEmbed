"""Native Marker-style PDF/image conversion; no Marker Python runtime required."""
import ctypes
import json
import os
import threading
from pathlib import Path

import numpy as np
from ._binding import _load_library


class _Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p), ("pdfium_path", ctypes.c_char_p),
        ("password", ctypes.c_char_p),
        *[(name, ctypes.c_int) for name in (
            "mode", "n_threads", "first_page", "last_page", "dpi", "max_tokens",
            "max_layout_tokens", "max_blocks", "max_pages")],
        ("max_pixels", ctypes.c_uint64), ("max_asset_bytes", ctypes.c_uint64),
        ("include_page_headers", ctypes.c_int), ("extract_images", ctypes.c_int),
    ]


def _encode(value, name):
    value = os.fspath(value)
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"{name} must be a NUL-free string")
    return value.encode("utf-8")


class CrispMarker:
    """C++ document processing with native PDFium and TeleOCR GGUF inference.

    ``mode='text'`` extracts a PDF's digital text without any model. ``auto``
    combines PDF text with native TeleOCR layout/OCR; ``ocr`` always recognizes
    page regions. Requires ``CRISPEMBED_MARKER=1`` (and the existing TeleOCR gate
    in auto/ocr mode). Explicit local paths only; never downloads dependencies.
    This implementation does not claim original Surya model/output parity.
    """
    def __init__(self, model_path=None, *, pdfium_path=None, mode="auto", n_threads=4,
                 first_page=0, last_page=-1, dpi=144, max_tokens=1024,
                 max_layout_tokens=2048, max_blocks=512, max_pages=1000,
                 max_pixels=16000000, max_asset_bytes=128 * 1024 * 1024,
                 include_page_headers=False, extract_images=True, password="", lib_path=None):
        self._ctx = None
        self._lock = threading.RLock()
        if mode not in ("auto", "text", "ocr"):
            raise ValueError("mode must be auto, text or ocr")
        lib = self._lib = _load_library(lib_path)
        try:
            lib.crispembed_marker_default_config.restype = _Config
        except AttributeError as exc:
            raise RuntimeError("Native library does not provide Marker support") from exc
        lib.crispembed_marker_default_config.argtypes = []
        signatures = {
            "init": ([ctypes.POINTER(_Config)], ctypes.c_void_p),
            "free": ([ctypes.c_void_p], None),
            "convert": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
            "convert_raw": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                             ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "render": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_char_p),
            "metadata": ([ctypes.c_void_p], ctypes.c_char_p),
            "error": ([ctypes.c_void_p], ctypes.c_char_p),
            "asset_count": ([ctypes.c_void_p], ctypes.c_int),
            "asset_name": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_char_p),
            "asset_data": ([ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
                           ctypes.POINTER(ctypes.c_uint8)),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(lib, "crispembed_marker_" + name)
            fn.argtypes, fn.restype = args, result
        cfg = lib.crispembed_marker_default_config()
        cfg.mode = ("auto", "text", "ocr").index(mode)
        for name, value in dict(n_threads=n_threads, first_page=first_page, last_page=last_page,
                                dpi=dpi, max_tokens=max_tokens, max_layout_tokens=max_layout_tokens,
                                max_blocks=max_blocks, max_pages=max_pages, max_pixels=max_pixels,
                                max_asset_bytes=max_asset_bytes).items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if name in ("max_pixels", "max_asset_bytes"):
                if not 0 <= value <= (1 << 64) - 1:
                    raise ValueError(f"{name} is out of range")
            elif not -(1 << 31) <= value < (1 << 31):
                raise ValueError(f"{name} is out of range")
            setattr(cfg, name, value)
        if not isinstance(include_page_headers, bool) or not isinstance(extract_images, bool):
            raise ValueError("include_page_headers/extract_images must be bool")
        cfg.include_page_headers, cfg.extract_images = include_page_headers, extract_images
        cfg.model_path = _encode(model_path or "", "model_path")
        cfg.pdfium_path = _encode(pdfium_path or "", "pdfium_path")
        cfg.password = _encode(password, "password")
        self._max_pixels = max_pixels
        self._ctx = lib.crispembed_marker_init(ctypes.byref(cfg))
        if not self._ctx:
            raise RuntimeError(self._error())

    def _error(self):
        value = self._lib.crispembed_marker_error(self._ctx)
        return value.decode("utf-8", errors="replace") if value else "Marker operation failed"

    def _require_open(self):
        if not self._ctx:
            raise RuntimeError("Marker context is closed")

    def convert(self, path, output_format="markdown"):
        """Convert a PDF or image, returning the requested UTF-8 output string."""
        with self._lock:
            self._require_open()
            if not self._lib.crispembed_marker_convert(self._ctx, _encode(path, "path")):
                raise RuntimeError(self._error())
            return self.render(output_format)

    def convert_image(self, image, output_format="markdown"):
        """Convert a uint8 gray/RGB/RGBA NumPy array or a PIL image."""
        if hasattr(image, "convert") and hasattr(image, "size"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError("image must be a uint8 gray/RGB/RGBA array")
        channels = 1 if array.ndim == 2 else array.shape[2]
        height, width = array.shape[:2]
        if channels not in (1, 3, 4) or width < 1 or height < 1 or width * height > self._max_pixels:
            raise ValueError("invalid image shape or pixel limit exceeded")
        array = np.ascontiguousarray(array)
        with self._lock:
            self._require_open()
            if not self._lib.crispembed_marker_convert_raw(
                    self._ctx, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    width, height, channels):
                raise RuntimeError(self._error())
            return self.render(output_format)

    def render(self, output_format="markdown"):
        """Render the last successful conversion; formats: markdown/html/json/chunks."""
        with self._lock:
            self._require_open()
            data = self._lib.crispembed_marker_render(self._ctx, _encode(output_format, "output_format"))
            if data is None:
                raise RuntimeError(self._error())
            return data.decode("utf-8")

    @property
    def metadata(self):
        with self._lock:
            self._require_open()
            data = self._lib.crispembed_marker_metadata(self._ctx)
            if data is None:
                raise RuntimeError(self._error())
            return json.loads(data)

    @property
    def assets(self):
        """Return independent copies of extracted image assets by safe basename."""
        with self._lock:
            self._require_open()
            result = {}
            for index in range(self._lib.crispembed_marker_asset_count(self._ctx)):
                name = self._lib.crispembed_marker_asset_name(self._ctx, index).decode("utf-8")
                size = ctypes.c_size_t()
                ptr = self._lib.crispembed_marker_asset_data(self._ctx, index, ctypes.byref(size))
                result[name] = ctypes.string_at(ptr, size.value)
            return result

    def save(self, path, output_format="markdown"):
        """Save the last conversion, its metadata and extracted image assets."""
        with self._lock:
            path = Path(path)
            text, meta, assets = self.render(output_format), self.metadata, self.assets
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            path.with_name(path.stem + "_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for name, data in assets.items():
                if Path(name).name != name or name in (".", ".."):
                    raise RuntimeError("unsafe native asset filename")
                (path.parent / name).write_bytes(data)

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.crispembed_marker_free(self._ctx)
                self._ctx = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
