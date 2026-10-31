"""Native Marker-style PDF/image conversion; no Marker Python runtime required."""
import ctypes
import json
import os
import stat
import tempfile
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


def _destination(path):
    """Check the target and existing parent chain without following symlinks."""
    path = Path(path)
    if not path.name or path.name in (".", "..") or "\0" in str(path):
        raise RuntimeError("unsafe Marker output filename")
    absolute = path if path.is_absolute() else Path.cwd() / path
    for candidate in (absolute, *absolute.parents):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError("cannot inspect Marker output destination") from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError("Marker output destinations must not use symlinks")
        if candidate == absolute:
            if not stat.S_ISREG(mode):
                raise RuntimeError("Marker output destination is not a regular file")
        elif not stat.S_ISDIR(mode):
            raise RuntimeError("Marker output parent is not a directory")
    return absolute


def _same_destination(left, right):
    if os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right)):
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _asset_matches(path, data):
    """A size-first, bounded comparison; never write an existing asset."""
    path = _destination(path)
    try:
        if path.stat().st_size != len(data):
            return False
        with path.open("rb") as stream:
            for first in range(0, len(data), 65536):
                if stream.read(min(65536, len(data) - first)) != data[first:first + 65536]:
                    return False
            return not stream.read(1)
    except FileNotFoundError:
        return False


def _atomic_output(path, data):
    path = _destination(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".marker-save-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        _destination(path)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _save_bundle(path, text, metadata, assets):
    """Preflight the whole bundle before changing its output or metadata."""
    path = _destination(path)
    meta_path = _destination(path.with_name(path.stem + "_meta.json"))
    if _same_destination(path, meta_path):
        raise RuntimeError("Marker output and metadata destinations collide")
    planned = []
    for name, data in assets.items():
        if (not isinstance(name, str) or not name or len(name) > 255 or name in (".", "..") or name.endswith(".") or
                any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name)):
            raise RuntimeError("unsafe native asset filename")
        stem = name.split(".", 1)[0].upper()
        if stem in ("CON", "PRN", "AUX", "NUL") or (len(stem) == 4 and stem[:3] in ("COM", "LPT") and stem[3] in "123456789"):
            raise RuntimeError("unsafe native asset filename")
        target = _destination(path.parent / name)
        if _same_destination(target, path) or _same_destination(target, meta_path):
            raise RuntimeError("Marker asset collides with output or metadata; use a separate directory per document")
        if any(_same_destination(target, other) for other, _ in planned):
            raise RuntimeError("Marker asset destinations collide; use a separate directory per document")
        if target.exists() and not _asset_matches(target, data):
            raise RuntimeError("Marker asset already exists with different bytes; use a separate directory per document")
        planned.append((target, data))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create assets exclusively. Identical assets are reused, preserving their
    # bytes, permissions and timestamps. A concurrent conflicting save fails
    # before the primary output or metadata is changed.
    for target, data in planned:
        _destination(target)
        try:
            stream = target.open("xb")
        except FileExistsError:
            if not _asset_matches(target, data):
                raise RuntimeError("Marker asset changed during save; use a separate directory per document")
            continue
        try:
            with stream:
                stream.write(data)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    _atomic_output(meta_path, metadata)
    _atomic_output(path, text)


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
            # Palette and RGB PNG tRNS transparency is metadata, not an A band.
            # Let Pillow resolve it before the native white-compositing step.
            image = image.convert("RGBA")
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
        """Save output and metadata, reusing only byte-identical existing assets.

        Conflicting assets, symlinks and destination aliases are rejected before
        output is written. Use a separate directory for each document.
        """
        with self._lock:
            text, meta, assets = self.render(output_format), self.metadata, self.assets
            _save_bundle(path, text.encode("utf-8"),
                         (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), assets)

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
