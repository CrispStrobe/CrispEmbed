"""MinerU2.5-Pro native GGML inference with the pinned official page pipeline.

``predict`` and ``native_extract`` need only NumPy/Pillow and the native library.
``two_step_extract`` delegates image/layout/postprocessing to mineru-vl-utils;
it never loads a Python model or forwards inference to an external service.

The reference generation policy is the Transformers blueprint: greedy,
repetition_penalty=1, no_repeat_ngram_size=100, and both model EOS IDs.
vLLM's presence/frequency penalties and concurrent batching are a different
policy and are not claimed by this wrapper.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np

from ._binding import _load_library

BLUEPRINT_REVISION = "c9797cde8f54cc48306196fc6aebe73ea31dd4a3"
BLUEPRINT_SOURCE_SHA256 = "f0c95996cb6ae01b64f0f4fe3ae88c43bea842625ef6b7a36e06fad32b50b43b"
MODEL_REVISION = "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"


def _check_blueprint() -> None:
    """Validate source bytes, including edits to an otherwise pinned install."""
    spec = importlib.util.find_spec("mineru_vl_utils")
    if spec is None or not spec.origin:
        raise ImportError(
            "Full MinerU page processing requires the pinned helper package: "
            "pip install 'mineru-vl-utils @ git+https://github.com/opendatalab/"
            f"mineru-vl-utils.git@{BLUEPRINT_REVISION}'"
        )
    root = Path(spec.origin).parent
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    if digest.hexdigest() != BLUEPRINT_SOURCE_SHA256:
        raise RuntimeError(
            "mineru-vl-utils source differs from the pinned blueprint "
            f"{BLUEPRINT_REVISION}; install the exact commit before using this parity path"
        )


class CrispMinerU:
    """Resident MinerU2.5-Pro context; serialized, synchronous native inference.

    The native gate ``CRISPEMBED_MINERU25_PRO=1`` must be set before construction.
    A context owns reusable model weights/KV state; ``close`` is idempotent.
    ``two_step_extract`` returns the official ``ExtractResult`` (list of dicts).
    ``last_trace`` retains native request/result metadata for parity evaluation.

    ``max_new_tokens=0`` means the model's total-context budget, not unlimited
    generation. By default truncation raises; explicitly set
    ``allow_truncated_content=True`` only when incomplete content is acceptable.
    No vLLM throughput, quantized quality, or numerical parity is implied by
    successfully loading a model or completing a page.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        n_threads: int = 4,
        lib_path: str | None = None,
        *,
        image_analysis: bool = False,
        simple_post_process: bool = False,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        enable_table_formula_eq_wrap: bool = False,
        enable_cross_page_table_merge: bool = False,
        allow_truncated_content: bool = False,
    ) -> None:
        self._ctx = None
        self._lock = threading.RLock()
        self._helper = None
        self.last_trace: list[dict[str, Any]] = []
        path = os.fspath(model_path)
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("MinerU requires a nonempty NUL-free local GGUF path")
        if type(n_threads) is not int or not 1 <= n_threads <= 256:
            raise ValueError("MinerU n_threads must be an integer in 1..256")
        options = dict(
            image_analysis=image_analysis,
            simple_post_process=simple_post_process,
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            enable_table_formula_eq_wrap=enable_table_formula_eq_wrap,
            enable_cross_page_table_merge=enable_cross_page_table_merge,
        )
        if any(type(v) is not bool for v in (*options.values(), allow_truncated_content)):
            raise TypeError("MinerU processing switches must be bool")
        self._helper_options = options
        self.allow_truncated_content = allow_truncated_content
        self._lib = _load_library(lib_path)
        signatures = (
            ("init", [C.c_char_p, C.c_int], C.c_void_p),
            ("free", [C.c_void_p], None),
            ("predict", [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int], C.c_char_p),
            ("extract", [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int, C.c_int, C.c_int, C.c_int], C.c_char_p),
            ("last_markdown", [C.c_void_p], C.c_char_p),
            ("extract_document", [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int], C.c_char_p),
            ("last_document", [C.c_void_p, C.c_char_p], C.c_char_p),
            ("last_error", [C.c_void_p], C.c_char_p),
        )
        for suffix, args, result in signatures:
            try:
                fn = getattr(self._lib, "crispembed_mineru_" + suffix)
            except AttributeError as error:
                raise RuntimeError("Native library lacks MinerU support; build this source revision") from error
            fn.argtypes, fn.restype = args, result
        self._ctx = self._lib.crispembed_mineru_init(path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        value = self._lib.crispembed_mineru_last_error(self._ctx)
        return value.decode("utf-8", errors="replace") if value else "MinerU operation failed"

    def _require_open(self) -> None:
        if not self._ctx:
            raise RuntimeError("MinerU context is closed")

    @staticmethod
    def _budget(value: int) -> int:
        if type(value) is not int or not 0 <= value <= 8192:
            raise ValueError("max_new_tokens must be an integer in 0..8192 (0 uses the context budget)")
        return value

    @staticmethod
    def _pil_image(image: Any):
        from PIL import Image

        if isinstance(image, Image.Image):
            # The upstream get_rgb_image first expands palette mode to RGBA
            # then converts to RGB (does not alpha-composite onto white).
            if image.mode == "P":
                image = image.convert("RGBA")
            return image if image.mode == "RGB" else image.convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            with Image.open(os.fspath(image)) as loaded:
                return CrispMinerU._pil_image(loaded).copy()
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim not in (2, 3):
                raise ValueError("MinerU arrays must be uint8 HxW or HxWx3/4")
            if image.ndim == 3 and image.shape[2] not in (3, 4):
                raise ValueError("MinerU arrays require 3 RGB or 4 RGBA channels")
            if min(image.shape[:2]) <= 0:
                raise ValueError("MinerU image must be nonempty")
            return CrispMinerU._pil_image(Image.fromarray(image))
        raise TypeError("MinerU image must be PIL.Image, uint8 NumPy array, or a local image path")

    @classmethod
    def _pixels(cls, image: Any):
        if image is None:
            return None, None, 0, 0, 0
        pil = cls._pil_image(image)
        pixels = np.ascontiguousarray(pil, dtype=np.uint8)
        height, width, channels = pixels.shape
        if width > 2**31 - 1 or height > 2**31 - 1:
            raise ValueError("MinerU image dimensions exceed the native ABI")
        return pixels, pixels.ctypes.data_as(C.POINTER(C.c_uint8)), width, height, channels

    def _decode_result(self, result: bytes | None) -> dict[str, Any]:
        if result is None:
            raise RuntimeError(self._error())
        parsed = json.loads(result.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("MinerU native result must be a JSON object")
        return parsed

    def predict(self, image: Any, prompt: str, max_new_tokens: int = 0) -> dict[str, Any]:
        """Run one exact chat request, optionally text-only with ``image=None``.

        Returned token IDs include an EOS token when one was emitted; ``text``
        retains layout/table/rotation tokens required by official postprocessing.
        This low-level method always returns cap flags; page methods reject
        capped content unless ``allow_truncated_content`` was explicitly enabled.
        """
        budget = self._budget(max_new_tokens)
        if not isinstance(prompt, str) or not prompt or "\0" in prompt:
            raise ValueError("MinerU prompt must be a nonempty NUL-free string")
        with self._lock:
            self._require_open()
            pixels, pointer, width, height, channels = self._pixels(image)
            # Keep pixels alive until the synchronous native call returns.
            raw = self._lib.crispembed_mineru_predict(
                self._ctx, pointer, width, height, channels, prompt.encode("utf-8"), budget
            )
            result = self._decode_result(raw)
            self.last_trace.append({"prompt": prompt, "width": width, "height": height, "result": result})
            return result

    def _text(self, image: Any, prompt: str, max_new_tokens: int) -> str:
        result = self.predict(image, prompt, max_new_tokens)
        if (result.get("capped", False) or not result.get("eos", False)) and not self.allow_truncated_content:
            raise RuntimeError("MinerU generation did not reach EOS; partial content is not accepted")
        text = result.get("text")
        if not isinstance(text, str):
            raise RuntimeError("MinerU native result lacks text")
        return text

    def native_extract(self, image: Any, *, image_analysis: bool | None = None,
                       max_new_tokens: int = 0) -> dict[str, Any]:
        """Native BASIC page pipeline; use ``two_step_extract`` for all repairs.

        Its JSON declares the limited processing profile. Table-image masking,
        full LaTeX repairs, image/chart parsing and cross-page merging belong to
        the pinned Python helper route, not this convenience method.
        """
        budget = self._budget(max_new_tokens)
        if image is None:
            raise ValueError("native_extract requires a page image")
        analysis = self._helper_options["image_analysis"] if image_analysis is None else image_analysis
        if type(analysis) is not bool:
            raise TypeError("image_analysis must be bool or None")
        with self._lock:
            self._require_open()
            self.last_trace = []
            pixels, pointer, width, height, channels = self._pixels(image)
            result = self._decode_result(self._lib.crispembed_mineru_extract(
                self._ctx, pointer, width, height, channels, int(analysis), budget
            ))
            if result.get("capped", False) and not self.allow_truncated_content:
                raise RuntimeError("MinerU native page extraction was capped; partial content is not accepted")
            return result

    def native_two_step_extract(self, image: Any, *, not_extract_list: list[str] | None = None,
                                image_analysis: bool | None = None, max_new_tokens: int = 0,
                                table_token_seed: int | None = None) -> list[dict[str, Any]]:
        """Complete two-step extraction entirely in C++ (no mineru-vl-utils).

        Requires ``CRISPEMBED_MINERU25_PRO_FULL_DOC=1``. Returns the upstream
        ExtractResult as plain dicts, byte-identical (as JSON) to
        ``two_step_extract`` for the same recognition outputs; the full native
        result (receipts, regions, caps) is kept in ``last_native_document``.
        ``table_token_seed`` seeds table-image tokens like ``random.seed``.
        Cross-page table merging is not part of the native route.
        """
        budget = self._budget(max_new_tokens)
        if image is None:
            raise ValueError("native_two_step_extract requires a page image")
        if image_analysis is not None and type(image_analysis) is not bool:
            raise TypeError("image_analysis must be bool or None")
        if table_token_seed is not None and (type(table_token_seed) is not int or table_token_seed < 0):
            raise ValueError("table_token_seed must be a non-negative int or None")
        if self._helper_options["enable_cross_page_table_merge"]:
            raise RuntimeError("cross-page table merging is not available in the native document route")
        opts = {k: v for k, v in self._helper_options.items() if k != "enable_cross_page_table_merge"}
        if image_analysis is not None:
            opts["image_analysis"] = image_analysis
        opts["not_extract_list"] = list(not_extract_list) if not_extract_list else None
        opts["allow_truncated_content"] = self.allow_truncated_content
        opts["table_token_seed"] = table_token_seed
        with self._lock:
            self._require_open()
            self.last_trace = []
            pixels, pointer, width, height, channels = self._pixels(image)
            result = self._decode_result(self._lib.crispembed_mineru_extract_document(
                self._ctx, pointer, width, height, channels, json.dumps(opts).encode("utf-8"), budget))
            self.last_native_document = result
            return result["blocks"]

    def native_render(self, fmt: str = "mineru-markdown") -> str:
        """Render the last native document: mineru-json, mineru-markdown (upstream
        json2md), or the Marker renderers markdown/html/json/chunks."""
        with self._lock:
            self._require_open()
            value = self._lib.crispembed_mineru_last_document(self._ctx, fmt.encode("utf-8"))
            if value is None:
                raise RuntimeError(self._error())
            return value.decode("utf-8")

    def _official_helper(self):
        if self._helper is None:
            _check_blueprint()
            try:
                from copy import deepcopy
                from mineru_vl_utils.mineru_client import (
                    DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS, MinerUClientHelper,
                )
            except ImportError as error:
                raise ImportError("Install the pinned mineru-vl-utils core dependencies for full page processing") from error
            self._helper = MinerUClientHelper(
                backend="transformers",  # preprocessing branch only; no model/client is created
                prompts=deepcopy(DEFAULT_PROMPTS),
                sampling_params=deepcopy(DEFAULT_SAMPLING_PARAMS),
                layout_image_size=(1036, 1036),
                min_image_edge=28,
                max_image_edge_ratio=50,
                debug=False,
                **self._helper_options,
            )
        return self._helper

    def _two_step(self, image: Any, not_extract_list: list[str] | None,
                  image_analysis: bool | None, budget: int):
        from mineru_vl_utils.structs import ExtractResult

        helper = self._official_helper()
        page = self._pil_image(image)
        layout_image = helper.prepare_for_layout(page)
        layout_text = self._text(layout_image, helper.prompts["[layout]"], budget)
        blocks = helper.parse_layout_output(layout_text)
        images, prompts, params, indices = helper.prepare_for_extract(page, blocks, not_extract_list, image_analysis)
        if not (len(images) == len(prompts) == len(params) == len(indices)):
            raise RuntimeError("Pinned MinerU helper returned inconsistent extraction inputs")
        for crop, prompt, sampling, index in zip(images, prompts, params, indices):
            # The reference Transformers adapter ignores presence/frequency
            # penalties, but honors these values. Reject unsupported changes.
            if (sampling.temperature != 0.0 or sampling.top_k != 1 or
                sampling.repetition_penalty != 1.0 or sampling.no_repeat_ngram_size != 100 or
                sampling.max_new_tokens is not None):
                raise RuntimeError("MinerU helper requested a generation policy unsupported by the native profile")
            blocks[index].content = self._text(crop, prompt, budget)
        return ExtractResult(helper.post_process(blocks))

    def two_step_extract(self, image: Any, *, not_extract_list: list[str] | None = None,
                         image_analysis: bool | None = None, max_new_tokens: int = 0):
        """Official single-page layout, crop, recognition and full postprocessing."""
        budget = self._budget(max_new_tokens)
        if image is None:
            raise ValueError("two_step_extract requires a page image")
        if image_analysis is not None and type(image_analysis) is not bool:
            raise TypeError("image_analysis must be bool or None")
        with self._lock:
            self._require_open()
            self._official_helper()
            self.last_trace = []
            return self._two_step(image, not_extract_list, image_analysis, budget)

    def batch_two_step_extract(self, images: Iterable[Any], *, not_extract_list: list[str] | None = None,
                               image_analysis: bool | None = None, max_new_tokens: int = 0):
        """Process pages serially in one resident context, then optional table merge.

        This method preserves input page order. It does not claim vLLM's
        continuous batching or concurrency performance.
        """
        budget = self._budget(max_new_tokens)
        if image_analysis is not None and type(image_analysis) is not bool:
            raise TypeError("image_analysis must be bool or None")
        with self._lock:
            self._require_open()
            self._official_helper()
            self.last_trace = []
            merge_module = None
            if self._helper_options["enable_cross_page_table_merge"]:
                from mineru_vl_utils.post_process import cross_page_table as merge_module

                if not merge_module._HAS_TABLE_MERGE:
                    raise RuntimeError(
                        "Cross-page table merging requires the official MinerU table_merge helpers; "
                        "the installed mineru-vl-utils package alone cannot perform it"
                    )
            results = [self._two_step(image, not_extract_list, image_analysis, budget) for image in images]
            if merge_module is not None:
                errors = []

                def batch_predict(prompts):
                    try:
                        return [self._text(None, prompt, budget) for prompt in prompts]
                    except Exception as error:
                        errors.append(error)
                        raise

                merge_module.detect_cross_page_cell_merge(results, batch_predict)
                # Official helper catches backend exceptions; do not conceal a
                # native failure from callers of the public binding.
                if errors:
                    raise RuntimeError("MinerU cross-page table inference failed") from errors[0]
            return results

    @staticmethod
    def to_markdown(blocks) -> str:
        """Official paragraph-continuation Markdown rendering."""
        _check_blueprint()
        from mineru_vl_utils.post_process import json2md

        return json2md(blocks)

    def close(self) -> None:
        with self._lock:
            if self._ctx:
                self._lib.crispembed_mineru_free(self._ctx)
                self._ctx = None

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
