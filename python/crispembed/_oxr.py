"""Native SoMark OXR inference and the pinned original document pipeline.

The native model performs recognition and reading order. Layout detection is a
separate, explicit input: supply regions/a D-FINE response or configure a layout
service. This wrapper does not claim to implement the D-FINE neural model.

The optional original pipeline is imported from verified source into a private
module namespace per context. Its crop, table, formula, reading-order repair,
deduplication and assembly functions execute unchanged. No Python OXR neural
model or vLLM server is loaded by this adapter.

OXR and its helpers have additional license terms; see licenses/oxr-LICENSE.
"""

from __future__ import annotations

import asyncio
import builtins
from copy import deepcopy
import ctypes as C
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ._binding import _load_library

BLUEPRINT_REVISION = "c1a1dcb312ee9f691404f127785510453c0d9350"
BLUEPRINT_SOURCE_SHA256 = "232ac3f2cc0e195b07622cd770d406524f964de6bf6a7e1d5a5db92eb558332f"

PROMPTS = {
    "text": "Recognize the content of the image.",
    "formula": "Recognize the formula of the image into KaTex format.",
    "table": "Recognize the table of the image into html format.",
    "code": "Recognize the code of the image into markdown code format.",
    "page": "Recognize the whole page content of the image.",
    "chemical_structure": "Recognize the chemical structure of the image into SMILES.",
    "layout": (
        "Detect the layout of the input image, layout categories include "
        "[Caption,Footnote,Formula,Text,Page-footer,Page-header,Picture,Title,Table,"
        "Stamp,Chemical Structure,Code Block]. Box format is [x1,y1,x2,y2] and "
        "output box with reading order."
    ),
}

# The exact class order in the official D-FINE response, not alphabetical order.
LAYOUT_CLASSES = (
    "Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
    "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure",
)

_PDFIUM_LOCK = threading.Lock()


def _source_snapshot(path: str | os.PathLike[str] | None):
    if path is None:
        spec = importlib.util.find_spec("oxr")
        if spec is None or not spec.origin:
            raise ImportError(
                "Full OXR document processing requires the pinned helper: "
                "pip install 'somark-oxr @ git+https://github.com/SoMarkAI/"
                f"OXR.git@{BLUEPRINT_REVISION}'"
            )
        root = Path(spec.origin).parent
    else:
        root = Path(path).resolve()
        if (root / "oxr" / "__init__.py").is_file():
            root = root / "oxr"
    if not (root / "__init__.py").is_file():
        raise ValueError("blueprint_path must contain the original oxr Python package")
    digest = hashlib.sha256()
    sources = {}
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(root).as_posix()
        contents = file.read_bytes()
        sources[relative] = contents
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    if digest.hexdigest() != BLUEPRINT_SOURCE_SHA256:
        raise RuntimeError(
            "OXR helper source differs from the pinned blueprint "
            f"{BLUEPRINT_REVISION}; source drift must be audited explicitly"
        )
    return root, sources


def _source_root(path: str | os.PathLike[str] | None) -> Path:
    return _source_snapshot(path)[0]


class _Blueprint:
    """Execute original module bytes with private internal import resolution.

    Upstream uses module-level clients/settings. A process-wide monkeypatch
    would interfere with other native contexts or an ordinary OXR installation.
    Each adapter instead has its own modules; ordinary ``oxr.*`` entries in
    sys.modules are neither replaced nor mutated.
    """

    def __init__(self, path: str | os.PathLike[str] | None):
        self.root, self.sources = _source_snapshot(path)
        self.prefix = "_crispembed_oxr_" + uuid4().hex
        self.lock = threading.RLock()
        package = types.ModuleType(self.prefix)
        package.__path__ = []
        package.__package__ = self.prefix
        sys.modules[self.prefix] = package

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        resolved = name
        if level:
            package = (globals or {}).get("__package__", "")
            resolved = importlib.util.resolve_name("." * level + name, package)
            if resolved.startswith(self.prefix + "."):
                resolved = resolved[len(self.prefix) + 1:]
        if resolved == "oxr" or resolved.startswith("oxr."):
            module = self.load(resolved)
            for item in fromlist or ():
                if item != "*" and not hasattr(module, item):
                    self.load(resolved + "." + item)
            return module if fromlist else self.load("oxr")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, name: str):
        with self.lock:
            qualified = self.prefix + "." + name
            existing = sys.modules.get(qualified)
            if existing is not None:
                return existing
            if name != "oxr" and not name.startswith("oxr."):
                raise ImportError("Only OXR modules belong to the private blueprint")
            relative = name.split(".")[1:]
            base = self.root.joinpath(*relative)
            package_relative = "/".join([*relative, "__init__.py"])
            is_package = package_relative in self.sources
            source = base / "__init__.py" if is_package else base.with_suffix(".py")
            source_relative = source.relative_to(self.root).as_posix()
            if source_relative not in self.sources:
                raise ImportError(f"Pinned OXR helper has no module {name}")
            parent_name, _, child_name = name.rpartition(".")
            parent = self.load(parent_name) if parent_name else sys.modules[self.prefix]
            spec = importlib.util.spec_from_file_location(
                qualified, source,
                submodule_search_locations=[str(base)] if is_package else None,
            )
            module = importlib.util.module_from_spec(spec)
            namespace = dict(vars(builtins))
            namespace["__import__"] = self._import
            module.__dict__["__builtins__"] = namespace
            sys.modules[qualified] = module
            setattr(parent, child_name, module)
            try:
                # Compile the verified source itself; do not accept a stale pyc.
                exec(compile(self.sources[source_relative], str(source), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(qualified, None)
                if getattr(parent, child_name, None) is module:
                    delattr(parent, child_name)
                raise
            return module

    def close(self):
        with self.lock:
            for name in list(sys.modules):
                if name == self.prefix or name.startswith(self.prefix + "."):
                    del sys.modules[name]


class CrispOXR:
    """Resident native OXR model plus optional original document processing.

    Set ``CRISPEMBED_OXR=1`` and, after reviewing the OXR License, ``CRISPEMBED_ACCEPT_OXR_LICENSE=1`` before construction. ``predict`` returns native
    diagnostics and never conceals truncation. ``process_pages`` and
    ``process_document`` retain the original capped-output recovery but reject
    requests still capped afterwards unless ``allow_truncated_content=True``.

    NumPy inputs are RGB uint8, matching Pillow. Upstream page helpers operate
    in BGR; conversion is performed explicitly at that boundary. Native model
    calls are serialized in one reusable context; this is not vLLM batching.
    """

    def __init__(
        self, model_path: str | os.PathLike[str], n_threads: int = 4,
        lib_path: str | None = None, *,
        blueprint_path: str | os.PathLike[str] | None = None,
        layout_url: str | None = None, layout_model: str | os.PathLike[str] | None = None,
        max_new_tokens: int = 8192,
        layout_score_threshold: float = 0.5,
        layout_nms_iou_threshold: float | None = 0.6,
        layout_provides_reading_order: bool = False,
        table_image_placeholders: bool = False,
        full_overlap_dedupe: bool = False,
        image_output_dir: str | os.PathLike[str] = "outputs",
        allow_truncated_content: bool = False,
    ):
        self._ctx = None
        self._lock = threading.RLock()
        self._blueprint = None
        self._orchestrator = None
        self._blueprint_path = blueprint_path
        self.last_trace: list[dict[str, Any]] = []
        self._last_outcomes: dict[Any, dict[str, Any]] = {}
        path = os.fspath(model_path)
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("OXR requires a nonempty NUL-free local GGUF path")
        if type(n_threads) is not int or not 1 <= n_threads <= 256:
            raise ValueError("OXR n_threads must be an integer in 1..256")
        self.max_new_tokens = self._budget(max_new_tokens)
        if self.max_new_tokens == 0:
            raise ValueError("OXR document max_new_tokens must be positive")
        switches = (layout_provides_reading_order, table_image_placeholders,
                    full_overlap_dedupe, allow_truncated_content)
        if any(type(value) is not bool for value in switches):
            raise TypeError("OXR processing switches must be bool")
        for name, value in (("layout_score_threshold", layout_score_threshold),
                            ("layout_nms_iou_threshold", layout_nms_iou_threshold)):
            if value is None and name == "layout_nms_iou_threshold":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in 0..1")
        if layout_url is not None:
            from urllib.parse import urlsplit
            if not isinstance(layout_url, str) or "\0" in layout_url:
                raise ValueError("layout_url must be an explicit HTTP(S) URL")
            parsed = urlsplit(layout_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("layout_url must be an explicit HTTP(S) URL")
        if layout_model is not None and layout_url is not None:
            raise ValueError("Pass layout_model (native D-FINE) or layout_url (service), not both")
        self._native_layout = None
        self._layout_model = None if layout_model is None else os.fspath(layout_model)
        self._layout_options = dict(
            url=layout_url, score_threshold=layout_score_threshold,
            nms_iou_threshold=layout_nms_iou_threshold,
            provides_reading_order=layout_provides_reading_order,
        )
        self._pipeline_options = dict(
            image_output_dir=os.fspath(image_output_dir),
            table_image_placeholders=table_image_placeholders,
            full_overlap_dedupe=full_overlap_dedupe,
        )
        self.allow_truncated_content = allow_truncated_content
        self._lib = _load_library(lib_path)
        signatures = (
            ("init", [C.c_char_p, C.c_int], C.c_void_p),
            ("free", [C.c_void_p], None),
            ("predict", [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int,
                         C.c_int, C.c_char_p, C.c_int, C.c_float], C.c_char_p),
            ("last_error", [C.c_void_p], C.c_char_p),
        )
        for suffix, args, result in signatures:
            try:
                fn = getattr(self._lib, "crispembed_oxr_" + suffix)
            except AttributeError as error:
                raise RuntimeError("Native library lacks OXR support; build this source revision") from error
            fn.argtypes, fn.restype = args, result
        self._ctx = self._lib.crispembed_oxr_init(path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(self._error())
        if self._layout_model is not None:
            from ._oxr_layout import CrispOXRLayout
            self._native_layout = CrispOXRLayout(self._layout_model, n_threads, lib_path)

    @staticmethod
    def _budget(value: int) -> int:
        if type(value) is not int or not 0 <= value <= 32768:
            raise ValueError("max_new_tokens must be an integer in 0..32768")
        return value

    @staticmethod
    def _rgb(image):
        from PIL import Image
        if isinstance(image, (str, os.PathLike)):
            with Image.open(os.fspath(image)) as opened:
                return np.ascontiguousarray(opened.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image):
            return np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
        if not isinstance(image, np.ndarray):
            raise TypeError("OXR images must be a local path, PIL.Image or uint8 RGB NumPy array")
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("OXR arrays must be uint8 HxW, HxWx3 RGB or HxWx4 RGBA")
        if min(image.shape[:2]) <= 0:
            raise ValueError("OXR image must be nonempty")
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("OXR array channel count must be 3 or 4")
        return np.ascontiguousarray(Image.fromarray(image).convert("RGB"), dtype=np.uint8)

    def _require_open(self):
        if not self._ctx:
            raise RuntimeError("OXR context is closed")

    def _error(self):
        raw = self._lib.crispembed_oxr_last_error(self._ctx)
        return raw.decode("utf-8", errors="replace") if raw else "OXR operation failed"

    def predict(self, image, prompt: str | None = None, max_new_tokens: int | None = None,
                *, task: str = "text", repetition_penalty: float = 1.0) -> dict[str, Any]:
        """Run one image request using the official service preprocessing profile.

        ``task='layout'`` is the upstream diagnostic model prompt. It does not
        substitute for D-FINE layout detection in the full document pipeline.
        An explicit prompt is passed unchanged and takes precedence over task.
        """
        if prompt is None:
            if task not in PROMPTS:
                raise ValueError(f"Unknown OXR task: {task}")
            prompt = PROMPTS[task]
        if not isinstance(prompt, str) or not prompt or "\0" in prompt:
            raise ValueError("OXR prompt must be a nonempty NUL-free string")
        budget = self._budget(self.max_new_tokens if max_new_tokens is None else max_new_tokens)
        if (isinstance(repetition_penalty, bool) or
                not isinstance(repetition_penalty, (int, float)) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("repetition_penalty must be finite and positive")
        pixels = self._rgb(image)
        height, width, channels = pixels.shape
        if width > 2**31 - 1 or height > 2**31 - 1:
            raise ValueError("OXR image dimensions exceed the native ABI")
        with self._lock:
            self._require_open()
            raw = self._lib.crispembed_oxr_predict(
                self._ctx, pixels.ctypes.data_as(C.POINTER(C.c_uint8)), width,
                height, channels, prompt.encode("utf-8"), budget, repetition_penalty,
            )
            if raw is None:
                raise RuntimeError(self._error())
            result = json.loads(raw.decode("utf-8"))
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise RuntimeError("OXR native result must be an object containing text")
            self.last_trace.append(dict(
                prompt=prompt, width=width, height=height,
                rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                max_new_tokens=budget, repetition_penalty=float(repetition_penalty),
                result=deepcopy(result),
            ))
            return result

    def _official_helper(self):
        if self._orchestrator is not None:
            return self._orchestrator
        blueprint = _Blueprint(self._blueprint_path)
        try:
            config = blueprint.load("oxr.config.settings")
            settings = config.Settings.model_construct(
                debug=False, server=config.ServerConfig(),
                layout_analyze_model=config.LayoutAnalyzeModelConfig(**self._layout_options),
                oxr_model=config.OXRModelConfig(max_tokens=self.max_new_tokens, max_concurrency=1),
                pipeline=config.PipelineConfig(**self._pipeline_options),
            )
            config.settings = settings
            client_module = blueprint.load("oxr.model.client")
            owner = self

            class NativeClient(client_module.OXRModelClient):
                async def _call_vision_model(self, prompt, image, timeout, model_options=None, max_tokens=None):
                    del timeout  # A synchronous native context has no HTTP timeout.
                    options = client_module.resolve_model_options(model_options)
                    if options["temperature"] != 0.0 or options["top_p"] != 1.0:
                        raise ValueError("Native OXR supports the deterministic temperature=0, top_p=1 profile")
                    rgb = np.ascontiguousarray(image[:, :, ::-1])
                    result = owner.predict(
                        rgb, prompt,
                        settings.oxr_model.max_tokens if max_tokens is None else max_tokens,
                        repetition_penalty=options["repetition_penalty"],
                    )
                    owner._last_outcomes[asyncio.current_task()] = result
                    reason = result.get("finish_reason")
                    if reason not in {"stop", "length"}:
                        reason = "stop" if result.get("eos") else "length" if result.get("capped") else "unknown"
                    return client_module.ModelOutput(result["text"], finish_reason=reason)

            client_module.model_client = NativeClient(settings.oxr_model)
            orchestrator = blueprint.load("oxr.pipeline.orchestrator")
            # The original helper's PDFium exclusion must remain process-wide
            # even though its Python modules are private to each native context.
            blueprint.load("oxr.utils.pdf")._PDFIUM_LOCK = _PDFIUM_LOCK
            self._blueprint = blueprint
            self._orchestrator = orchestrator
            return orchestrator
        except BaseException:
            blueprint.close()
            raise

    @staticmethod
    def layout_response_from_regions(pages: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        """Build the official layout response from pixel-coordinate regions.

        Each region has ``type`` (one of LAYOUT_CLASSES), ``bbox`` (four pixel
        coordinates), and optional ``score``. Original thresholding, clipping,
        classwise NMS and geometry validation run when processing the pages.
        """
        labels, boxes, scores = [], [], []
        for page in pages:
            page_labels, page_boxes, page_scores = [], [], []
            for region in page:
                if not isinstance(region, Mapping) or set(region) - {"type", "bbox", "score"}:
                    raise ValueError("OXR regions support only type, bbox and score")
                kind = region.get("type")
                if kind not in LAYOUT_CLASSES:
                    raise ValueError(f"Unsupported OXR layout class: {kind}")
                page_labels.append(LAYOUT_CLASSES.index(kind))
                page_boxes.append(deepcopy(region.get("bbox")))
                page_scores.append(region.get("score", 1.0))
            labels.append(page_labels)
            boxes.append(page_boxes)
            scores.append(page_scores)
        return {"status": "success", "params": {}, "data": {
            "labels": labels, "boxes": boxes, "scores": scores,
        }}

    @staticmethod
    def _check_sync_context():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("Use aprocess_pages/aprocess_document from an active asyncio event loop")

    def _process(self, images, file_name, output_formats, keep_header_footer,
                 model_options, element_formats, layout_response, asset_dir,
                 asset_url_prefix, *, document_bytes=None):
        self._check_sync_context()
        if not isinstance(file_name, str) or not file_name or "\0" in file_name:
            raise ValueError("file_name must be a nonempty NUL-free string")
        if type(keep_header_footer) is not bool:
            raise TypeError("keep_header_footer must be bool")
        formats = list(output_formats)
        if not formats or any(value not in {"json", "markdown"} for value in formats):
            raise ValueError("output_formats supports json and markdown")
        if model_options is not None and (
                not isinstance(model_options, Mapping) or
                set(model_options) - {"temperature", "top_p", "repetition_penalty"}):
            raise ValueError("Unsupported OXR generation option")
        with self._lock:
            self._require_open()
            helper = self._official_helper()
            if layout_response is None and not self._layout_options["url"] and self._native_layout is None:
                raise ValueError("Document processing requires supplied layouts, layout_model or layout_url")
            self.last_trace = []
            self._last_outcomes = {}
            original_layout = helper.analyze_layouts
            layout = self._blueprint.load("oxr.pipeline.layout")
            config = self._blueprint.load("oxr.config.settings").settings
            parser = self._blueprint.load("oxr.pipeline.postprocess.layout_postprocess")
            supplied = deepcopy(layout_response)

            async def analyze_layouts(page_images, options=None):
                if supplied is not None:
                    return parser.parse_layout_batch(
                        supplied, [(int(p.shape[1]), int(p.shape[0])) for p in page_images],
                        config.layout_analyze_model.score_threshold,
                        config.layout_analyze_model.nms_iou_threshold,
                    )
                if self._native_layout is not None:
                    # Same boundary as the HTTP client: helper pages are BGR,
                    # the service decodes them to RGB before preprocessing.
                    response = await asyncio.to_thread(
                        self._native_layout.analyze, [np.ascontiguousarray(p[:, :, ::-1]) for p in page_images])
                    return parser.parse_layout_batch(
                        response, [(int(p.shape[1]), int(p.shape[0])) for p in page_images],
                        config.layout_analyze_model.score_threshold,
                        config.layout_analyze_model.nms_iou_threshold,
                    )
                # An explicit config makes the original helper own and close
                # its HTTP client inside this run's event loop.
                return await layout.analyze_layouts(
                    page_images, options, config=config.layout_analyze_model,
                )

            async def run():
                started_at = time.perf_counter()
                page_images = images
                if document_bytes is not None:
                    page_images = await asyncio.to_thread(
                        helper._decode_document_images, document_bytes, file_name,
                    )
                return await helper.process_decoded_document(
                    page_images, file_name, formats, keep_header_footer,
                    dict(model_options) if model_options is not None else None,
                    dict(element_formats) if element_formats is not None else None,
                    asset_dir=Path(asset_dir) if asset_dir is not None else None,
                    asset_url_prefix=asset_url_prefix, started_at=started_at,
                )

            helper.analyze_layouts = analyze_layouts
            try:
                result = asyncio.run(run())
            except BaseException:
                self._last_outcomes = {}
                raise
            finally:
                helper.analyze_layouts = original_layout
            unfinished = [r for r in self._last_outcomes.values()
                          if r.get("capped") or not r.get("eos")]
            self._last_outcomes = {}
            if unfinished and not self.allow_truncated_content:
                raise RuntimeError(
                    f"OXR has {len(unfinished)} request(s) without EOS after upstream recovery; "
                    "inspect last_trace, or explicitly allow_truncated_content"
                )
            result["metadata"]["crispembed"] = {
                "engine": "oxr-1.0",
                "layout_source": ("provided" if supplied is not None
                                  else "native_dfine" if self._native_layout is not None else "http_service"),
                "native_requests": len(self.last_trace),
                "unresolved_capped_requests": len(unfinished),
                "blueprint_revision": BLUEPRINT_REVISION,
            }
            return result

    def process_pages(self, images: Sequence[Any], *, layouts=None, layout_response=None,
                      file_name: str = "document.png", output_formats=("json", "markdown"),
                      keep_header_footer: bool = False, model_options=None, element_formats=None,
                      asset_dir=None, asset_url_prefix: str = "") -> dict[str, Any]:
        """Run the full original document helper on predecoded RGB pages.

        Supply either ``layouts`` (a region list per page) or the original
        ``layout_response`` batch object. With neither, ``layout_url`` must have
        been configured explicitly. No implicit document upload is performed.
        """
        if layouts is not None and layout_response is not None:
            raise ValueError("Pass layouts or layout_response, not both")
        if layouts is not None:
            layout_response = self.layout_response_from_regions(layouts)
        pages = [np.ascontiguousarray(self._rgb(image)[:, :, ::-1]) for image in images]
        if not pages:
            raise ValueError("OXR requires at least one page")
        return self._process(pages, file_name, output_formats, keep_header_footer,
                             model_options, element_formats, layout_response, asset_dir,
                             asset_url_prefix)

    def process_document(self, document, *, file_name: str | None = None, layouts=None,
                         layout_response=None, output_formats=("json", "markdown"),
                         keep_header_footer: bool = False, model_options=None,
                         element_formats=None, asset_dir=None, asset_url_prefix: str = ""):
        """Process a local PDF/image or bytes using original OXR page decoding."""
        if isinstance(document, (str, os.PathLike)):
            path = Path(document)
            data = path.read_bytes()
            file_name = file_name or path.name
        elif isinstance(document, (bytes, bytearray)):
            data = bytes(document)
            if file_name is None:
                raise ValueError("file_name is required for document bytes")
        else:
            raise TypeError("document must be a local path or bytes")
        if layouts is not None and layout_response is not None:
            raise ValueError("Pass layouts or layout_response, not both")
        if layouts is not None:
            layout_response = self.layout_response_from_regions(layouts)
        return self._process(None, file_name, output_formats, keep_header_footer,
                             model_options, element_formats, layout_response, asset_dir,
                             asset_url_prefix, document_bytes=data)

    async def aprocess_pages(self, images: Sequence[Any], **kwargs):
        """Async convenience wrapper; one native context still serializes work."""
        return await asyncio.to_thread(self.process_pages, images, **kwargs)

    async def aprocess_document(self, document, **kwargs):
        return await asyncio.to_thread(self.process_document, document, **kwargs)

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.crispembed_oxr_free(self._ctx)
                self._ctx = None
            if getattr(self, "_native_layout", None) is not None:
                self._native_layout.close()
                self._native_layout = None
            if self._blueprint is not None:
                self._blueprint.close()
                self._blueprint = None
                self._orchestrator = None

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
