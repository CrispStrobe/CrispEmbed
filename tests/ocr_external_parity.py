#!/usr/bin/env python3
"""Head-to-head OCR parity: CrispEmbed vs Tesseract / EasyOCR / PaddleOCR /
a full document parser.

Existing harnesses in this repo compare CrispEmbed against per-stage tensor
references.  That proves a graph is faithful; it says nothing about whether the
shipped pipeline reads a page as well or as fast as the engine it ports.  This
runs the external engines and the native ones over the same images and reports
the two numbers a user actually feels: transcription error and latency.

Two quality scores are produced because they answer different questions:

  * ``cer``/``wer`` vs ground truth  — absolute quality.  Only available for the
    synthetic corpus (``tests/ocr_synth_corpus.py``), which knows its own text.
  * ``cer`` vs a reference *engine*  — port fidelity.  A CrispEmbed engine that
    disagrees with the upstream engine it ports is a runtime bug to bisect with
    ``crispembed-diff``, independent of how hard the page is.

Timing is deliberately reported in two columns, because the arms are not
symmetric (dev-guide rule 4a): ``proc_ms`` is wall time for a whole invocation
including model load — what a CLI user pays — while ``engine_ms`` excludes model
load (parsed from native stage-bench stderr, or measured directly for the
in-process Python engines).  Never mix the two columns in a speed claim.

One arm is not a flat OCR engine at all but a whole document parser (layout +
OCR + table structure).  It is scored the same way on purpose: what a caller
gets back from a document parser is the exported document, so its layout stage
is part of its transcription quality, not a separate concern.  See ``DoclingPy``
for the two views it records and why.

Usage:
  python tests/ocr_synth_corpus.py --output /tmp/ocr-synth
  python tests/ocr_external_parity.py --images /tmp/ocr-synth \
      --model-dir /Volumes/backups/ai/crispembed-gguf --repeats 3 \
      --output /tmp/ocr-parity.json --markdown /tmp/ocr-parity.md

  # document-parser arm only (its own venv), labelled fixtures only:
  HF_HOME=$HOME/.cache/hf-docling ~/venvs/docling/bin/python \
      tests/ocr_external_parity.py --images tests/regression/images/cc0 \
      --require-truth --repeats 3 --output /tmp/cc0.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESSDATA = "/opt/homebrew/share/tessdata"

# Native stage-bench lines (stderr, opt-in via env) carry the load-excluded cost.
STAGE_BENCH = {
    "ppocrv6": re.compile(r"\[ppocrv6-stage-bench\].*?total=([0-9.]+) ms"),
    # easyocr's total= includes model load; detect+recognize= is the
    # load-excluded figure this column claims to be.
    "easyocr": re.compile(r"\[easyocr-stage-bench\].*?detect\+recognize=([0-9.]+) ms"),
    "tesseract": re.compile(r"\[tesseract-stage-bench\].*?total=([0-9.]+) ms"),
}
REGIONS_RE = re.compile(r"^regions=(\d+)\s+mean_conf=([0-9.]+)", re.M)


# ---------------------------------------------------------------- text metrics

def _edit(a: list | str, b: list | str) -> int:
    try:
        import Levenshtein

        if isinstance(a, str):
            return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(s: str) -> str:
    """Collapse layout differences so CER measures recognition, not line breaks.

    Engines disagree on where a line ends and whether regions are joined by a
    space or a newline; scoring that as character error would drown the signal
    we care about.  Case and punctuation are preserved — they are real errors.
    """
    s = s.replace("­", "")  # soft hyphen: never in ground truth
    return re.sub(r"\s+", " ", s).strip()


def score(hyp: str, ref: str) -> dict:
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return {"cer": None, "wer": None, "ref_chars": 0}
    cer = _edit(h, r) / len(r)
    hw, rw = h.split(), r.split()
    wer = _edit(hw, rw) / max(1, len(rw))
    # A document parser emits blocks in the order its reading-order stage chose,
    # which can differ from the page order the ground truth records while every
    # glyph is right; CER charges that at roughly half the page.  Comparing the
    # two word *multisets* is blind to both ordering and to how the arm chunks
    # the page into lines or paragraphs, so a gap between `wer` and
    # `wer_unordered` localises the defect to reading order rather than
    # recognition.  It is a diagnostic, not a better headline: quote both.
    unordered = _edit(sorted(hw), sorted(rw)) / max(1, len(rw))
    return {"cer": round(cer, 5), "wer": round(wer, 5), "ref_chars": len(r),
            "wer_unordered": round(unordered, 5), "exact": h == r}


# ------------------------------------------------------------- engine adapters

class Engine:
    """One OCR arm.  ``run`` returns (text, proc_ms, engine_ms, extra)."""

    name = "?"
    kind = "?"  # external | crispembed

    def available(self) -> str:
        return ""

    def run(self, image: Path, repeats: int) -> tuple[str, float, float | None, dict]:
        raise NotImplementedError


class TesseractCLI(Engine):
    kind = "external"

    def __init__(self, lang: str = "eng", psm: int = 6):
        self.lang, self.psm = lang, psm
        self.name = f"tesseract-cli:{lang}"

    def available(self) -> str:
        if not shutil.which("tesseract"):
            return "tesseract not on PATH"
        return ""

    def run(self, image: Path, repeats: int):
        env = os.environ.copy()
        env.pop("TESSDATA_PREFIX", None)  # CLAUDE.md: never inherit an ambiguous one
        cmd = ["tesseract", str(image), "stdout", "-l", self.lang,
               "--psm", str(self.psm)]
        if Path(TESSDATA).is_dir():
            cmd += ["--tessdata-dir", TESSDATA]
        times, out = [], ""
        for _ in range(repeats):
            t = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            times.append((time.perf_counter() - t) * 1000)
            out = p.stdout
        # Tesseract loads its model per invocation; it has no separable
        # "engine only" cost from the CLI, so engine_ms stays None rather than
        # being faked from the wall clock.
        return out, statistics.median(times), None, {}


class EasyOCRPy(Engine):
    kind = "external"
    name = "easyocr-py"

    def __init__(self, langs=("en",), gpu=False):
        self.langs, self.gpu, self._reader = list(langs), gpu, None

    def available(self) -> str:
        try:
            import easyocr  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import easyocr failed: {exc}"
        return ""

    def _ensure(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return self._reader

    def run(self, image: Path, repeats: int):
        reader = self._ensure()
        times, lines = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            lines = reader.readtext(str(image), detail=0, paragraph=False)
            times.append((time.perf_counter() - t) * 1000)
        med = statistics.median(times)
        # In-process: the model is already resident, so wall == engine cost.
        return "\n".join(lines), med, med, {"regions": len(lines)}


class PaddleOCRPy(Engine):
    kind = "external"
    name = "paddleocr-py"

    def __init__(self, lang: str = "en"):
        self.lang, self._ocr = lang, None

    def available(self) -> str:
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import paddleocr failed: {exc}"
        return ""

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True, show_log=False)
        return self._ocr

    @staticmethod
    def _texts(result) -> list[str]:
        # PaddleOCR 2.x returns [[ [box, (text, conf)], ... ]]; 3.x returns a
        # dict-like record.  Accept both rather than pinning a version.
        out = []
        if not result:
            return out
        page = result[0]
        if isinstance(page, dict):
            return list(page.get("rec_texts", []))
        for item in page or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                payload = item[1]
                out.append(payload[0] if isinstance(payload, (list, tuple)) else str(payload))
        return out

    def run(self, image: Path, repeats: int):
        ocr = self._ensure()
        times, texts = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            try:
                res = ocr.ocr(str(image), cls=True)
            except TypeError:
                res = ocr.ocr(str(image))
            times.append((time.perf_counter() - t) * 1000)
            texts = self._texts(res)
        med = statistics.median(times)
        return "\n".join(texts), med, med, {"regions": len(texts)}


class DoclingPy(Engine):
    """Full document-parser arm: layout analysis + OCR + structure, in-process.

    This is a different *shape* of reference from the flat OCR arms above.  A
    document parser does not emit "every recognised line"; it emits a document
    tree, and only the text its layout stage files under a text-bearing element
    survives into the export.  Anything the layout model calls a picture is
    exported as an image placeholder and its recognised words are dropped.  So
    two numbers are recorded per fixture:

      * ``text``          — the plain text of the exported document.  This is
                            what a caller of the library actually receives, and
                            it is what CER/WER score.
      * ``alt_texts["all_text_items"]`` — every recognised text item in the
                            document, including ones nested under a picture.
                            Scoring both separates "the OCR missed it" from
                            "the layout stage discarded it"; collapsing them
                            into one number would blame the recogniser for a
                            layout decision.

    Markdown is kept alongside (``extra["markdown"]``) so a later structure gate
    can score tables/headings without re-running the parse.

    The parser is built once and warmed with a throwaway parse, so the reported
    ``proc_ms`` and ``engine_ms`` are the same load-excluded number.  The warm-up
    matters more than usual here: the layout detector runs under a tracing JIT,
    so the very first parse pays a multi-second compile that is not part of
    steady-state cost.
    """

    kind = "external"

    def __init__(self, name: str = "docling-py", force_full_page_ocr: bool = False,
                 ocr_engine: str = "auto"):
        self.name = name
        self.force_full_page_ocr = force_full_page_ocr
        self.ocr_engine = ocr_engine
        self._conv = None
        self._backend = None
        self._warmed = False
        self._log_evidence: list[str] = []

    def available(self) -> str:
        try:
            import docling  # noqa: F401
            import docling_core  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import docling failed: {exc}"
        return ""

    # The engine that actually reads the pixels is picked at pipeline-build time
    # from whatever happens to be importable in this environment, so the
    # configured value ("auto") is not the answer.  Read it off the constructed
    # pipeline after the warm-up parse, and record the concrete model files.
    def _detect_backend(self) -> dict:
        info: dict = {"configured": self.ocr_engine, "selected": "unknown"}
        pipelines = list(getattr(self._conv, "initialized_pipelines", {}).values())
        for pipe in pipelines:
            model = getattr(pipe, "ocr_model", None)
            if model is None:
                continue
            # The auto-selector delegates to a concrete model it holds privately.
            inner = getattr(model, "_engine", None) or model
            info["selected"] = type(inner).__name__
            opts = getattr(inner, "options", None)
            if opts is not None:
                info["selected_kind"] = getattr(opts, "kind", None)
                for attr in ("backend", "lang", "model_storage_directory"):
                    if hasattr(opts, attr):
                        info[attr] = getattr(opts, attr)
            reader = getattr(inner, "reader", None)
            if reader is not None:
                info["reader"] = type(reader).__name__
            break
        if self._log_evidence:
            info["log_evidence"] = self._log_evidence
        return info

    @staticmethod
    def _capture_selection_log(fn):
        """Run ``fn`` with INFO capture, keeping the lines that name an engine.

        The selection is only ever *announced*; there is no attribute holding
        "which weights did you just load".  Keeping the log lines verbatim means
        the recorded backend is evidence rather than an inference.
        """
        import logging

        wanted = re.compile(r"(?i)(ocr model selected|cannot be used|Using .*\.(pth|onnx))")
        seen: list[str] = []

        class _Grab(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                except Exception:  # pragma: no cover - defensive
                    return
                if wanted.search(msg) and msg not in seen:
                    seen.append(msg)

        root = logging.getLogger()
        handler, old_level = _Grab(level=logging.INFO), root.level
        root.addHandler(handler)
        if old_level > logging.INFO or old_level == logging.NOTSET:
            root.setLevel(logging.INFO)
        try:
            fn()
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        return seen
        return info

    def _ensure(self, image: Path):
        if self._conv is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )

            opts = PdfPipelineOptions()
            opts.do_ocr = True
            if self.ocr_engine != "auto":
                from docling.models.factories import get_ocr_factory

                factory = get_ocr_factory()
                opts.ocr_options = factory.create_options(kind=self.ocr_engine)
            opts.ocr_options.force_full_page_ocr = self.force_full_page_ocr
            self._conv = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE, InputFormat.PDF],
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                },
            )
        if not self._warmed:
            # Throwaway parse: builds the pipeline, downloads/loads every model
            # and pays the JIT compile, so nothing below times model load.
            self._log_evidence = self._capture_selection_log(
                lambda: self._conv.convert(str(image))
            )
            self._warmed = True
            self._backend = self._detect_backend()
        return self._conv

    @staticmethod
    def _plain_text(doc) -> str:
        """Document text with the markdown decoration removed.

        ``export_to_text`` still serialises tables with pipe separators, which
        would be scored as character errors against plain ground truth.  Walk
        the tree instead and take the raw strings, table cells included.
        """
        from docling_core.types.doc import TableItem, TextItem

        parts: list[str] = []
        for item, _level in doc.iterate_items():
            if isinstance(item, TableItem):
                seen = set()
                for cell in item.data.table_cells:
                    key = (cell.start_row_offset_idx, cell.start_col_offset_idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    if cell.text:
                        parts.append(cell.text)
            elif isinstance(item, TextItem):
                if item.text:
                    parts.append(item.text)
        return "\n".join(parts)

    def run(self, image: Path, repeats: int):
        conv = self._ensure(image)
        times, doc = [], None
        for _ in range(repeats):
            t = time.perf_counter()
            res = conv.convert(str(image))
            times.append((time.perf_counter() - t) * 1000)
            doc = res.document
        med = statistics.median(times)
        text = self._plain_text(doc)
        every = "\n".join(t.text for t in doc.texts if t.text)
        extra = {
            "ocr_backend": self._backend,
            "force_full_page_ocr": self.force_full_page_ocr,
            "markdown": doc.export_to_markdown(),
            "n_text_items": len(doc.texts),
            "n_tables": len(doc.tables),
            "n_pictures": len(doc.pictures),
            "regions": len(doc.texts),
            "alt_texts": {"all_text_items": every},
        }
        # In-process and pre-warmed: wall time is the engine cost.
        return text, med, med, extra


class Qwen25VLPy(Engine):
    """Vision-language transcription arm, run through the reference PyTorch stack.

    This exists to give the native VL lane (``src/qwen2vl_ocr.cpp``) a gold it can
    be held to.  Unlike the flat OCR arms, the *only* thing that makes a VL model
    an OCR engine is the prompt, so the contract this adapter reproduces is the
    native lane's, verbatim:

      * ``PROMPT`` below is byte-for-byte the string the native lane uses when it
        detects a plain Qwen2.5-VL checkpoint.  Changing it changes the arm's
        quality; changing it *silently* would make the gold measure a different
        question than the lane answers.
      * The chat wrapper is produced by the checkpoint's own chat template — not
        hand-rolled — with the default system turn and a single user turn whose
        content is [image, text] in that order.  The applied string is recorded
        per run so a template change upstream shows up as a diff, not as a
        mysterious quality shift.
      * Greedy decoding, no sampling.  Anything else makes the gold irreproducible.
      * Preprocessor defaults are left alone: ``min_pixels``/``max_pixels`` decide
        how many vision tokens a page gets, i.e. the effective resolution the model
        reads at, and overriding them would make the numbers unquotable against the
        published model.

    Weights are loaded once and warmed with a throwaway generation, so ``proc_ms``
    and ``engine_ms`` are the same load-excluded number.  A page here costs seconds
    to minutes, so ``--repeats 1`` is the expected setting; timing is only ever
    comparable within one host.

    ``transcripts`` writes one ``<fixture>.txt`` per page plus a ``manifest.json``
    pinning model id, resolved revision, prompt, decoding params and hardware —
    that directory is the artifact the native lane's CER gate reads.
    """

    kind = "external"

    # Must stay identical to the OCR prompt in src/qwen2vl_ocr.cpp.
    PROMPT = "Read all the text in this image. Output the exact text content only."
    SYSTEM = "You are a helpful assistant."

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 name: str | None = None, dtype: str = "bfloat16",
                 device: str = "auto", max_new_tokens: int = 2048,
                 transcripts: Path | None = None, device_map: str = "",
                 max_memory: str = "", attn: str = "sdpa"):
        self.model_id = model_id
        short = model_id.split("/")[-1].replace("-Instruct", "").lower()
        self.name = name or f"qwen-vl-py:{short}"
        self.dtype_name = dtype
        self.device_arg = device
        # 7B at 16-bit is ~15.5 GiB of weights, which does not fit one 16 GB
        # accelerator alongside a page's vision tokens.  ``device_map`` hands
        # placement to accelerate so the same script runs sharded across several
        # smaller GPUs; when it is set the model is already placed and must not
        # be moved with ``.to()``.
        self.device_map = device_map
        # accelerate's placement fills the first device before moving on, so the
        # weights can leave a device with almost nothing free — and the *inputs*
        # all live on that first device.  A dense page then asks for a multi-GiB
        # activation and OOMs even though the box has spare VRAM elsewhere
        # (measured: 3.98 GiB requested against 2.26 GiB free on device 0, while
        # device 1 was half empty).  Capping the per-device weight budget is what
        # reserves that headroom; it is a placement knob, not a quality one.
        self.max_memory = max_memory
        # Eager attention materialises the full score matrix.  The vision tower
        # sees ~6k patches for a 4.8 Mpix scan, so that is a multi-GiB tensor per
        # full-attention layer and it is what actually ran the device out of
        # memory.  ``sdpa`` computes the same attention without ever holding the
        # matrix; it is a memory/implementation choice, not a model change, and
        # the resolved value is recorded so the claim is checkable.
        self.attn = attn
        self.max_new_tokens = max_new_tokens
        self.transcripts = Path(transcripts) if transcripts else None
        self._model = None
        self._proc = None
        self._device = None
        self._revision = None
        self._warmed = False
        self._applied_template: str | None = None
        self._placements: list[str] = []
        self._attn_resolved: str | None = None
        self._weight_gib: dict = {}
        self._peak_gib: dict = {}

    def available(self) -> str:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import torch/transformers failed: {exc}"
        if self._vl_class() is None:
            return ("transformers has no Qwen2_5_VL generation class "
                    f"(version {transformers.__version__})")
        return ""

    @staticmethod
    def _vl_class():
        """The Qwen2.5-VL head moved names across transformers majors.

        Probing for it (rather than importing one path) is what lets this adapter
        skip cleanly on an old install instead of dying at fixture time.
        """
        import transformers

        for attr in ("Qwen2_5_VLForConditionalGeneration",
                     "Qwen2_5_VLForVisionText2Text",
                     "AutoModelForImageTextToText"):
            cls = getattr(transformers, attr, None)
            if cls is not None:
                return cls
        return None

    def _pick_device(self):
        import torch

        if self.device_arg != "auto":
            return torch.device(self.device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure(self, image: Path):
        if self._model is None:
            import torch
            from transformers import AutoProcessor

            dtype = getattr(torch, self.dtype_name)
            self._device = self._pick_device()
            cls = self._vl_class()
            # transformers renamed the weight-dtype kwarg (``torch_dtype`` ->
            # ``dtype``) in v5.  Both are swallowed by **kwargs, so a wrong guess
            # does not raise — it silently loads fp32 and doubles the footprint.
            # Branch on the major version and assert the result instead.
            import transformers

            major = int(transformers.__version__.split(".")[0])
            key = "dtype" if major >= 5 else "torch_dtype"
            kw = {key: dtype}
            if self.device_map:
                kw["device_map"] = self.device_map
            if self.attn:
                kw["attn_implementation"] = self.attn
            if self.max_memory:
                kw["max_memory"] = {
                    (int(k) if k.strip().isdigit() else k.strip()): v.strip()
                    for k, v in (part.split("=", 1)
                                 for part in self.max_memory.split(","))}
            self._model = cls.from_pretrained(self.model_id, **kw)
            got = next(self._model.parameters()).dtype
            if got != dtype:
                raise RuntimeError(
                    f"requested {dtype} but weights loaded as {got}; the "
                    f"'{key}' kwarg was ignored by transformers {transformers.__version__}")
            if self.device_map:
                # accelerate already placed every module; record where the inputs
                # have to go rather than assuming a single device.
                self._device = next(self._model.parameters()).device
                # When the weights do not fit, accelerate offloads layers to CPU
                # or disk and each generated token streams them back.  The
                # transcripts stay correct and the timings become meaningless, so
                # record the placements instead of leaving that invisible.
                self._placements = sorted({
                    str(v) for v in getattr(self._model, "hf_device_map", {}).values()})
                # Placement knobs are easy to pass and easy to have silently
                # ignored — a wrong guess looks exactly like a correct one until
                # a page OOMs.  Read back the resident bytes per device so the
                # budget is measured, not assumed.
                if torch.cuda.is_available():
                    self._weight_gib = {
                        i: round(torch.cuda.memory_allocated(i) / 2 ** 30, 2)
                        for i in range(torch.cuda.device_count())}
                    print(f"    [vl] weights resident per device: {self._weight_gib}",
                          flush=True)
            else:
                self._model.to(self._device)
            self._model.eval()
            self._proc = AutoProcessor.from_pretrained(self.model_id)
            cfg = getattr(self._model, "config", None)
            self._revision = getattr(cfg, "_commit_hash", None) or "unknown"
            self._attn_resolved = getattr(cfg, "_attn_implementation", None) or self.attn
        if not self._warmed:
            # Throwaway generation: pays lazy kernel compile and any first-call
            # allocator growth, neither of which is steady-state page cost.
            self._generate(image, max_new_tokens=8)
            self._warmed = True
        return self._model

    def _messages(self) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": self.PROMPT}]},
        ]

    def _generate(self, image: Path, max_new_tokens: int | None = None) -> tuple[str, int]:
        import torch
        from PIL import Image

        img = Image.open(image).convert("RGB")
        text = self._proc.apply_chat_template(self._messages(), tokenize=False,
                                              add_generation_prompt=True)
        self._applied_template = text
        inputs = self._proc(text=[text], images=[img], return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(**inputs, do_sample=False,
                                       max_new_tokens=max_new_tokens or self.max_new_tokens)
        n_in = inputs["input_ids"].shape[1]
        new = out[0][n_in:]
        decoded = self._proc.tokenizer.decode(new, skip_special_tokens=True)
        return decoded.strip(), int(new.shape[0])

    def run(self, image: Path, repeats: int):
        import torch

        self._ensure(image)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        times, text, n_new = [], "", 0
        for _ in range(repeats):
            t = time.perf_counter()
            text, n_new = self._generate(image)
            times.append((time.perf_counter() - t) * 1000)
        med = statistics.median(times)
        if torch.cuda.is_available():
            self._peak_gib = {i: round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 2)
                              for i in range(torch.cuda.device_count())}
        if self.transcripts:
            self.transcripts.mkdir(parents=True, exist_ok=True)
            (self.transcripts / f"{image.name}.txt").write_text(text + "\n")
        extra = {
            "model_id": self.model_id,
            "revision": self._revision,
            "dtype": self.dtype_name,
            "device": str(self._device),
            "device_map": self.device_map or None,
            "device_map_placements": self._placements or None,
            "attn_implementation": self._attn_resolved,
            "prompt": self.PROMPT,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "new_tokens": n_new,
            "weights_gib_per_device": self._weight_gib or None,
            "peak_gib_per_device": self._peak_gib or None,
            "regions": len([ln for ln in text.splitlines() if ln.strip()]),
        }
        # In-process and pre-warmed: wall time is the engine cost.
        return text, med, med, extra

    def manifest(self, hardware: str) -> dict:
        import torch
        import transformers

        return {
            "model_id": self.model_id,
            "revision": self._revision,
            "prompt": self.PROMPT,
            "system": self.SYSTEM,
            "chat_template_applied_example": self._applied_template,
            "decoding": {"do_sample": False, "num_beams": 1,
                         "max_new_tokens": self.max_new_tokens},
            "preprocessor": "model defaults (min_pixels/max_pixels not overridden)",
            "dtype": self.dtype_name,
            "device": str(self._device),
            "device_map": self.device_map or None,
            "device_map_placements": self._placements or None,
            "max_memory": self.max_memory or None,
            "attn_implementation": self._attn_resolved,
            "weights_gib_per_device": self._weight_gib or None,
            "hardware": hardware,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }


class CrispEmbedCLI(Engine):
    kind = "crispembed"

    def __init__(self, name: str, engine: str, binary: Path, det: Path | None,
                 rec: Path | None, bench_env: dict[str, str] | None = None,
                 stage_key: str | None = None, extra_args: list[str] | None = None):
        self.name = name
        self.engine = engine
        self.binary = binary
        self.det, self.rec = det, rec
        self.bench_env = bench_env or {}
        self.stage_key = stage_key
        self.extra_args = extra_args or []

    def available(self) -> str:
        if not self.binary.exists():
            return f"missing binary {self.binary}"
        for m in (self.det, self.rec):
            if m is not None and not m.exists():
                return f"missing model {m}"
        return ""

    def run(self, image: Path, repeats: int):
        env = os.environ.copy()
        env.update(self.bench_env)
        cmd = [str(self.binary), "--ocr-pipeline", str(image), "--ocr-engine", self.engine]
        if self.det:
            cmd += ["--ocr-det", str(self.det)]
        if self.rec:
            cmd += ["--ocr-rec", str(self.rec)]
        cmd += self.extra_args
        times, stages, out, err, rc = [], [], "", "", 0
        for _ in range(repeats):
            t = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               cwd=ROOT, timeout=1800)
            times.append((time.perf_counter() - t) * 1000)
            out, err, rc = p.stdout, p.stderr, p.returncode
            if self.stage_key and self.stage_key in STAGE_BENCH:
                m = STAGE_BENCH[self.stage_key].search(err)
                if m:
                    stages.append(float(m.group(1)))
        extra = {"returncode": rc}
        m = REGIONS_RE.search(out)
        if m:
            extra["regions"] = int(m.group(1))
            extra["mean_conf"] = float(m.group(2))
        text = REGIONS_RE.sub("", out).strip()
        if rc != 0:
            # A non-zero exit never gets timed as a win (dev-guide rule 4a).
            extra["stderr_tail"] = err[-600:]
            return "", statistics.median(times), None, extra
        return text, statistics.median(times), (statistics.median(stages) if stages else None), extra


# ------------------------------------------------------------------- the sweep

def load_fixtures(images: Path) -> list[dict]:
    gt_path = images / "ground_truth.json"
    truth = {}
    if gt_path.exists():
        for rec in json.loads(gt_path.read_text())["records"]:
            truth[rec["file"]] = rec["text"]
    fixtures = []
    for p in sorted(images.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        fixtures.append({"name": p.name, "path": p, "truth": truth.get(p.name)})
    return fixtures


def build_engines(args) -> list[Engine]:
    binary = ROOT / args.build_dir / "crispembed"
    md = Path(args.model_dir) if args.model_dir else None
    engines: list[Engine] = [
        TesseractCLI(args.tesseract_lang, args.psm),
        EasyOCRPy(),
        PaddleOCRPy(),
        DoclingPy(force_full_page_ocr=args.docling_full_page,
                  ocr_engine=args.docling_ocr),
    ]
    # The VL arm is opt-in rather than probe-and-skip like the others: its
    # availability probe (torch + transformers) is satisfied by several unrelated
    # venvs in this repo, and being "available" there would silently start a
    # multi-GB weight download in a run that only wanted the OCR arms.
    if args.qwen:
        engines.append(Qwen25VLPy(model_id=args.qwen_model, dtype=args.qwen_dtype,
                                  device=args.qwen_device,
                                  max_new_tokens=args.qwen_max_new_tokens,
                                  transcripts=args.qwen_transcripts,
                                  device_map=args.qwen_device_map,
                                  max_memory=args.qwen_max_memory,
                                  attn=args.qwen_attn))

    def model(name: str) -> Path | None:
        return (md / name) if md else None

    if md:
        engines += [
            CrispEmbedCLI("crispembed-tesseract", "tesseract", binary,
                          model(args.dbnet), model(args.tess_rec),
                          {"CRISPEMBED_TESSERACT_BENCH": "1"}, "tesseract"),
            CrispEmbedCLI("crispembed-easyocr", "easyocr", binary,
                          model(args.dbnet), model(args.easyocr_rec),
                          {"CRISPEMBED_EASYOCR_BENCH": "1"}, "easyocr"),
            CrispEmbedCLI("crispembed-ppocrv6", "ppocrv6", binary,
                          model(args.ppocr_det), model(args.ppocr_rec),
                          {"CRISPEMBED_PPOCRV6_BENCH": "1"}, "ppocrv6"),
        ]
    return [e for e in engines if e.name not in args.skip]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path,
                    help="directory of fixtures; ground_truth.json is used when present")
    ap.add_argument("--model-dir", default=os.environ.get("CRISPEMBED_GGUF_DIR", ""))
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tesseract-lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--dbnet", default="dbnet-ic15-q8_0.gguf")
    ap.add_argument("--tess-rec", default="tesseract-eng-q8_0-seeded.gguf")
    ap.add_argument("--easyocr-rec", default="easyocr-english-g2-f16.gguf")
    ap.add_argument("--ppocr-det", default="PP-OCRv6_small_det-f16.gguf")
    ap.add_argument("--ppocr-rec", default="PP-OCRv6_small_rec-f16.gguf")
    ap.add_argument("--docling-full-page", action="store_true",
                    help="force the document parser to OCR whole pages")
    ap.add_argument("--docling-ocr", default="auto",
                    help="document-parser OCR engine ('auto' lets it choose)")
    ap.add_argument("--qwen", action="store_true",
                    help="enable the vision-language arm (downloads weights)")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="VL checkpoint; only the 7B is quotable as a reference "
                         "(the 3B ships under a research licence)")
    ap.add_argument("--qwen-dtype", default="bfloat16")
    ap.add_argument("--qwen-device", default="auto")
    ap.add_argument("--qwen-attn", default="sdpa",
                    help="attention implementation; eager materialises the full "
                         "score matrix and OOMs the vision tower on dense pages")
    ap.add_argument("--qwen-max-memory", default="",
                    help="per-device weight budget for accelerate, e.g. "
                         "'0=9GiB,1=13GiB'; reserves activation headroom on the "
                         "device the inputs land on")
    ap.add_argument("--qwen-device-map", default="",
                    help="hand placement to accelerate (e.g. 'auto') so the "
                         "weights can shard across several GPUs")
    ap.add_argument("--qwen-max-new-tokens", type=int, default=2048)
    ap.add_argument("--qwen-transcripts", type=Path,
                    help="directory to write one transcript per fixture plus a "
                         "manifest.json pinning model/prompt/decoding/hardware")
    ap.add_argument("--hardware", default="",
                    help="free-text hardware label recorded in the transcript "
                         "manifest; timings are only comparable within one host")
    ap.add_argument("--only", default="",
                    help="comma-separated fixture filenames to run; lets a "
                         "retry pass revisit just the fixtures that failed")
    ap.add_argument("--require-truth", action="store_true",
                    help="only run fixtures that ground_truth.json labels; "
                         "unlabelled fixtures in a corpus are out of scope, and "
                         "running them costs time without producing a number")
    ap.add_argument("--reference", default="tesseract-cli:eng",
                    help="engine whose output port-fidelity CER is measured against")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    fixtures = load_fixtures(args.images)
    if args.require_truth:
        fixtures = [f for f in fixtures if f["truth"]]
    if args.only:
        want = {n.strip() for n in args.only.split(",") if n.strip()}
        fixtures = [f for f in fixtures if f["name"] in want]
    if args.limit:
        fixtures = fixtures[: args.limit]
    if not fixtures:
        print(f"no fixtures in {args.images}", file=sys.stderr)
        return 2

    engines = build_engines(args)
    skipped = {}
    active = []
    for e in engines:
        why = e.available()
        if why:
            skipped[e.name] = why
            print(f"SKIP {e.name}: {why}", flush=True)
        else:
            active.append(e)

    rows = []
    for fx in fixtures:
        record = {"fixture": fx["name"], "has_truth": fx["truth"] is not None, "engines": {}}
        for e in active:
            # One page must not be able to destroy a run.  A VLM arm costs
            # GPU-minutes per fixture, and a single dense page that exhausts
            # device memory used to abort the whole corpus and throw away every
            # transcript already produced.  Record the failure as a row instead:
            # `error` is a result, and an arm that cannot read a page should show
            # up as a failure on that page rather than as a missing corpus.
            try:
                text, proc_ms, engine_ms, extra = e.run(fx["path"], args.repeats)
            except Exception as exc:
                print(f"  {fx['name']:28} {e.name:22} FAILED: {exc}", flush=True)
                record["engines"][e.name] = {
                    "kind": e.kind, "text": "", "proc_ms": None, "engine_ms": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    # The message alone says how big the failed allocation was
                    # but not which stage asked for it, which is the half that
                    # tells you what to change.  Keep the frames.
                    "traceback": traceback.format_exc().splitlines()[-25:],
                }
                continue
            entry = {
                "kind": e.kind,
                "text": text,
                "proc_ms": round(proc_ms, 1),
                "engine_ms": round(engine_ms, 1) if engine_ms is not None else None,
                **extra,
            }
            if fx["truth"]:
                entry.update(score(text, fx["truth"]))
                # An arm may expose a second view of the same run (e.g. the
                # recognised text a document parser discarded during layout).
                # Score it separately so the primary CER stays the number a
                # caller of that arm would actually get.
                alts = entry.get("alt_texts") or {}
                if alts:
                    entry["alt_scores"] = {
                        k: score(v, fx["truth"]) for k, v in alts.items()
                    }
            record["engines"][e.name] = entry
            tag = f"cer={entry.get('cer')}" if fx["truth"] else f"chars={len(normalize(text))}"
            print(f"  {fx['name']:28} {e.name:22} {proc_ms:8.1f} ms  {tag}", flush=True)

        # Port fidelity: every arm scored against the chosen reference engine's
        # own transcription of this same image.
        ref = record["engines"].get(args.reference, {}).get("text")
        if ref:
            for name, entry in record["engines"].items():
                if name == args.reference:
                    continue
                entry["vs_reference"] = score(entry["text"], ref)
        rows.append(record)

    result = {
        "version": 1,
        "images": str(args.images),
        "repeats": args.repeats,
        "reference_engine": args.reference,
        "skipped": skipped,
        "fixtures": rows,
    }
    summary = summarize(result)
    result["summary"] = summary
    for e in active:
        if isinstance(e, Qwen25VLPy) and e.transcripts:
            e.transcripts.mkdir(parents=True, exist_ok=True)
            man = e.manifest(args.hardware or platform.platform())
            man["images"] = str(args.images)
            man["fixtures"] = [fx["name"] for fx in fixtures]
            man["date"] = time.strftime("%Y-%m-%d")
            (e.transcripts / "manifest.json").write_text(
                json.dumps(man, indent=2) + "\n")
            result["vl_manifest"] = man
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    md = render_markdown(result)
    if args.markdown:
        args.markdown.write_text(md)
    print()
    print(md)
    return 0


def summarize(result: dict) -> dict:
    per_engine: dict[str, dict] = {}
    for fx in result["fixtures"]:
        for name, entry in fx["engines"].items():
            agg = per_engine.setdefault(name, {
                "kind": entry["kind"], "cer": [], "wer": [], "ref_cer": [],
                "wer_unordered": [],
                "proc_ms": [], "engine_ms": [], "failures": 0, "n": 0,
                "alt": {},
            })
            for key, sc in (entry.get("alt_scores") or {}).items():
                if sc.get("cer") is not None:
                    slot = agg["alt"].setdefault(key, {"cer": [], "wer": []})
                    slot["cer"].append(sc["cer"])
                    slot["wer"].append(sc["wer"])
            agg["n"] += 1
            if entry.get("returncode", 0) != 0 or not entry["text"].strip():
                agg["failures"] += 1
            if entry.get("cer") is not None:
                agg["cer"].append(entry["cer"])
                agg["wer"].append(entry["wer"])
                if entry.get("wer_unordered") is not None:
                    agg["wer_unordered"].append(entry["wer_unordered"])
            if entry.get("vs_reference", {}).get("cer") is not None:
                agg["ref_cer"].append(entry["vs_reference"]["cer"])
            # A failed fixture has no timing; keeping it out of the median means
            # the latency column describes the pages the arm actually read,
            # while `failures` still says how many it did not.
            if entry["proc_ms"] is not None:
                agg["proc_ms"].append(entry["proc_ms"])
            if entry["engine_ms"] is not None:
                agg["engine_ms"].append(entry["engine_ms"])
    out = {}
    for name, agg in per_engine.items():
        mean = lambda xs: round(statistics.fmean(xs), 5) if xs else None  # noqa: E731
        med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
        out[name] = {
            "kind": agg["kind"],
            "n": agg["n"],
            "failures": agg["failures"],
            "mean_cer": mean(agg["cer"]),
            "mean_wer": mean(agg["wer"]),
            "mean_wer_unordered": mean(agg["wer_unordered"]),
            "mean_cer_vs_reference": mean(agg["ref_cer"]),
            "median_proc_ms": med(agg["proc_ms"]),
            "median_engine_ms": med(agg["engine_ms"]),
        }
        if agg["alt"]:
            out[name]["alt"] = {
                k: {"mean_cer": mean(v["cer"]), "mean_wer": mean(v["wer"]),
                    "n": len(v["cer"])}
                for k, v in agg["alt"].items()
            }
    return out


def render_markdown(result: dict) -> str:
    lines = [
        f"### OCR head-to-head ({result['images']}, repeats={result['repeats']})",
        "",
        f"Port-fidelity reference engine: `{result['reference_engine']}`.",
        "`proc_ms` includes model load (one CLI invocation); `engine_ms` excludes it.",
        "The two columns are not comparable to each other.",
        "",
        "`WER (unord)` compares the two word multisets: a gap to `WER` is a",
        "reading-order difference, not a recognition one.",
        "",
        "| engine | kind | n | fail | CER↓ | WER↓ | WER (unord)↓ | CER vs ref | proc ms | engine ms |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    fmt = lambda v: "—" if v is None else f"{v}"  # noqa: E731
    for name, s in sorted(result["summary"].items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        lines.append(
            f"| `{name}` | {s['kind']} | {s['n']} | {s['failures']} | {fmt(s['mean_cer'])} | "
            f"{fmt(s['mean_wer'])} | {fmt(s.get('mean_wer_unordered'))} | "
            f"{fmt(s['mean_cer_vs_reference'])} | "
            f"{fmt(s['median_proc_ms'])} | {fmt(s['median_engine_ms'])} |"
        )
    alt_rows = [(n, k, v) for n, s in sorted(result["summary"].items())
                for k, v in (s.get("alt") or {}).items()]
    if alt_rows:
        lines += [
            "",
            "Secondary views of the same runs (not what the arm returns to a caller):",
            "",
            "| engine | view | n | CER↓ | WER↓ |",
            "|---|---|--:|--:|--:|",
        ]
        for n, k, v in alt_rows:
            lines.append(f"| `{n}` | {k} | {v['n']} | {fmt(v['mean_cer'])} | {fmt(v['mean_wer'])} |")
    if result["skipped"]:
        lines += ["", "Skipped: " + ", ".join(f"`{k}` ({v})" for k, v in result["skipped"].items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
