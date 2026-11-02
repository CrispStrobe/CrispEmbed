#!/usr/bin/env python3
"""Exact parity of the native OXR document route against the pinned helper.

Modes
  helpers  Randomized differential tests of every translated helper branch:
           the original SoMarkAI/OXR @ c1a1dcb3 functions versus
           ``test-oxr-document --helpers``. Model-free.
  replay   Full ``process_pages``: the original helper (through CrispOXR's
           adapter) with a scripted deterministic recognizer, then the native
           C++ route replaying the recorded requests. Compares JSON and
           Markdown exactly and the request sequence. Model-free.
  native   Same pages/layouts through CrispOXR.process_pages (original helper,
           native model) and CrispOXR.process_pages_native (C++ route, native
           model). Requires the built library and an OXR GGUF.

Run with the pinned helper environment (tools/oxr-blueprint-requirements.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crispembed import _oxr as crisp_oxr  # noqa: E402

crisp_oxr._source_snapshot(None)  # refuses unpinned helper source

import cv2  # noqa: E402
from oxr.model.prompts import build_reading_order_prompt  # noqa: E402
from oxr.pipeline import reading_order as ro  # noqa: E402
from oxr.pipeline.models import Block, BlockType, ContentFormat  # noqa: E402
from oxr.pipeline.postprocess import (  # noqa: E402
    chemical_structure_postprocess as chem_pp, code_postprocess as code_pp,
    cross_category_dedupe as cross_pp, exact_text_dedupe as exact_pp,
    formula_postprocess as formula_pp, full_overlap_dedupe as full_pp,
    json_postprocess as json_pp, layout_postprocess as layout_pp,
    overlap_dedupe as overlap_pp, table_postprocess as table_pp,
    text_postprocess as text_pp,
)
from oxr.pipeline.postprocess.assembly_postprocess import blocks_to_markdown  # noqa: E402
from oxr.pipeline.xycut_order import xycut_plus_order  # noqa: E402
from oxr.utils.image import resize_image  # noqa: E402

TYPES = [t.value for t in BlockType]
LA_TYPES = [t.value for t in layout_pp.LA_CLASS_ID_TO_BLOCK_TYPE]


class Driver:
    def __init__(self, binary):
        self.proc = subprocess.Popen([binary, "--helpers"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8")

    def __call__(self, request):
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def block_dict(b: Block):
    return {"idx": b.idx, "type": b.type.value, "bbox": list(b.bbox), "content": b.content,
            "format": b.format.value, "text_before": b.text_before, "text_after": b.text_after,
            "generated_role": b.generated_role, "layout_score": b.layout_score}


def make_block(d):
    return Block(idx=d.get("idx", 0), type=BlockType(d["type"]), bbox=list(d["bbox"]),
                 content=d.get("content", ""), format=ContentFormat(d.get("format", "text")),
                 text_before=d.get("text_before", ""), text_after=d.get("text_after", ""),
                 generated_role=d.get("generated_role", ""), layout_score=d.get("layout_score"))


FRAGMENTS = [
    "$$", "$", "\\[", "\\]", "\\(", "\\)", "\\tag{1}", "\\tag*{A}", "\\tag {2.1}", "\\leqno (3)", "\\eqno 4",
    "\\tagx", "```latex\n", "```\n", "\n```", "```python\n", "```html\n", "```HTML\n", "\\text{a b}", "\\textbf{x}",
    "\\verb", "\\\\", "\\\\(", "\\\\{", "%c\n", "{", "}", "<b>", "</td>", "<!-- c -->", "<!DOCTYPE x>", "<?x?>",
    "&lt;b&gt;", "&#60;i&#x3E;", "`", "``", "~~~", " ", "  ", "\n", "\n\n", "\t", "x", "y=", "+", "a b", "word text",
    "é", "中文", " ", "\\smiles{CCO}", "<table><tr><td>1</td></tr></table>", "\\ce{H2O}", "\\mathrm{Na}",
    "_{2}", "\\to", "->", "=", "1.", "- ", "# ", "    ", "|", "*", "\\alpha", "\\", "İ", "　",
]


def rand_text(rng, n=None):
    return "".join(rng.choice(FRAGMENTS) for _ in range(n if n is not None else rng.randint(0, 9)))


def rand_boxes(rng, n, w=1000, h=1400):
    out = []
    style = rng.random()
    for i in range(n):
        if style < 0.35:  # two-column page
            col = rng.randint(0, 1)
            x1 = 60 + col * 470 + rng.uniform(-5, 5)
            x2 = x1 + rng.uniform(300, 420)
            y1 = rng.uniform(50, 1300)
            y2 = y1 + rng.uniform(15, 120)
        elif style < 0.5:  # grid-aligned with ties
            x1 = float(rng.choice([50, 50, 300, 550]))
            y1 = float(rng.choice([100, 100, 400, 700, 1000]))
            x2 = x1 + rng.choice([200, 240, 900])
            y2 = y1 + rng.choice([50, 250])
        else:
            x1, y1 = rng.uniform(0, w - 20), rng.uniform(0, h - 20)
            x2, y2 = x1 + rng.uniform(1.2, w - x1), y1 + rng.uniform(1.2, min(400, h - y1))
        out.append([min(x1, w - 2), min(y1, h - 2), min(max(x2, x1 + 1.1), w), min(max(y2, y1 + 1.1), h)])
    return out


def rand_blocks(rng, n, types=None):
    return [{"idx": i, "type": rng.choice(types or TYPES), "bbox": b} for i, b in enumerate(rand_boxes(rng, n))]


def py_safe(fn, *args):
    try:
        return {"value": fn(*args)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


class Checker:
    def __init__(self):
        self.counts = {}
        self.failures = []

    def compare(self, family, request, expected, actual, ok):
        stats = self.counts.setdefault(family, [0, 0])
        stats[0] += 1
        if not ok:
            stats[1] += 1
            if len(self.failures) < 40:
                self.failures.append({"family": family, "request": request, "python": expected, "native": actual})


def run_helpers(binary, cases, seed):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    drv = Driver(binary)
    chk = Checker()
    try:
        # float repr
        values = [rng.uniform(-1e4, 1e4) for _ in range(200)] + [1e16, 1e-5, 0.1, 3.0, 123456789012345678.0, 2.5e-7]
        got = drv({"fn": "float_repr", "values": values})
        chk.compare("float_repr", None, [repr(float(v)) for v in values], got.get("repr"),
                    got.get("repr") == [repr(float(v)) for v in values])

        for _ in range(cases):
            # text helpers
            s = rand_text(rng, rng.randint(0, 14))
            for fn, py in (("space_markdown_math", formula_pp.space_markdown_math),
                           ("clean_formula", formula_pp.clean_formula_markdown_delimiters),
                           ("move_numbering", formula_pp.move_formula_numbering_outside_math)):
                got = drv({"fn": fn, "text": s})
                exp = py(s)
                chk.compare(fn, s, exp, got, got.get("text") == exp)
            # postprocess by type
            t = rng.choice(["Text", "Title", "Formula", "Table", "Code Block", "Chemical Structure"])
            raw = rand_text(rng, rng.randint(0, 12))
            b = Block(idx=0, type=BlockType(t), bbox=[0, 0, 1, 1], content="", format=ContentFormat.TEXT)
            if t == "Table":
                table_pp.postprocess_table(b, raw)
            elif t == "Formula":
                formula_pp.postprocess_formula(b, raw)
            elif t == "Code Block":
                code_pp.postprocess_code(b, raw)
            elif t == "Chemical Structure":
                chem_pp.postprocess_chemical_structure(b, raw)
            else:
                text_pp.postprocess_text(b, raw)
            exp = {"content": b.content, "format": b.format.value, "text_before": b.text_before,
                   "text_after": b.text_after}
            got = drv({"fn": "postprocess", "type": t, "raw": raw})
            chk.compare("postprocess:" + t, raw, exp, got, got == exp)

            # reading order prompt / parse / repair / supplement / XY-Cut++
            n = rng.randint(1, 26)
            blocks = rand_blocks(rng, n)
            py_blocks = [make_block(d) for d in blocks]
            width, height = rng.choice([(1000, 1400), (1240, 1754), (777, 333)])
            objs = {"ref": [x.type.value for x in py_blocks], "bbox": [x.bbox for x in py_blocks], "bbox_type": "real"}
            exp = build_reading_order_prompt(objs, image_size=(width, height))
            got = drv({"fn": "ro_prompt", "blocks": blocks, "width": width, "height": height})
            chk.compare("ro_prompt", blocks, exp, got, got.get("prompt") == exp)

            exp = py_safe(xycut_plus_order, py_blocks)
            got = drv({"fn": "xycut", "blocks": blocks})
            ok = ("error" in exp) == ("error" in got) and exp.get("value") == got.get("order")
            chk.compare("xycut", blocks, exp, got, ok)

            perm = list(range(n))
            rng.shuffle(perm)
            mode = rng.random()
            if mode < 0.25:
                text = " ".join(map(str, perm))
            elif mode < 0.5:
                drop = rng.sample(perm, min(len(perm), rng.randint(1, 3)))
                text = " ".join(str(v) for v in perm if v not in drop)
            elif mode < 0.7:
                text = " ".join(map(str, perm + perm[: rng.randint(1, 2)]))
            elif mode < 0.8:
                text = " ".join(map(str, perm[: max(0, n // 2 - 1)]))
            elif mode < 0.9:
                text = rng.choice(["", "  \n", "1, 2", f"{n}", "0 1", "a", "1 2 x"])
            else:
                text = "\n".join(map(str, perm)) + " "
            for fn, py in (("ro_repair", ro.repair_incomplete_reading_order),
                           ("ro_supplement", ro.supplement_incomplete_reading_order)):
                exp = py_safe(py, text, py_blocks)
                got = drv({"fn": fn, "text": text, "blocks": blocks})
                ok = ("error" in exp) == ("error" in got) and exp.get("value") == got.get("order")
                chk.compare(fn, {"text": text, "blocks": blocks}, exp, got, ok)
            exp = py_safe(ro.parse_reading_order, text, n)
            got = drv({"fn": "ro_parse", "text": text, "count": n})
            chk.compare("ro_parse", text, exp, got, exp.get("value") == got.get("order"))

            # image operations
            w, h = rng.randint(1, 90), rng.randint(1, 90)
            img = nprng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            exp_img = resize_image(img, 0.75)
            dh, dw = exp_img.shape[:2]
            got = drv({"fn": "resize_area", "pixels": img.tobytes().hex(), "w": w, "h": h, "cn": 3, "dw": dw, "dh": dh})
            if "error" in got:
                chk.compare("resize_area", [w, h], "ok", got, (dw, dh) == (w, h))
            else:
                chk.compare("resize_area", [w, h], None, None, got["pixels"] == exp_img.tobytes().hex())
            exp_gray = cv2.cvtColor(np.ascontiguousarray(img[:, :, ::-1]), cv2.COLOR_BGR2GRAY)
            got = drv({"fn": "gray", "pixels": img.tobytes().hex(), "w": w, "h": h})
            chk.compare("gray", [w, h], None, None, got.get("pixels") == exp_gray.tobytes().hex())

            # layout parsing, thresholds, clipping, NMS, validation errors
            pages = rng.randint(1, 2)
            sizes = [(rng.randint(50, 1500), rng.randint(50, 1500)) for _ in range(pages)]
            payload = {"status": "success", "params": {}, "data": {"labels": [], "boxes": [], "scores": []}}
            for (pw, ph) in sizes:
                k = rng.randint(0, 14)
                labels = [rng.randint(0, 11) for _ in range(k)]
                boxes = [[rng.uniform(-30, pw), rng.uniform(-30, ph), 0, 0] for _ in range(k)]
                for bx in boxes:
                    bx[2] = bx[0] + rng.uniform(0.5, pw / 2)
                    bx[3] = bx[1] + rng.uniform(0.5, ph / 2)
                if k and rng.random() < 0.4:  # near-duplicates for NMS
                    j = rng.randrange(k)
                    boxes.append([v + rng.uniform(-3, 3) for v in boxes[j]])
                    labels.append(labels[j] if rng.random() < 0.7 else rng.randint(0, 11))
                scores = [round(rng.random(), 3) for _ in boxes]
                if rng.random() < 0.05 and boxes:
                    rng.choice([lambda: labels.__setitem__(0, 3.0), lambda: labels.__setitem__(0, 12),
                                lambda: scores.__setitem__(0, 1.5), lambda: boxes[0].__setitem__(2, boxes[0][0])])()
                payload["data"]["labels"].append(labels)
                payload["data"]["boxes"].append(boxes)
                payload["data"]["scores"].append(scores)
            threshold = rng.choice([0.0, 0.3, 0.5])
            nms = rng.choice([0.6, 0.6, None, 0.0])
            exp = py_safe(layout_pp.parse_layout_batch, payload, sizes, threshold, nms)
            got = drv({"fn": "parse_layout", "payload": payload, "sizes": sizes, "threshold": threshold, "nms": nms})
            if "error" in exp:
                ok = "error" in got and exp["error"].split(": ", 1)[1] == got["error"]
            else:
                exp_pages = [[{"idx": b.idx, "type": b.type.value, "bbox": b.bbox, "layout_score": b.layout_score}
                              for b in page] for page in exp["value"]]
                ok = got.get("pages") == exp_pages
            chk.compare("parse_layout", payload, exp if "error" in exp else "pages", got, ok)

            # page cleanup + JSON + Markdown assembly
            case = cleanup_case(rng, nprng)
            for full in (False, True):
                exp = py_safe(python_cleanup, case, full)
                got = drv({"fn": "page_cleanup", **case, "full_overlap": full})
                if "error" in exp:
                    ok = "error" in got
                else:
                    ok = got.get("page") == exp["value"]["page"] and got.get("markdown") == exp["value"]["markdown"]
                chk.compare("page_cleanup:" + ("full" if full else "default"), case if not ok else None,
                            exp, got, ok)
    finally:
        drv.close()
    return chk


def cleanup_case(rng, nprng):
    w, h = 200, 260
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    raw = []
    contents = ["Alpha beta gamma.", "beta gamma", "Alpha", "x = y + 1", "$x = y + 1$", "\\ce{H2O}",
                "Delta epsilon zeta eta.", "Delta epsilon", "zeta eta.", "", "  ", "![](p.png)", "<table></table>",
                "a $b$ c", "# head", "Alpha beta\ngamma delta", "Alpha beta", "gamma delta", "H_{2}O \\to OH"]
    n = rng.randint(1, 10)
    base = rand_boxes(rng, n, w, h)
    for i in range(n):
        t = rng.choice(TYPES)
        bbox = base[i]
        if i and rng.random() < 0.3:  # exact duplicate geometry / containment
            src = raw[rng.randrange(len(raw))]["bbox"]
            bbox = list(src) if rng.random() < 0.6 else [src[0] + 2, src[1] + 2, src[2] - 2, max(src[3] - 2, src[1] + 3)]
        if t in ("Picture", "Chemical Structure") or rng.random() < 0.2:
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            img[y1 + 2:max(y1 + 3, y2 - 2), x1 + 2:max(x1 + 3, x2 - 2)] = nprng.integers(0, 120)
        raw.append({"idx": i, "type": t, "bbox": bbox, "layout_score": rng.choice([None, 0.4, 0.55, 0.9, 0.95])})
    blocks = []
    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        d = dict(raw[i])
        if d["type"] in ("Page-header", "Page-footer") and rng.random() < 0.5:
            continue
        d["content"] = rng.choice(contents)
        d["format"] = {"Formula": "latex", "Table": "html", "Picture": "markdown"}.get(d["type"], "text")
        if d["type"] == "Formula" and rng.random() < 0.5:
            d["text_before"] = rng.choice(["", "(1)", " 2 "])
            d["text_after"] = rng.choice(["", "(3)", "A"])
        blocks.append(d)
    return {"raw": raw, "blocks": blocks, "pixels": img.tobytes().hex(), "w": w, "h": h}


def python_cleanup(case, full):
    image = np.frombuffer(bytes.fromhex(case["pixels"]), np.uint8).reshape(case["h"], case["w"], 3)[:, :, ::-1]
    raw = [make_block(d) for d in case["raw"]]
    blocks = [make_block(d) for d in case["blocks"]]
    if not full:
        blocks = cross_pp.dedupe_systematic_cross_category_blocks(blocks)
    blocks = formula_pp.expand_formula_numbering_blocks(blocks)
    if full:
        from oxr.pipeline import orchestrator
        blocks = orchestrator._dedupe_full_page(deepcopy(raw), blocks, image)
    else:
        blocks = overlap_pp.dedupe_visual_parents(raw, blocks, image)
        blocks = overlap_pp.dedupe_complete_text(blocks)
        blocks = exact_pp.dedupe_exact_textual_blocks(blocks)
    serialized, notes = json_pp.serialize_final_blocks(blocks)
    page = {"page_num": 0, "page_size": {"w": case["w"], "h": case["h"]}, "merge_content_from_pre_page": False,
            "blocks": serialized}
    if notes:
        page["unattached_formula_numbering"] = notes
    return {"page": page, "markdown": formula_pp.space_markdown_math(blocks_to_markdown(blocks))}


# ------------------------------------------------------------------- replay
class ScriptedOXR(crisp_oxr.CrispOXR):
    """CrispOXR with the original helper and a deterministic fake recognizer."""

    def __init__(self, script, **kwargs):  # noqa: D401 - no native library
        import threading
        self._ctx = object()
        self._lock = threading.RLock()
        self._blueprint = None
        self._orchestrator = None
        self._blueprint_path = None
        self.last_trace = []
        self._last_outcomes = {}
        self.max_new_tokens = kwargs.pop("max_new_tokens", 8192)
        self._layout_options = dict(url=None, score_threshold=kwargs.pop("layout_score_threshold", 0.5),
                                    nms_iou_threshold=kwargs.pop("layout_nms_iou_threshold", 0.6),
                                    provides_reading_order=kwargs.pop("layout_provides_reading_order", False))
        self._pipeline_options = dict(image_output_dir=kwargs.pop("image_output_dir", "outputs"),
                                      table_image_placeholders=False,
                                      full_overlap_dedupe=kwargs.pop("full_overlap_dedupe", False))
        self.allow_truncated_content = kwargs.pop("allow_truncated_content", False)
        self.script = script

    def predict(self, image, prompt=None, max_new_tokens=None, *, task="text", repetition_penalty=1.0):
        pixels = np.ascontiguousarray(image, dtype=np.uint8)
        h, w = pixels.shape[:2]
        sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        result = self.script(prompt, sha, max_new_tokens, float(repetition_penalty), w, h)
        self.last_trace.append(dict(prompt=prompt, width=w, height=h, rgb_sha256=sha, max_tokens=max_new_tokens,
                                    repetition_penalty=float(repetition_penalty), result=deepcopy(result)))
        if "raise" in result:
            raise RuntimeError(result["raise"])
        return result

    def close(self):
        if self._blueprint is not None:
            self._blueprint.close()
            self._blueprint = None

    def __del__(self):
        pass


def make_script(rng_seed):
    """Deterministic fake responses keyed on request content, covering every
    recognition/recovery/reading-order branch."""

    def respond(prompt, sha, max_tokens, penalty, w, h):
        r = random.Random(f"{rng_seed}:{prompt}:{sha}:{max_tokens}:{penalty}")
        def done(text, reason="stop"):
            return {"text": text, "eos": reason == "stop", "capped": reason == "length", "finish_reason": reason}
        if prompt.startswith("\nGiven the detected"):
            n = prompt.count("<|object_ref_start|>")
            perm = list(range(n))
            r.shuffle(perm)
            mode = r.random()
            if mode < 0.5:
                return done(" ".join(map(str, perm)))
            if mode < 0.65:
                return done(" ".join(map(str, perm[:-1])))
            if mode < 0.75:
                return done(" ".join(map(str, perm + perm[:1])))
            if mode < 0.85:
                return done("not an order")
            return done(" ".join(map(str, perm[: n // 2])))
        capped = r.random() < 0.15 and penalty == 1.0 and max_tokens > 64
        reason = "length" if capped else "stop"
        if "formula" in prompt:
            return done(r.choice(["$$x^2 \\tag{1}$$", "```latex\nE=mc^2\n```", "\\[a+b \\leqno (2)\\]", "y = 2 \\\\(x)"]), reason)
        if "table" in prompt:
            return done(r.choice(["```html\n<table><tr><td>1</td></tr></table>\n```", "<table><tr><td>a</td><td>b</td></tr></table>"]), reason)
        if "code" in prompt:
            return done(r.choice(["```python\nprint(1)\n```", "x = 1"]), reason)
        if "chemical" in prompt:
            return done(r.choice(["$\\smiles{CCO}$", "```\nC1=CC=CC=C1\n```", "CCN"]), reason)
        return done(r.choice(["Alpha beta gamma.", "Line one\n\n\nline two", "Energy $E=mc^2 \\tag{3}$ holds.",
                              "Überschrift", "日本語のテキスト", "  spaced  ", "a $$b$$ c", ""]), reason)

    return respond


def replay_case(binary, pages, layouts, options, seed, workdir, name):
    script = make_script(seed)
    py_opts = {k: v for k, v in options.items() if k in ("max_new_tokens", "full_overlap_dedupe", "allow_truncated_content",
                                                           "layout_score_threshold", "layout_nms_iou_threshold",
                                                           "layout_provides_reading_order", "image_output_dir")}
    model = ScriptedOXR(script, **py_opts)
    call = {k: v for k, v in options.items() if k in ("keep_header_footer", "model_options", "element_formats")}
    file_name = options.get("file_name", "document.png")
    try:
        py = {"value": model.process_pages(pages, layouts=layouts, file_name=file_name, **call)}
    except Exception as e:  # noqa: BLE001
        py = {"error": f"{type(e).__name__}: {e}"}
    trace = model.last_trace
    model.close()
    page_specs = []
    for i, image in enumerate(pages):
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB")), dtype=np.uint8)
        path = Path(workdir) / f"{name}-page{i}.rgb"
        path.write_bytes(arr.tobytes())
        page_specs.append({"w": arr.shape[1], "h": arr.shape[0], "rgb_path": str(path)})
    native_opts = dict(call, file_name=file_name)
    for k in ("max_new_tokens", "full_overlap_dedupe", "allow_truncated_content", "image_output_dir"):
        if k in options:
            native_opts[k] = options[k]
    for k_py, k_c in (("layout_score_threshold", "score_threshold"), ("layout_nms_iou_threshold", "nms_iou_threshold"),
                      ("layout_provides_reading_order", "provides_reading_order")):
        if k_py in options:
            native_opts[k_c] = options[k_py]
    case = {"pages": page_specs, "layout_response": crisp_oxr.CrispOXR.layout_response_from_regions(layouts),
            "options_json": json.dumps(native_opts), "trace": trace}
    case_path = Path(workdir) / f"{name}.json"
    case_path.write_text(json.dumps(case, ensure_ascii=False))
    out = subprocess.run([binary, "--replay", str(case_path)], capture_output=True, text=True, encoding="utf-8")
    native = json.loads(out.stdout)
    return py, native, trace


def compare_documents(py, native):
    """Return (identical, first differing field)."""
    if "error" in py or "error" in native:
        same = ("error" in py) == ("error" in native)
        return same, None if same else f"error: python={py.get('error')} native={native.get('error')}"
    a, b = deepcopy(py["value"]), deepcopy(native["result"])
    for doc in (a, b):
        doc["metadata"].pop("processing_time_ms", None)
        doc["metadata"].pop("crispembed", None)
    if a == b:
        return True, None

    def walk(x, y, path):
        if type(x) is not type(y) and not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return path
        if isinstance(x, dict):
            for k in list(x) + [k for k in y if k not in x]:
                if k not in x or k not in y:
                    return f"{path}.{k}"
                d = walk(x[k], y[k], f"{path}.{k}")
                if d:
                    return d
            return None
        if isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y)):
                d = walk(u, v, f"{path}[{i}]")
                if d:
                    return d
            return None if len(x) == len(y) else f"{path}.length"
        return None if x == y else path
    return False, walk(a, b, "$")


def synthetic_page(rng, w=900, h=1200):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    regions = []
    y = 40
    for i in range(rng.randint(2, 9)):
        t = rng.choice(LA_TYPES)
        x1 = rng.choice([40.3, 60.5, 470.0])
        x2 = x1 + rng.uniform(200, 400)
        y2 = y + rng.uniform(40, 150)
        draw.rectangle([x1 + 3, y + 3, x2 - 3, y2 - 3], fill=(rng.randint(0, 150),) * 3)
        regions.append({"type": t, "bbox": [x1, y, min(x2, w), y2], "score": round(rng.uniform(0.3, 1.0), 3)})
        if rng.random() < 0.2:
            regions.append({"type": rng.choice(["Text", "Formula"]), "bbox": list(regions[-1]["bbox"]),
                            "score": round(rng.uniform(0.5, 1.0), 3)})
        y = y2 + rng.uniform(-20, 40)
        if y > h - 100:
            break
    return img, regions


def run_replay(binary, cases, seed, out_dir):
    rng = random.Random(seed)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(cases):
            npages = rng.choice([1, 1, 2])
            pages, layouts = zip(*[synthetic_page(rng) for _ in range(npages)])
            options = {"file_name": rng.choice(["doc.png", "Report.PDF", "a.b/c.jpeg"]),
                       "keep_header_footer": rng.random() < 0.3,
                       "allow_truncated_content": rng.random() < 0.7,
                       "full_overlap_dedupe": rng.random() < 0.2,
                       "image_output_dir": os.path.join(tmp, "imgs"),
                       "element_formats": {"cs": "image"} if rng.random() < 0.2 else None,
                       "layout_provides_reading_order": rng.random() < 0.15}
            py, native, trace = replay_case(binary, list(pages), list(layouts), options, seed * 1000 + k, tmp, f"c{k}")
            same, where = compare_documents(py, native)
            req_same = "error" in native or native.get("native_requests") == len(trace)
            results.append({"case": k, "identical": same and req_same, "first_difference": where,
                            "python_requests": len(trace), "native_requests": native.get("native_requests"),
                            "python_error": py.get("error"), "native_error": native.get("error")})
    return results


def run_native(binary, lib, gguf, cases, workdir, threads, live, layout_gguf=None):
    """Real native model. The original helper (CrispOXR.process_pages) runs once
    and its native requests are recorded; the C++ route replays exactly those
    results. Optionally the C++ route also runs live on the same model."""
    from PIL import Image
    results = []
    os.makedirs(workdir, exist_ok=True)
    for spec in cases:
        name, image_path, layout_path = spec.split("=", 1)[0], *spec.split("=", 1)[1].split(":", 1)
        with Image.open(image_path) as opened:
            page = opened.convert("RGB")
        layout = json.loads(Path(layout_path).read_text())["response"]
        img_dir = os.path.join(workdir, name + "-imgs")
        options = {"file_name": Path(image_path).name, "image_output_dir": img_dir}
        entry = {"case": name}
        extra = {"layout_model": layout_gguf} if layout_gguf else {}
        with crisp_oxr.CrispOXR(gguf, n_threads=threads, lib_path=lib, allow_truncated_content=True,
                                image_output_dir=img_dir, **extra) as model:
            try:
                py = {"value": model.process_pages([page], layout_response=layout, file_name=options["file_name"])}
            except Exception as e:  # noqa: BLE001
                py = {"error": f"{type(e).__name__}: {e}"}
            trace = model.last_trace
            Path(workdir, name + "-python.json").write_text(json.dumps({"python": py, "trace": trace},
                                                                        ensure_ascii=False, indent=1))
            arr = np.ascontiguousarray(np.asarray(page), dtype=np.uint8)
            rgb_path = Path(workdir, name + ".rgb")
            rgb_path.write_bytes(arr.tobytes())
            case = {"pages": [{"w": arr.shape[1], "h": arr.shape[0], "rgb_path": str(rgb_path)}],
                    "layout_response": layout, "trace": trace,
                    "options_json": json.dumps(dict(options, allow_truncated_content=True))}
            case_path = Path(workdir, name + "-replay-case.json")
            case_path.write_text(json.dumps(case, ensure_ascii=False))
            out = subprocess.run([binary, "--replay", str(case_path)], capture_output=True, text=True, encoding="utf-8")
            replay = json.loads(out.stdout)
            same, where = compare_documents(py, replay)
            entry.update(replay_identical=same and ("error" in replay or replay.get("native_requests") == len(trace)),
                         replay_first_difference=where, python_requests=len(trace),
                         replay_requests=replay.get("native_requests"), python_error=py.get("error"),
                         replay_error=replay.get("error"))
            if layout_gguf:
                case_nl = dict(case)
                case_nl.pop("layout_response")
                case_nl["layout_gguf"] = layout_gguf
                nl_path = Path(workdir, name + "-replay-native-layout.json")
                nl_path.write_text(json.dumps(case_nl, ensure_ascii=False))
                out = subprocess.run([binary, "--replay", str(nl_path)], capture_output=True, text=True, encoding="utf-8")
                nl = json.loads(out.stdout)
                same, where = compare_documents(py, nl)
                entry.update(native_layout_replay_identical=same, native_layout_replay_first_difference=where,
                             native_layout_replay_error=nl.get("error"))
                if "native_layout_response" in nl:
                    size = [(arr.shape[1], arr.shape[0])]
                    ref_blocks = layout_pp.parse_layout_batch(layout, size, 0.5, 0.6)
                    nat_blocks = layout_pp.parse_layout_batch(nl["native_layout_response"], size, 0.5, 0.6)
                    as_t = lambda pages: [[(b.type.value, [round(v, 3) for v in b.bbox]) for b in p] for p in pages]
                    entry["layout_blocks_reference"] = len(ref_blocks[0])
                    entry["layout_blocks_native"] = len(nat_blocks[0])
                    entry["layout_blocks_identical_types"] = [b.type for b in ref_blocks[0]] == [b.type for b in nat_blocks[0]]
                    entry["layout_blocks_exact"] = [(b.type, b.bbox, b.layout_score) for b in ref_blocks[0]] == \
                        [(b.type, b.bbox, b.layout_score) for b in nat_blocks[0]]
                    entry["layout_blocks_identical_rounded"] = as_t(ref_blocks) == as_t(nat_blocks)
                    Path(workdir, name + "-native-layout.json").write_text(json.dumps(nl["native_layout_response"]))
            if "value" in py:
                meta = py["value"]["metadata"].get("crispembed", {})
                entry["unresolved_capped_requests"] = meta.get("unresolved_capped_requests")
                entry["blocks"] = len(py["value"]["outputs"]["json"]["pages"][0]["blocks"])
                entry["markdown_sha256"] = hashlib.sha256(py["value"]["outputs"]["markdown"].encode()).hexdigest()
            if live:
                os.environ["CRISPEMBED_OXR_NATIVE_DOC"] = "1"
                try:
                    nat = {"result": model.process_pages_native([page], layout_response=layout,
                                                                file_name=options["file_name"])}
                except Exception as e:  # noqa: BLE001
                    nat = {"error": f"{type(e).__name__}: {e}"}
                ntrace = model.last_trace
                same, where = compare_documents(py, nat)
                key = lambda t: (t["prompt"], t["rgb_sha256"], t["max_tokens"], round(float(t["repetition_penalty"]), 6))
                entry.update(live_identical=same, live_first_difference=where, live_requests=len(ntrace),
                             live_request_set_identical=sorted(map(key, trace)) == sorted(map(key, ntrace)),
                             live_error=nat.get("error"))
                Path(workdir, name + "-native.json").write_text(json.dumps({"native": nat, "trace": ntrace},
                                                                            ensure_ascii=False, indent=1))
                if layout_gguf:
                    # End to end: native D-FINE + native C++ assembly vs the
                    # original service-backed Python pipeline (reference layout).
                    try:
                        e2e = {"result": model.process_pages_native([page], file_name=options["file_name"])}
                    except Exception as e:  # noqa: BLE001
                        e2e = {"error": f"{type(e).__name__}: {e}"}
                    same, where = compare_documents(py, e2e)
                    entry.update(e2e_identical=same, e2e_first_difference=where, e2e_requests=len(model.last_trace),
                                 e2e_layout_source=(e2e.get("result") or {}).get("metadata", {}).get(
                                     "crispembed", {}).get("layout_source"), e2e_error=e2e.get("error"))
                    Path(workdir, name + "-e2e.json").write_text(json.dumps({"native": e2e, "trace": model.last_trace},
                                                                             ensure_ascii=False, indent=1))
                try:
                    entry["render_markdown_chars"] = len(model.render_native_document("markdown"))
                    entry["render_html_chars"] = len(model.render_native_document("html"))
                except Exception as e:  # noqa: BLE001
                    entry["render_error"] = str(e)
        results.append(entry)
        print(json.dumps(entry), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["helpers", "replay", "native"])
    ap.add_argument("--lib", help="native library (native mode)")
    ap.add_argument("--gguf", help="OXR GGUF (native mode)")
    ap.add_argument("--case", action="append", default=[], help="native mode: name=image:dfine.json")
    ap.add_argument("--workdir", default="oxr-native-doc-work")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--layout-gguf", help="native mode: OXR D-FINE GGUF for the end-to-end native-layout checks")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.mode == "helpers":
        chk = run_helpers(args.binary, args.cases, args.seed)
        report = {"mode": "helpers", "seed": args.seed, "cases": args.cases,
                  "families": {k: {"checks": v[0], "mismatches": v[1]} for k, v in sorted(chk.counts.items())},
                  "failures": chk.failures}
        ok = not chk.failures
    elif args.mode == "native":
        res = run_native(args.binary, args.lib, args.gguf, args.case, args.workdir, args.threads, not args.no_live,
                         args.layout_gguf)
        report = {"mode": "native", "cases": res}
        ok = bool(res) and all(r.get("replay_identical") and r.get("live_identical", True)
                               and r.get("e2e_identical", True) and r.get("native_layout_replay_identical", True)
                               for r in res)
    else:
        res = run_replay(args.binary, args.cases, args.seed, args.output)
        report = {"mode": "replay", "seed": args.seed, "cases": res,
                  "identical": sum(r["identical"] for r in res), "total": len(res)}
        ok = all(r["identical"] for r in res)
    report["numpy"] = np.__version__
    report["opencv"] = cv2.__version__
    report["python"] = sys.version.split()[0]
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    summary = report.get("families") or {"identical": report.get("identical"), "total": report.get("total"),
                                         "native": report.get("cases") if args.mode == "native" else None}
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
