"""Page-level parity: pinned Python helper vs the native MinerU document route.

Both sides receive the SAME recognition outputs. The Python side runs the
unchanged mineru-vl-utils MinerUClientHelper exactly as
CrispMinerU._two_step does (prepare_for_layout, parse_layout_output,
prepare_for_extract, post_process, json2md), with `random.seed(seed)` before
prepare_for_extract so table-image tokens are reproducible. The recognition
outputs are either recorded native GGML outputs (--recorded, from
mineru25_doc_record.py) or synthetic per-prompt outputs for branch coverage.
The native side (test-mineru-doc page CASE.json) replays them.

Compared per case: blocks JSON (json.dumps ensure_ascii=False, compact),
Markdown, the same two rendered from the crisp_marker document model, every
region request (prompt, crop size) and the FNV-1a hash of every crop's RGB
bytes. Crops of tables with embedded images contain token glyphs that are
rendered approximately (not Pillow/FreeType) and are reported separately.

usage:
  python -I page_parity.py BINARY OUTDIR --synthetic N [--seed S]
  python -I page_parity.py BINARY OUTDIR --recorded RECORD.json [...]
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from mineru_vl_utils.mineru_client import DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS, MinerUClientHelper
from mineru_vl_utils.post_process import json2md

logger.remove()
sys.path.insert(0, str(Path(__file__).parent))
import fuzz_postprocess as fz  # noqa: E402  (generators only; main() guarded below)


def fnv1a(b: bytes) -> str:
    h = 1469598103934665603
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


OPTION_KEYS = ("image_analysis", "simple_post_process", "handle_equation_block", "abandon_list",
               "abandon_paratext", "enable_table_formula_eq_wrap")


def helper_for(opts):
    kw = dict(image_analysis=False, simple_post_process=False, handle_equation_block=True, abandon_list=False,
              abandon_paratext=False, enable_table_formula_eq_wrap=False)
    kw.update({k: opts[k] for k in OPTION_KEYS if k in opts})
    return MinerUClientHelper(backend="transformers", prompts=dict(DEFAULT_PROMPTS),
                              sampling_params=dict(DEFAULT_SAMPLING_PARAMS), layout_image_size=(1036, 1036),
                              min_image_edge=28, max_image_edge_ratio=50, debug=False, **kw)


def python_side(image_path, layout_text, recognize, opts, seed):
    """recognize(index, prompt, crop) -> (text, eos, capped)."""
    helper = helper_for(opts)
    with Image.open(image_path) as im:
        page = im.convert("RGB") if im.mode != "RGB" else im.copy()
    layout_image = helper.prepare_for_layout(page)
    blocks = helper.parse_layout_output(layout_text)
    random.seed(seed)
    images, prompts, _params, indices = helper.prepare_for_extract(page, blocks, opts.get("not_extract_list"), None)
    recs = []
    for i, (crop, prompt, idx) in enumerate(zip(images, prompts, indices)):
        text, eos, capped = recognize(i, prompt, crop, blocks[idx])
        if (capped or not eos) and not opts.get("allow_truncated_content", False):
            raise RuntimeError("truncated recognition without allow_truncated_content")
        blocks[idx].content = text
        masked = bool(blocks[idx].get("_table_image_token_map"))
        recs.append({"prompt": prompt, "width": crop.width, "height": crop.height, "text": text, "eos": eos,
                     "capped": capped, "fnv1a": fnv1a(np.asarray(crop, dtype=np.uint8).tobytes()),
                     "masked_table": masked})
    result = helper.post_process(blocks)
    return (json.dumps(list(result), ensure_ascii=False, separators=(",", ":")), json2md(result), recs,
            layout_image.size)


def compare(binary, outdir, name, image_path, layout, recognize, opts, seed):
    blocks_json, md, recs, _ = python_side(image_path, layout["text"], recognize, opts, seed)
    case = {"image_path": str(image_path), "options": dict(opts, table_token_seed=seed), "layout": layout,
            "recognitions": recs}
    case_path = outdir / f"{name}.case.json"
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    p = subprocess.run([binary, "page", str(case_path)], capture_output=True, text=True)
    row = {"case": name, "regions": len(recs)}
    if p.returncode:
        row.update(status="native-error", error=p.stderr.strip()[-500:])
        return row
    nat = json.loads(p.stdout)

    def first_diff(a, b):
        for k, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return {"offset": k, "python": a[max(0, k - 40):k + 80], "native": b[max(0, k - 40):k + 80]}
        return None if len(a) == len(b) else {"offset": min(len(a), len(b)), "python_len": len(a), "native_len": len(b)}

    checks = {
        "blocks_json": first_diff(blocks_json, nat["blocks_json"]),
        "markdown": first_diff(md, nat["markdown"]),
        "document_blocks_json": first_diff(blocks_json, nat["document_blocks_json"]),
        "document_markdown": first_diff(md, nat["document_markdown"]),
    }
    crop_diff = []
    approx = 0
    for i, (r, c) in enumerate(zip(recs, nat["crops"])):
        if r["prompt"] != c["prompt"] or r["width"] != c["width"] or r["height"] != c["height"]:
            crop_diff.append({"index": i, "kind": "request"})
        elif r["fnv1a"] != c["fnv1a"]:
            if r["masked_table"] and c["masked_table"]:
                approx += 1
            else:
                crop_diff.append({"index": i, "kind": "pixels"})
    if len(recs) != len(nat["crops"]):
        crop_diff.append({"kind": "count", "python": len(recs), "native": len(nat["crops"])})
    ok = all(v is None for v in checks.values()) and not crop_diff and not nat["request_mismatch"]
    row.update(status="identical" if ok else "different", first_differences={k: v for k, v in checks.items() if v},
               crop_differences=crop_diff, masked_table_crops_with_approximate_glyphs=approx,
               request_mismatch=nat["request_mismatch"], bytes_json=len(blocks_json), bytes_markdown=len(md))
    return row


# ------------------------------------------------------------- synthetic ----
TYPES_ALL = ["text", "title", "doc_title", "paragraph_title", "table", "equation", "formula_number", "code",
             "algorithm", "aside_text", "ref_text", "index", "phonetic", "list_item", "table_caption",
             "image_caption", "code_caption", "caption", "table_footnote", "image_footnote", "footnote", "header",
             "footer", "page_number", "page_footnote", "image", "chart", "list", "image_block", "equation_block",
             "unknown", "inline_formula"]


def synth_page(rng, path, w, h):
    im = Image.new("RGB", (w, h), (rng.randint(200, 255),) * 3)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(10, 60)):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        x1, y1 = min(w, x0 + rng.randint(1, w // 2)), min(h, y0 + rng.randint(1, h // 3))
        d.rectangle([x0, y0, x1, y1], fill=tuple(rng.randrange(256) for _ in range(3)))
    arr = np.asarray(im).copy()
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(0, 30, arr.shape, dtype=np.uint8)
    Image.fromarray(arr ^ noise).save(path)


def box(rng, x0=None, y0=None, w=None, h=None):
    x0 = rng.randint(0, 900) if x0 is None else x0
    y0 = rng.randint(0, 900) if y0 is None else y0
    x1 = min(1000, x0 + (rng.randint(1, 400) if w is None else w))
    y1 = min(1000, y0 + (rng.randint(1, 200) if h is None else h))
    return x0, y0, x1, y1


def tok(b, t, rot=None, tail=""):
    r = {None: "", 0: "<|rotate_up|>", 90: "<|rotate_right|>", 180: "<|rotate_down|>", 270: "<|rotate_left|>"}[rot]
    return f"<|box_start|>{b[0]} {b[1]} {b[2]} {b[3]}<|box_end|><|ref_start|>{t}<|ref_end|>{r}{tail}"


def synth_layout(rng, scenario):
    parts = []
    rots = [None, 0, 90, 180, 270]
    if scenario == "table_images":
        tb = box(rng, rng.randint(50, 300), rng.randint(50, 300), rng.randint(300, 600), rng.randint(200, 500))
        rot = rng.choice(rots)
        parts.append(tok(tb, "table", rot))
        for _ in range(rng.randint(1, 4)):
            ix = rng.randint(tb[0], tb[2] - 20)
            iy = rng.randint(tb[1], tb[3] - 20)
            parts.append(tok((ix, iy, min(tb[2], ix + rng.randint(10, 120)), min(tb[3], iy + rng.randint(10, 80))),
                             "image", rng.choice(rots)))
        parts.append(tok(box(rng, tb[0] + 5, tb[1] + 5, 30, 20), "text"))  # internal text, filtered
    if scenario == "equations":
        eb = box(rng, 100, 100, 600, 300)
        parts.append(tok(eb, "equation_block"))
        for k in range(rng.randint(1, 3)):
            parts.append(tok((eb[0] + 10, eb[1] + 10 + 60 * k, eb[2] - 10, eb[1] + 60 + 60 * k), "equation"))
    if scenario == "fallback":
        # equation_block covering >1 unextracted equation: upstream raises
        # TypeError in post_process and returns the helper fallback list.
        parts.append(tok((100, 100, 700, 400), "equation_block"))
        parts.append(tok((110, 110, 690, 200), "equation"))
        parts.append(tok((110, 250, 690, 390), "equation"))
        parts.append(tok((100, 500, 700, 600), "list_item"))
    if scenario == "images":
        ib = box(rng, 100, 100, 500, 500)
        parts.append(tok(ib, "image_block"))
        parts.append(tok((ib[0] + 10, ib[1] + 10, ib[0] + 200, ib[1] + 200), "image"))
        parts.append(tok((ib[0] + 10, ib[1] + 300, ib[0] + 300, ib[1] + 330), "image_caption"))
        parts.append(tok(box(rng, 600, 600, 300, 300), rng.choice(["image", "chart", "unknown"])))
        parts.append(tok(box(rng, 50, 900, 60, 50), "chart"))  # small, may be ineligible
    for _ in range(rng.randint(2, 12)):
        t = rng.choice(TYPES_ALL)
        tail = "txt_contd_tgt" if t == "text" and rng.random() < .3 else ""
        parts.append(tok(box(rng), t if rng.random() > .05 else t.upper(), rng.choice(rots), tail))
    if rng.random() < .2:
        parts.append(tok((1, 1, 1001, 5), "text"))
    if rng.random() < .2:
        parts.append(tok((10, 10, 11, 600), "text"))  # extreme aspect -> padding
    if rng.random() < .2:
        parts.append(tok((10, 10, 12, 13), "text"))  # tiny -> min-edge resize / empty crop
    if rng.random() < .2:
        parts.append(parts[rng.randrange(len(parts))])  # duplicate -> dedup
    return "".join(parts)


def synth_text(rng, prompt, block, crop):
    t = block.type
    if prompt == "\nTable Recognition:":
        out = fz.gen_otsl(rng)
        if rng.random() < .5:
            out += "<fcel>" + rng.choice(["\\(x^2\\)", "\\[a+b\\]", "<eq>k</eq> \\(y\\)", "\\( \\)"]) + "<nl>"
        tokens = list(block.get("_table_image_token_map", {}))
        for tkn in tokens:
            out += "<fcel>" + rng.choice([tkn, tkn.replace("[", "[ "), "x"]) + "<nl>"
        return out
    if prompt == "\nFormula Recognition:":
        return fz.gen_latex(rng) if rng.random() < .9 else "[Non-Text]"
    if prompt == "\nImage Analysis:":
        return fz.gen_image(rng)
    r = rng.random()
    if r < .08:
        return " [Non-Text] "
    if r < .12:
        return ""
    if t in ("header", "footer") and r < .3:
        return "[Non-Text]"
    return fz.gen_text(rng) if rng.random() < .6 else rng.choice(["Hello world", "中文段落", "line\nbreak", "x" * 50])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("outdir")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--recorded", action="append", default=[])
    ap.add_argument("--report")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(a.seed)
    scenarios = ["table_images", "equations", "images", "mixed", "fallback"]
    for i in range(a.synthetic):
        scen = scenarios[i % len(scenarios)]
        w, h = rng.randint(200, 1400), rng.randint(200, 1400)
        img = out / f"synthetic-{i}.png"
        synth_page(rng, img, w, h)
        opts = {"image_analysis": rng.random() < .7}
        if i % 9 == 3:
            opts["simple_post_process"] = True
        if i % 11 == 5:
            opts["handle_equation_block"] = False
        if i % 7 == 2:
            opts["abandon_list"] = opts["abandon_paratext"] = True
        if i % 3 == 1:
            opts["enable_table_formula_eq_wrap"] = True
        if scen == "fallback":
            opts["not_extract_list"] = ["equation"]
        if i % 13 == 6:
            opts["not_extract_list"] = rng.sample(["equation", "table", "text", "image", "bogus"], 2)
        if i % 17 == 8:
            opts["allow_truncated_content"] = True
        layout_text = synth_layout(rng, scen)
        sub = random.Random(rng.randrange(1 << 30))
        cap_at = sub.randrange(5) if opts.get("allow_truncated_content") else -1

        def recognize(k, prompt, crop, block, sub=sub, cap_at=cap_at):
            return synth_text(sub, prompt, block, crop), k != cap_at, k == cap_at

        layout = {"text": layout_text, "eos": True, "capped": False}
        try:
            rows.append(compare(a.binary, out, f"synthetic-{i}-{scen}", img, layout, recognize, opts,
                                seed=rng.randrange(1 << 31)))
        except RuntimeError as e:
            rows.append({"case": f"synthetic-{i}", "status": "python-error", "error": str(e)})
    for rec_path in a.recorded:
        rec = json.loads(Path(rec_path).read_text(encoding="utf-8"))
        for c in rec["cases"]:
            outs = c["recognitions"]

            def recognize(k, prompt, crop, block, outs=outs):
                r = outs[k]
                if r["prompt"] != prompt or r["width"] != crop.width or r["height"] != crop.height:
                    raise RuntimeError(f"recorded request {k} does not match the Python helper request")
                return r["text"], r["eos"], r["capped"]

            rows.append(compare(a.binary, out, c["case"], c["image_path"], c["layout"], recognize, c["options"],
                                c["seed"]))
    summary = {"python": sys.version.split()[0], "cases": len(rows),
               "identical": sum(r["status"] == "identical" for r in rows), "rows": rows}
    for r in rows:
        print(f"{r['case']:40s} {r['status']:12s} regions={r.get('regions', '-')}"
              f" approx_glyph_crops={r.get('masked_table_crops_with_approximate_glyphs', 0)}"
              + (f" {r.get('first_differences') or ''}{r.get('crop_differences') or ''}{r.get('error', '')}"
                 if r["status"] != "identical" else ""))
    print(f"{summary['identical']}/{summary['cases']} identical")
    if a.report:
        Path(a.report).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(0 if summary["identical"] == summary["cases"] else 1)


if __name__ == "__main__":
    main()
