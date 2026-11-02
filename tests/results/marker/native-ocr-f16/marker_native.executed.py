#!/usr/bin/env python3
"""Sequential real-GGUF native document tests; pipeline and OCR quality are separate."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from crispembed import CrispMarker


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(data)
    return h.hexdigest()


def walk(node):
    yield node
    for child in node.get("children") or []: yield from walk(child)


class Visible(HTMLParser):
    def __init__(self): super().__init__(); self.data = []; self.cells = []; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"): self.cell = []
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cells.append("".join(self.cell)); self.cell = None
    def handle_data(self, data):
        self.data.append(data)
        if self.cell is not None: self.cell.append(data)


def normalize(text): return " ".join(text.split())
def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True); p.add_argument("--model", required=True)
    p.add_argument("--expected-model-sha256", required=True); p.add_argument("--pdfium", required=True)
    p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--panel", choices=("synthetic", "full"), default="synthetic")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    os.environ["CRISPEMBED_MARKER"] = "1"; os.environ["CRISPEMBED_TELEOCR"] = "1"
    os.environ["CRISPEMBED_OFFLINE"] = "1"
    synthetic_gold = json.loads((ROOT / "tests/fixtures/teleocr/synthetic-text.json").read_text())
    pdf_gold = json.loads((ROOT / "tests/fixtures/marker/expected.json").read_text())
    digital = next(c for c in pdf_gold["cases"] if c["id"] == "simpledigital")
    table = next(c for c in pdf_gold["cases"] if c["id"] == "table")
    cases = [("synthetic-image", ROOT / "tests/fixtures/teleocr/synthetic-text.png", synthetic_gold["gold_text"], None)]
    if args.panel == "full":
        cases += [("digital-auto", ROOT / "tests/fixtures/marker/simpledigital.pdf", "\n".join(digital["logical_order"]), None),
                  ("table-image", ROOT / "tests/fixtures/marker/table-page-1.png", "\n".join(table["logical_order"]), table["expected_table"])]
    manifest = {
        "kind":"native-marker-real-model-panel", "model_file":Path(args.model).name,
        "expected_model_sha256":args.expected_model_sha256,
        "library_sha256":sha(args.library), "source_sha256":{
            str(f.relative_to(ROOT)):sha(f) for f in [ROOT / "src/marker.cpp", ROOT / "src/marker_backend.cpp",
            ROOT / "src/marker_document.cpp", ROOT / "src/qwen2vl_ocr.cpp", ROOT / "tests/marker_native.py"]},
        "max_tokens":1024,"max_layout_tokens":1024,"threads":args.threads,"backend":"CPU",
        "cases":[{"id":name,"input":str(path.relative_to(ROOT)),"sha256":sha(path),"gold_text":gold,"table":grid}
                 for name,path,gold,grid in cases],
        "quality_protocol":"Whitespace collapse only; preserve punctuation/case/Unicode; exact cells scored separately.",
        "surya_parity":False,"teleocr_strict_numerical_qualification":False,
    }
    # Freeze inputs and thresholds before loading or running any model.
    (out / "declared-panel.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    actual_hash = sha(args.model)
    if actual_hash != args.expected_model_sha256: raise RuntimeError("actual model hash differs from declared checkpoint")
    results = []; start = time.monotonic()
    with CrispMarker(args.model, pdfium_path=args.pdfium, mode="auto", n_threads=args.threads,
                     max_tokens=1024,max_layout_tokens=1024,lib_path=args.library) as model:
        for name,path,gold,grid in cases:
            before = time.monotonic()
            item = {"id":name}
            try:
                doc = json.loads(model.convert(path,"json"))
                for fmt in ("markdown","html","json","chunks"):
                    (out / f"{name}.{fmt}").write_text(model.render(fmt),encoding="utf-8")
                metadata = model.metadata
                (out / f"{name}.meta.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n")
                for asset_name,data in model.assets.items(): (out / asset_name).write_bytes(data)
                parsed = Visible(); parsed.feed(model.render("html"))
                actual = normalize(" ".join(parsed.data)); expected = normalize(gold)
                nodes = [n for n in walk(doc) if not n.get("children") and n.get("block_type") not in ("Document","Page")]
                pages = doc["children"]
                checks = {
                    "has_layout_tokens":any(p.get("layout_generated_tokens",0)>0 for p in pages),
                    "no_layout_cap":all(not p.get("layout_token_cap_reached",False) for p in pages),
                    "no_block_cap":all(not n.get("token_cap_reached",False) for n in nodes),
                    "valid_confidences":all(math.isfinite(n.get("confidence",0)) and 0<=n.get("confidence",0)<=1 for n in nodes),
                    "document_nonempty":bool(actual),
                    "actual_backend":metadata["backend"]=="crispembed-teleocr",
                }
                if name == "digital-auto":
                    checks["embedded_text_used"] = any(n.get("source","").startswith("pdftext") for n in nodes)
                else: checks["native_recognition_ran"] = any(n.get("generated_tokens",0)>0 for n in nodes)
                quality = {"text_exact":actual==expected,
                           "CER":distance(list(expected),list(actual))/max(1,len(expected)),
                           "WER":distance(expected.split(),actual.split())/max(1,len(expected.split()))}
                if grid is not None:
                    expected_cells = [normalize(cell) for row in grid for cell in row]
                    quality["cells_exact"] = [normalize(c) for c in parsed.cells]==expected_cells
                    quality["cells_expected"] = expected_cells; quality["cells_actual"] = parsed.cells
                item.update(checks=checks,functional_passed=all(checks.values()),quality=quality,
                            quality_passed=quality["text_exact"] and quality.get("cells_exact",True),
                            actual_text=actual,metadata=metadata)
            except Exception as exc: item.update(functional_passed=False,quality_passed=False,error=repr(exc))
            item["seconds"] = time.monotonic()-before
            results.append(item)
            (out / "progress.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
            print(json.dumps({k:item[k] for k in ("id","functional_passed","quality_passed","seconds")}),flush=True)
    report = {**manifest,"model_sha256":actual_hash,"results":results,"seconds":time.monotonic()-start,
              "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "functional_passed":all(r["functional_passed"] for r in results),
              "quality_passed":all(r["quality_passed"] for r in results)}
    (out / "summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return 0 if report["functional_passed"] and report["quality_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
