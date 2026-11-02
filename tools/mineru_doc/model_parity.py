"""Model-backed parity: Python helper route vs pure native document route.

For each protocol case, one resident CrispMinerU context runs
  A) two_step_extract: native GGML inference + the pinned mineru-vl-utils
     helper (random.seed(seed) before the call, for table-image tokens), and
  B) native_two_step_extract: the C++ document route (table_token_seed=seed).
The outputs are compared byte-for-byte (blocks JSON with
json.dumps(ensure_ascii=False, separators=(",", ":")), Markdown via json2md),
and per region: prompt, crop size, generated token IDs, text, EOS and cap.
Recorded Python-side recognition (A) is written for the weight-free native
replay (page_parity.py --recorded). Pages are decoded once with Pillow and
saved as PNG so both routes and the replay see identical pixels.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

from crispembed import CrispMinerU

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--image-analysis", action="store_true", default=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT / "docs/mineru25-pro-publication-v1.json").read_text())["case_inputs"]
    ids = list(inputs) if a.cases == "all" else a.cases.split(",")
    model = CrispMinerU(a.gguf, n_threads=a.threads, lib_path=a.lib, image_analysis=a.image_analysis,
                        allow_truncated_content=True)
    rows, record = [], {"cases": []}
    for n, cid in enumerate(ids):
        image_rel, task = inputs[cid]
        with Image.open(ROOT / "tests/regression/images" / image_rel) as im:
            page = CrispMinerU._pil_image(im).copy()
        png = out / f"{cid}.png"
        page.save(png)
        seed = 1000 + n
        row = {"case": cid, "image": image_rel, "task": task, "seed": seed}
        try:
            random.seed(seed)
            t0 = time.perf_counter()
            py_blocks = model.two_step_extract(page, max_new_tokens=a.max_tokens)
            row["python_route_s"] = time.perf_counter() - t0
            trace = list(model.last_trace)
            py_json = json.dumps(list(py_blocks), ensure_ascii=False, separators=(",", ":"))
            py_md = model.to_markdown(py_blocks)
            t0 = time.perf_counter()
            model.native_two_step_extract(page, max_new_tokens=a.max_tokens, table_token_seed=seed)
            row["native_route_s"] = time.perf_counter() - t0
            nat = model.last_native_document
            nat_json = model.native_render("mineru-json")
            nat_md = model.native_render("mineru-markdown")
            (out / f"{cid}.python.json").write_text(py_json, encoding="utf-8")
            (out / f"{cid}.python.md").write_text(py_md, encoding="utf-8")
            (out / f"{cid}.native.json").write_text(nat_json, encoding="utf-8")
            (out / f"{cid}.native.md").write_text(nat_md, encoding="utf-8")
            (out / f"{cid}.native-document.json").write_text(json.dumps(nat, ensure_ascii=False), encoding="utf-8")
            (out / f"{cid}.marker.md").write_text(model.native_render("markdown"), encoding="utf-8")

            def diff(x, y):
                for k, (p, q) in enumerate(zip(x, y)):
                    if p != q:
                        return {"offset": k, "python": x[max(0, k - 60):k + 100], "native": y[max(0, k - 60):k + 100]}
                return None if len(x) == len(y) else {"offset": min(len(x), len(y)), "python_len": len(x),
                                                       "native_len": len(y)}

            row["json"] = diff(py_json, nat_json) or "identical"
            row["markdown"] = diff(py_md, nat_md) or "identical"
            row["markdown_result_field"] = "identical" if nat["markdown"] == py_md else "different"
            native_reqs = [nat["layout"]] + [r["prediction"] for r in nat["recognitions"]]
            req = []
            for k, (p, q) in enumerate(zip(trace, native_reqs)):
                pr = p["result"]
                same = (p["prompt"] == q["prompt"] and p["width"] == q["image_width"] and p["height"] == q["image_height"]
                        and pr["token_ids"] == q["token_ids"] and pr["text"] == q["text"]
                        and pr["eos"] == q["eos"] and pr["capped"] == q["capped"])
                if not same:
                    req.append({"index": k, "prompt": p["prompt"], "python_size": [p["width"], p["height"]],
                                "native_size": [q["image_width"], q["image_height"]],
                                "token_ids_equal": pr["token_ids"] == q["token_ids"]})
            row["requests"] = {"python": len(trace), "native": len(native_reqs), "differences": req}
            row["capped_regions"] = sum(bool(p["result"]["capped"]) for p in trace)
            row["status"] = ("identical" if row["json"] == "identical" and row["markdown"] == "identical"
                             and not req and len(trace) == len(native_reqs) else "different")
            record["cases"].append({
                "case": cid, "image_path": str(png), "seed": seed,
                "options": {"image_analysis": a.image_analysis, "allow_truncated_content": True},
                "layout": {"text": trace[0]["result"]["text"], "eos": trace[0]["result"]["eos"],
                           "capped": trace[0]["result"]["capped"]},
                "recognitions": [{"prompt": p["prompt"], "width": p["width"], "height": p["height"],
                                  "text": p["result"]["text"], "eos": p["result"]["eos"],
                                  "capped": p["result"]["capped"]} for p in trace[1:]]})
        except Exception as e:  # noqa: BLE001 - retain the failure in the receipt
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("case", "status", "requests", "json", "markdown")},
                         ensure_ascii=False)[:600], flush=True)
    Path(a.record).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    summary = {"model": Path(a.gguf).name, "max_new_tokens": a.max_tokens, "image_analysis": a.image_analysis,
               "python": sys.version.split()[0], "identical": sum(r["status"] == "identical" for r in rows),
               "cases": len(rows), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "summary.txt", "w") as f:
        for r in rows:
            f.write(f"{r['case']:14s} {r['status']:10s} regions={r.get('requests', {}).get('python', '-')} "
                    f"capped={r.get('capped_regions', '-')} py={r.get('python_route_s', 0):.1f}s "
                    f"native={r.get('native_route_s', 0):.1f}s\n")
        f.write(f"{summary['identical']}/{summary['cases']} identical\n")
    model.close()
    sys.exit(0 if summary["identical"] == summary["cases"] else 1)


if __name__ == "__main__":
    main()
