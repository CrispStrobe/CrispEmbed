"""Unlimited-OCR-2 precision sweep worker (runs inside the cloned repo on Kaggle CPU).

Invoked by kernel.py with the uv-built Python 3.12 venv. Steps:
  1. download the pinned checkpoint (docs/unlimited-ocr2-source.json) and verify sha256
  2. convert F32 (lossless, the quantization source) and F16 (--fp16)
  3. build crispembed-cli + crispembed-quantize
  4. for each precision: produce GGUF (quantized from F32 so 1-D tensors such as norms and
     attn_dash.head_cls stay F32), run the native CLI on the three reference pages, delete
  5. write /kaggle/working/precision/precision.json + summary.md after every case

Quantizer notes (tools/quantize.cpp), checked statically:
  * 1-D tensors (all RMSNorm weights, attn_dash.{q,k}_norm, attn_dash.head_cls[128],
    attn_idx.*_norm) are never quantized (ggml_n_dims < 2) and keep source type.
  * attn_dash.{q,k}_proj [128,1280] are ordinary 2-D matrices (1280 % 256 == 0), so they
    take the requested type unless overridden; the runtime dequantizes them via to_f32.
  * built-in floors: embeddings/lm_head/mlp_gate (router) -> Q8_0, vision (v./c./proj.) -> Q8_0.
  * --tensor-type supports f32/f16/q8_0/q6_k/q5_k (not q4_k), so mixed recipes quantize to
    q4_k and lift the non-expert decoder matrices with overrides.
  * expert down_exps rows are 896 wide (not 256-aligned) -> Q4_K falls back to Q4_0; the
    per-tensor type census below records what actually happened.

The metric helpers are importable without side effects (see tests/test_uocr2_precision_metrics.py).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

# Both the DeepSeek-style [[x1, y1, x2, y2]] and this model's observed text[x1, y1, x2, y2] boxes.
BOX_RE = re.compile(r"\[\[[-\d.,\s\[\]]*\]\]|\[\s*-?\d+(?:\s*,\s*-?\d+){3}\s*\]")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

PRECISIONS = [
    # name, source ("convert-f32"/"convert-f16"/"quantize"), quant type, overrides
    ("f32", "convert-f32", None, []),
    ("f16", "convert-f16", None, []),
    ("q8_0", "quantize", "q8_0", []),
    ("q4_k", "quantize", "q4_k", []),
    # experts Q4_K; attention (q/k/v/o + DASH router q/k + unused indexer), shared experts and the
    # dense layer-0 FFN at Q8_0; router/lm_head/embeddings/vision via the built-in Q8_0 floors.
    ("mix-q8attn-q4exp", "quantize", "q4_k",
     ["l.blk.*.attn_*=q8_0", "l.blk.*.shared_exp.*=q8_0", "l.blk.0.ffn_*=q8_0"]),
    # same with attention/DASH/router/embeddings/lm_head at F16
    ("mix-f16attn-q4exp", "quantize", "q4_k",
     ["l.blk.*.attn_*=f16", "l.blk.*.mlp_gate.weight=f16", "l.blk.*.shared_exp.*=q8_0", "l.blk.0.ffn_*=q8_0",
      "l.embed_tokens.weight=f16", "l.lm_head.weight=f16"]),
]
CASES = ("small-untiled", "para-en", "table")


def strip_boxes(text):
    return BOX_RE.sub("[[]]", text or "")


def box_numbers(text):
    out = []
    for m in BOX_RE.finditer(text or ""):
        out.append([float(x) for x in NUM_RE.findall(m.group(0))])
    return out


def max_box_delta(a, b):
    """Max |coordinate delta| over aligned boxes; None if the box structure differs."""
    ba, bb = box_numbers(a), box_numbers(b)
    if len(ba) != len(bb) or any(len(x) != len(y) for x, y in zip(ba, bb)):
        return None
    return max((abs(x - y) for p, q in zip(ba, bb) for x, y in zip(p, q)), default=0.0)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


def first_diff(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_measured(cmd, env, log_path):
    """Run cmd, return (exit, stdout, stderr, wall_s, peak_rss_mb) with per-child rusage."""
    t0 = time.monotonic()
    with open(log_path + ".out", "w") as fo, open(log_path + ".err", "w") as fe:
        p = subprocess.Popen(cmd, env=env, stdout=fo, stderr=fe)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    out = open(log_path + ".out", errors="replace").read()
    err = open(log_path + ".err", errors="replace").read()
    return p.returncode, out, err, wall, ru.ru_maxrss / 1024.0


def tensor_census(path):
    try:
        import gguf
        r = gguf.GGUFReader(path)
        types = {}
        dash = {}
        for t in r.tensors:
            types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
            if ".attn_dash." in t.name and t.name.startswith("l.blk.1."):
                dash[t.name] = t.tensor_type.name
        return dict(type_counts=types, layer1_dash=dash)
    except Exception as e:  # diagnostic only
        return dict(error=repr(e))


def main():
    t_start = time.monotonic()
    budget = float(os.environ.get("UOCR2_PREC_BUDGET_S", str(11 * 3600)))
    commit = os.environ.get("UOCR2_PREC_COMMIT", "unknown")
    tmp, out_dir = "/kaggle/temp", "/kaggle/working/precision"
    os.makedirs(out_dir, exist_ok=True)
    logs = os.path.join(tmp, "prec-logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable

    def sh(*a, **k):
        print("+", *a, flush=True)
        subprocess.run(list(a), check=True, **k)

    report = dict(schema="crispembed-unlimited-ocr2-precision-v1", commit=commit, steps={}, precisions=[])

    def save():
        json.dump(report, open(os.path.join(out_dir, "precision.json"), "w"), indent=2, ensure_ascii=False)
        open(os.path.join(out_dir, "summary.md"), "w").write(summary_md(report))

    # 1. pinned checkpoint
    c = json.load(open("docs/unlimited-ocr2-source.json"))
    src = os.path.join(tmp, "uocr2-src")
    from huggingface_hub import hf_hub_download
    t0 = time.monotonic()
    for n, p in c["files"].items():
        f = hf_hub_download(c["repository"], n, revision=c["revision"], local_dir=src)
        if "sha256" in p:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(blk)
            assert h.hexdigest() == p["sha256"], n
    report.update(repository=c["repository"], revision=c["revision"])
    report["steps"]["download_s"] = round(time.monotonic() - t0, 1)

    # 2. conversions (single-threaded BLAS for numpy bf16 widening)
    cenv = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=tmp)
    ggufs = {"f32": os.path.join(tmp, "uocr2-f32.gguf"), "f16": os.path.join(tmp, "uocr2-f16.gguf")}
    for name, extra in (("f32", []), ("f16", ["--fp16"])):
        t0 = time.monotonic()
        sh(py, "models/convert-unlimited-ocr-to-gguf.py", "--model-dir", src, "--output", ggufs[name], *extra, env=cenv)
        report["steps"][f"convert_{name}_s"] = round(time.monotonic() - t0, 1)
    shutil.rmtree(src, ignore_errors=True)  # free ~6.7 GB before quantizing

    # 3. build
    t0 = time.monotonic()
    b = os.path.join(tmp, "b")
    sh("cmake", "-S", ".", "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF")
    sh("cmake", "--build", b, "--target", "crispembed-cli", "crispembed-quantize", "--parallel", "4")
    report["steps"]["build_s"] = round(time.monotonic() - t0, 1)
    cli = next(p for p in (os.path.join(b, "crispembed"), os.path.join(b, "bin", "crispembed")) if os.path.exists(p))
    qbin = next(p for p in (os.path.join(b, "crispembed-quantize"), os.path.join(b, "bin", "crispembed-quantize"))
                if os.path.exists(p))

    # reference (mounted kernel output)
    ref_json = next(os.path.join(d, f) for d, _, fs in os.walk("/kaggle/input") for f in fs
                    if f == "reference.json" and d.endswith("uocr2"))
    ref = json.load(open(ref_json))
    base = os.path.dirname(ref_json)
    report["reference_revision"] = ref.get("revision")
    cases = [cs for cs in ref["cases"] if cs["id"] in CASES]
    save()

    for name, kind, qtype, overrides in PRECISIONS:
        entry = dict(precision=name, quant_type=qtype, overrides=overrides, cases=[])
        report["precisions"].append(entry)
        if time.monotonic() - t_start > budget:
            entry["skipped"] = "time budget"
            save()
            continue
        path = ggufs.get(name) or os.path.join(tmp, f"uocr2-{name}.gguf")
        if kind == "quantize":
            cmd = [qbin, ggufs["f32"], path, qtype]
            for o in overrides:
                cmd += ["--tensor-type", o]
            rc, _, qerr, wall, rss = run_measured(cmd, os.environ.copy(), os.path.join(logs, f"quant-{name}"))
            entry.update(quantize_exit=rc, quantize_s=round(wall, 1), quantize_peak_rss_mb=round(rss),
                         quantize_stderr_tail=qerr[-2000:])
            if rc != 0:
                save()
                continue
        entry["file_bytes"] = os.path.getsize(path)
        entry["census"] = tensor_census(path)
        for cs in cases:
            if time.monotonic() - t_start > budget:
                entry["cases"].append(dict(id=cs["id"], skipped="time budget"))
                continue
            tiled = cs["id"] != "small-untiled"
            env = dict(os.environ, CRISPEMBED_UNLIMITED_OCR2="1", UOCR_DBG="1", OMP_NUM_THREADS=str(os.cpu_count()))
            if tiled:
                env["CRISPEMBED_UNLIMITED_OCR2_TILES"] = "1"
            img = os.path.join(base, cs["id"] + ".png")
            rc, out, err, wall, rss = run_measured(
                [cli, "-m", path, "--ocr", img, "--ocr-max-tokens", "384", "-t", str(os.cpu_count())], env,
                os.path.join(logs, f"run-{name}-{cs['id']}"))
            m = re.search(r"gen_ids \(\d+\):([^\n]*)", err)
            ids = [int(x) for x in m.group(1).split()] if m else None
            ref_ids = next((cs[k] for k in ("token_ids", "ids", "output_ids") if k in cs), None)
            text, ref_text = out.strip(), (cs.get("text") or "").strip()
            entry["cases"].append(dict(
                id=cs["id"], tiled=tiled, exit=rc, n_ids=None if ids is None else len(ids),
                n_ref_ids=None if ref_ids is None else len(ref_ids),
                ids_exact=ids is not None and ids == ref_ids, first_id_difference=first_diff(ids, ref_ids),
                text_exact=text == ref_text, text_equal_ignoring_boxes=strip_boxes(text) == strip_boxes(ref_text),
                max_box_delta=max_box_delta(text, ref_text), cer=round(cer(text, ref_text), 6),
                cer_ignoring_boxes=round(cer(strip_boxes(text), strip_boxes(ref_text)), 6),
                wall_s=round(wall, 2), peak_rss_mb=round(rss), text=text,
                stderr_tail=err[-1500:] if rc != 0 or ids is None else ""))
            save()
        if name != "f32":
            os.remove(path)  # f32 stays as the quantization source
        save()
    report["total_s"] = round(time.monotonic() - t_start, 1)
    save()
    print(summary_md(report), flush=True)


def summary_md(r):
    lines = [f"# Unlimited-OCR-2 precision sweep\n", f"commit `{r.get('commit')}`, checkpoint revision "
             f"`{r.get('revision')}`, reference revision `{r.get('reference_revision')}`. CPU, greedy, 384-token cap.\n",
             "| precision | size GB | case | ids exact | first diff | text= (no boxes) | max box d | CER | wall s | peak RSS MB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in r.get("precisions", []):
        size = f"{e['file_bytes'] / 1e9:.2f}" if "file_bytes" in e else "-"
        if e.get("skipped") or not e.get("cases"):
            lines.append(f"| {e['precision']} | {size} | - | {e.get('skipped') or 'quantize exit %s' % e.get('quantize_exit')}"
                         " | | | | | | |")
        for c in e.get("cases", []):
            if c.get("skipped"):
                lines.append(f"| {e['precision']} | {size} | {c['id']} | skipped | | | | | | |")
                continue
            lines.append(f"| {e['precision']} | {size} | {c['id']} | {c['ids_exact']} | {c['first_id_difference']} | "
                         f"{c['text_equal_ignoring_boxes']} | {c['max_box_delta']} | {c['cer']} | {c['wall_s']} | "
                         f"{c['peak_rss_mb']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
