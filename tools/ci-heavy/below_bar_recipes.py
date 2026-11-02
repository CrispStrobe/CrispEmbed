#!/usr/bin/env python3
"""Repair published registry artifacts that the CUDA audit (PR #98) found below the
owner's precision bars, on a GitHub-hosted CPU runner.

    gh workflow run heavy-cpu.yml --ref fix/below-bar-artifacts \
        -f script=tools/ci-heavy/below_bar_recipes.py -f args="sweep f2llm" \
        -f isolated_lane=below-bar-f2llm -f pip="torch transformers>=4.51,<5 sentence-transformers gguf"

MODEL is f2llm (codefuse-ai/F2LLM-v2-0.6B), clip (openai/clip-vit-base-patch16 text
tower) or bgem3 (BAAI/bge-m3 with its sparse and ColBERT heads).

Phase `sweep` (no uploads):
  1. original Python reference on the panel: the audit's 13 texts plus the 65-text
     held-out imatrix eval corpus (German/English prose, code, newline-heavy text,
     queries through the model's own query prompt). F2LLM via sentence-transformers,
     CLIP via transformers get_text_features, BGE-M3 via FlagEmbedding BGEM3FlagModel
     (dense, lexical weights and ColBERT vectors). Revisions are pinned and recorded.
  2. the F32 source GGUF (published file for F2LLM/CLIP; for BGE-M3 a fresh
     `--crisp` conversion that keeps the heads) is checked against that reference.
  3. the published below-bar files are re-measured, then diagnostic arms separate
     weight error from activation quantisation (a Q4_K file dequantised to F16 runs
     F16 matmuls with exactly the Q4_K weights), then per-class and per-layer
     sensitivity, then a greedy precision recipe that adds the most effective
     Q8_0 groups until the bar is cleared with margin on BOTH references.
Phase `publish`: rebuilds the recipes recorded in tests/results/below-bar/recipes.json,
re-qualifies each output on the same panel (refuses to upload below the bar), uploads
new immutable files with models/upload_to_hf.py's publish_revision(), and re-verifies
an empty-cache download from the returned commit.

All metrics are minimum and mean cosine over every panel text (ColBERT: over every
content token, aligned to FlagEmbedding, which drops the CLS position). Bars:
F32 0.99, F16 0.98, Q8_0 0.97, Q4_K 0.94 (docs/precision-acceptance.md).
"""
import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path(os.environ.get("HEAVY_OUT", "/tmp/below-bar-out"))
S = Path(os.environ.get("HEAVY_SCRATCH", "/tmp/below-bar-scratch"))
OUT.mkdir(parents=True, exist_ok=True)
S.mkdir(parents=True, exist_ok=True)
NT = str(os.cpu_count() or 4)
BARS = {"f32": 0.99, "f16": 0.98, "q8_0": 0.97, "q6_k": 0.97, "q5_k": 0.94, "q4_k": 0.94, "iq4_xs": 0.94}
MARGIN = 0.01  # greedy recipes stop only once every metric clears bar + MARGIN
F2LLM_QUERY = "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: "

CFG = {
    "f2llm": {
        "hf": "codefuse-ai/F2LLM-v2-0.6B", "repo": "cstr/f2llm-v2-0.6b-GGUF", "src": "f2llm-v2-0.6b.gguf",
        "published": ["f2llm-v2-0.6b-q4_k-imatrix.gguf", "f2llm-v2-0.6b-q4_k-imatrix-c2.gguf",
                      "f2llm-v2-0.6b-q4_k.gguf", "f2llm-v2-0.6b-q8_0.gguf"],
        "query_prefix": F2LLM_QUERY, "modes": ["embed"], "stem": "f2llm-v2-0.6b",
    },
    "clip": {
        "hf": "openai/clip-vit-base-patch16", "repo": "cstr/clip-text-base-GGUF", "src": "clip-text-base.gguf",
        "published": ["clip-text-base-q4_k.gguf", "clip-text-base-q4_k-imatrix.gguf", "clip-text-base-iq4_xs.gguf",
                      "clip-text-base-q8_0.gguf"],
        "query_prefix": "", "modes": ["embed"], "stem": "clip-text-base",
    },
    "bgem3": {
        "hf": "BAAI/bge-m3", "repo": "cstr/bge-m3-GGUF", "src": None,
        "published": [], "query_prefix": "", "modes": ["embed", "sparse", "colbert"], "stem": "bge-m3-heads",
    },
}


def log(*a):
    print(*a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd, env=None, timeout=7200):
    t = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout)
    return r, time.time() - t


# ───────────────────────────── build + inputs ─────────────────────────────
def build():
    b = S / "build-cpu"
    r, _ = sh(["cmake", "-S", ".", "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
               "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
               "-DCRISPEMBED_BUILD_SHARED=OFF"])
    (OUT / "configure.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit("configure failed")
    r, dt = sh(["cmake", "--build", b, "--parallel", NT, "--target", "crispembed-cli", "crispembed-quantize"])
    (OUT / "build.log").write_text(r.stdout[-20000:] + r.stderr[-20000:])
    if r.returncode:
        log(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("build failed")
    find = lambda n: next(p for p in b.rglob(n) if p.is_file() and os.access(p, os.X_OK))
    log(f"build {dt:.0f}s")
    return find("crispembed"), find("crispembed-quantize")


def corpus(stem):
    p = ROOT / "tools/kaggle/crispembed-imatrix-quant" / f"{stem}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(r.get("role", "doc"), r["text"]) for r in rows if r.get("text")]


def panel(cfg):
    audit = [l for l in (ROOT / "tests/results/gpu-audit/metal-m1/embed_texts.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ev = corpus("eval_corpus")
    docs = audit + [t for r, t in ev if not r.startswith("query")]
    queries = [t for r, t in ev if r.startswith("query")]
    if not cfg["query_prefix"]:  # no query prompt for this model: queries are plain texts
        docs, queries = docs + queries, []
    cal = corpus("calib_corpus")
    cdocs = [t for r, t in cal if not r.startswith("query")]
    cq = [t for r, t in cal if r.startswith("query")]
    if not cfg["query_prefix"]:
        cdocs, cq = cdocs + cq, []
    assert not any(t.startswith("-") for t in docs + queries + cdocs + cq)
    return {"docs": docs, "queries": queries, "n_audit": len(audit)}, {"docs": cdocs, "queries": cq}


def hf_snapshot(repo):
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi().model_info(repo).sha
    path = snapshot_download(repo, revision=sha, cache_dir=str(S / "hf-src"),
                             ignore_patterns=["onnx/*", "*.onnx*", "img/*", "imgs/*", "*.h5", "*.msgpack",
                                              "*.ot", "openvino/*"])
    return sha, path


def fetch(repo, name, rev="main"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, revision=rev, cache_dir=str(S / "hf-gguf"))


# ───────────────────────────── references ─────────────────────────────
def reference(model, cfg, P):
    import torch
    torch.set_num_threads(int(NT))
    sha, path = hf_snapshot(cfg["hf"])
    t0 = time.time()
    ref = {"hf_repo": cfg["hf"], "hf_revision": sha}
    if model == "f2llm":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(path, device="cpu", model_kwargs={"torch_dtype": torch.float32})
        enc = lambda xs, pr: [v.tolist() for v in m.encode(xs, batch_size=1, normalize_embeddings=True, prompt=pr)]
        ref["embed"] = enc(P["docs"], None) + enc(P["queries"], cfg["query_prefix"])
        ref["impl"] = "sentence_transformers.SentenceTransformer (fp32, batch 1)"
    elif model == "clip":
        from transformers import CLIPModel, CLIPTokenizer
        m = CLIPModel.from_pretrained(path, torch_dtype=torch.float32).eval()
        tok = CLIPTokenizer.from_pretrained(path)
        vs = []
        with torch.no_grad():
            for t in P["docs"]:
                e = tok([t], return_tensors="pt", truncation=True, max_length=77)
                f = m.get_text_features(**e)
                f = f if torch.is_tensor(f) else getattr(f, "text_embeds", f[0])
                vs.append((f[0] / f[0].norm()).tolist())
        ref["embed"] = vs
        ref["impl"] = "transformers CLIPModel.get_text_features (fp32, truncation 77)"
    else:
        from FlagEmbedding import BGEM3FlagModel
        try:
            m = BGEM3FlagModel(path, use_fp16=False, devices="cpu")
        except TypeError:
            m = BGEM3FlagModel(path, use_fp16=False, device="cpu")
        o = m.encode(P["docs"], batch_size=1, max_length=8192, return_dense=True, return_sparse=True,
                     return_colbert_vecs=True)
        ref["embed"] = [np.asarray(v, np.float32).tolist() for v in o["dense_vecs"]]
        ref["sparse"] = [{int(k): float(v) for k, v in dict(d).items()} for d in o["lexical_weights"]]
        ref["colbert"] = [np.asarray(v, np.float32).tolist() for v in o["colbert_vecs"]]
        ref["impl"] = "FlagEmbedding.BGEM3FlagModel (fp32, batch 1, max_length 8192)"
    ref["seconds"] = time.time() - t0
    del m
    gc.collect()
    return ref, path


# ───────────────────────────── native runs ─────────────────────────────
def native(cli, gguf, P, mode, prefix_q, env=None):
    flag = {"embed": [], "sparse": ["--sparse"], "colbert": ["--colbert"]}[mode]
    res, wall = [], 0.0
    for texts, pre in ((P["docs"], ""), (P["queries"], prefix_q)):
        if not texts:
            continue
        r, dt = sh([cli, "-m", gguf, "-t", NT, "--json", "--prefix", pre] + flag + texts, env=env)
        wall += dt
        if r.returncode:
            raise RuntimeError(f"{mode} rc={r.returncode}: {r.stderr[-1500:]}")
        try:
            j = json.loads(r.stdout)
        except json.JSONDecodeError:  # some engines (CLIP text) print one JSON object per line
            j = [json.loads(l.rstrip().rstrip(",")) for l in r.stdout.splitlines() if l.strip().startswith("{")]
        if len(j) != len(texts):
            raise RuntimeError(f"{mode}: {len(j)} outputs for {len(texts)} texts")
        if mode == "embed":
            res += [x["embedding"] for x in j]
        elif mode == "sparse":
            res += [{int(e["token_id"]): float(e["weight"]) for e in x["sparse"]} for x in j]
        else:
            res += [x["vectors"] for x in j]
    return res, wall


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def compare(mode, x, y, drop_cls_x=False):
    """x = native candidate, y = reference (original or native F32)."""
    if mode == "embed":
        cs = [cos(a, b) for a, b in zip(x, y)]
        return {"min": min(cs), "mean": float(np.mean(cs)), "worst_text": int(np.argmin(cs)),
                "per_text": [round(c, 6) for c in cs]}
    if mode == "sparse":
        cs, same = [], 0
        for a, b in zip(x, y):
            k = sorted(set(a) | set(b))
            cs.append(cos([a.get(i, 0.0) for i in k], [b.get(i, 0.0) for i in k]))
            same += set(a) == set(b)
        return {"min": min(cs), "mean": float(np.mean(cs)), "worst_text": int(np.argmin(cs)),
                "same_term_sets": same, "n": len(cs), "per_text": [round(c, 6) for c in cs]}
    cs, cls, counts = [], [], 0
    for ti, (a, b) in enumerate(zip(x, y)):
        a = np.asarray(a)
        b = np.asarray(b)
        if drop_cls_x:
            a = a[1:]
        else:  # native vs native: report the CLS row separately (the original never emits it)
            cls.append(cos(a[0], b[0]))
            a, b = a[1:], b[1:]
        counts += len(a) == len(b)
        n = min(len(a), len(b))
        cs += [(cos(a[i], b[i]), ti, i) for i in range(n)]
    v = [c for c, _, _ in cs]
    w = min(cs)
    r = {"min": w[0], "mean": float(np.mean(v)), "worst_text": w[1], "worst_token": w[2] + 1,
         "tokens": len(v), "same_token_counts": counts == len(x)}
    if cls:
        r["cls_row_min"] = min(cls)
    return r


def census(path):
    import gguf
    rd = gguf.GGUFReader(str(path))
    byt, cnt, el = {}, {}, {}
    for t in rd.tensors:
        if len([d for d in t.shape if d > 1]) < 2:
            continue
        n = t.tensor_type.name.lower()
        byt[n] = byt.get(n, 0) + int(t.n_bytes)
        cnt[n] = cnt.get(n, 0) + 1
        el[n] = el.get(n, 0) + int(t.n_elements)
    # Dominant type by element count over 2-D weights, as the CUDA audit reads precision.
    dom = max(el, key=el.get) if el else None
    return {"bytes_by_type": byt, "count_by_type": cnt, "elements_by_type": el, "dominant": dom}


def tensor_groups(path):
    """Quantisable 2-D weight classes (suffix after the layer index) and layer prefixes."""
    import gguf
    rd = gguf.GGUFReader(str(path))
    classes, layers, other = {}, {}, []
    for t in rd.tensors:
        if len([d for d in t.shape if d > 1]) < 2 or not t.name.endswith("weight") or "norm" in t.name:
            continue
        m = re.match(r"^(.*?)\.(\d+)\.(.+)$", t.name)
        if m:
            classes.setdefault(m.group(3), 0)
            classes[m.group(3)] += 1
            layers.setdefault(int(m.group(2)), f"{m.group(1)}.{m.group(2)}.*")
        else:
            other.append(t.name)
    return sorted(classes), [layers[k] for k in sorted(layers)], other


class Evaluator:
    def __init__(self, model, cfg, cli, quant, P, ref):
        self.model, self.cfg, self.cli, self.quant, self.P, self.ref = model, cfg, cli, quant, P, ref
        self.f32 = None
        self.rows = []

    def run(self, name, path, recipe=None, keep=False, env=None):
        row = {"arm": name, "file": Path(path).name, "recipe": recipe, "bytes": os.path.getsize(path),
               "sha256": sha256(path), "census": census(path), "vs_original": {}, "vs_f32": {}, "wall_s": {}}
        outs = {}
        try:
            for mode in self.cfg["modes"]:
                v, dt = native(self.cli, path, self.P, mode, self.cfg["query_prefix"],
                               env=dict(os.environ, **env) if env else None)
                outs[mode] = v
                row["wall_s"][mode] = round(dt, 2)
                row["vs_original"][mode] = compare(mode, v, self.ref[mode], drop_cls_x=True)
                if self.f32 is not None:
                    row["vs_f32"][mode] = compare(mode, v, self.f32[mode])
        except Exception as e:  # noqa: BLE001 — a failing arm is a recorded result, not a crash
            row["error"] = f"{type(e).__name__}: {e}"[:2000]
        prec = row["census"]["dominant"]
        row["bar"] = BARS.get(prec)
        row["score"] = self.score(row)
        row["qualifies"] = bool(row["bar"] is not None and row["score"] is not None and row["score"] > row["bar"])
        self.rows.append(row)
        log(f"  {name:48s} {row['bytes']/2**20:7.1f} MB dom={prec} score={row['score']} "
            f"{json.dumps({m: round(r['min'], 5) for m, r in row['vs_original'].items()})} {row.get('error', '')}")
        self.flush()
        if keep:
            return row, outs
        return row

    @staticmethod
    def score(row):
        if row.get("error"):
            return None
        vals = [r["min"] for r in row["vs_original"].values()] + [r["min"] for r in row["vs_f32"].values()]
        return min(vals) if vals else None

    def flush(self):
        (OUT / f"arms-{self.model}.json").write_text(json.dumps(self.rows, indent=1))


def quantize(quant, src, dst, qtype, imatrix=None, overrides=()):
    cmd = [quant, src, dst, qtype] + (["--imatrix", imatrix] if imatrix else [])
    for o in overrides:
        cmd += ["--tensor-type", o]
    r, dt = sh(cmd)
    if r.returncode:
        raise RuntimeError(f"quantize {Path(dst).name}: {r.stderr[-1500:]}{r.stdout[-1500:]}")
    tail = [l for l in r.stdout.splitlines() if "quantized," in l or "with imatrix" in l]
    return {"seconds": round(dt, 1), "summary": tail[-1].strip() if tail else ""}


def calibrate(cli, src, C, cfg, path):
    if os.path.exists(path):
        os.remove(path)
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(path))
    for mode in cfg["modes"]:  # BGE-M3: the head matmuls only run in their own mode
        native(cli, src, C, mode, cfg["query_prefix"], env=env)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        sys.exit("calibration produced no imatrix")
    return {"bytes": os.path.getsize(path), "sha256": sha256(path), "n_docs": len(C["docs"]),
            "n_queries": len(C["queries"])}


def prepare_source(model, cfg, cli, src_path_hf):
    if cfg["src"]:
        return Path(fetch(cfg["repo"], cfg["src"]))
    out = S / "bge-m3-heads-f32.gguf"
    if not out.exists():
        r, dt = sh([sys.executable, "models/convert-bert-to-gguf.py", "--model", src_path_hf, "--output", out,
                    "--dtype", "f32", "--crisp"])
        (OUT / "convert-f32.log").write_text(r.stdout[-8000:] + r.stderr[-4000:])
        if r.returncode:
            sys.exit("bge-m3 --crisp conversion failed")
    return out


# ───────────────────────────── sweep ─────────────────────────────
def sweep(model, cfg, cli, quant, P, C, ref, hfpath):
    ev = Evaluator(model, cfg, cli, quant, P, ref)
    src = prepare_source(model, cfg, cli, hfpath)
    _, ev.f32 = ev.run("source-f32", src, keep=True)
    if model == "clip":  # A/B of the EOS-preserving truncation fix against the old path
        ev.run("source-f32 CRISPEMBED_CLIP_TEXT_KEEP_EOS=0 (old truncation)", src,
               env={"CRISPEMBED_CLIP_TEXT_KEEP_EOS": "0"})
    work = S / "arms"
    work.mkdir(exist_ok=True)
    imat = S / f"{model}.imatrix"
    meta = {"calibration": calibrate(cli, src, C, cfg, imat)}

    def arm(name, qtype, im=False, ov=(), src_=None):
        dst = work / (re.sub(r"[^A-Za-z0-9_.+-]+", "_", name) + ".gguf")
        try:
            q = quantize(quant, src_ or src, dst, qtype, imat if im else None, ov)
        except RuntimeError as e:
            ev.rows.append({"arm": name, "error": str(e)[:2000]})
            log(f"  {name}: {e}")
            return None
        row = ev.run(name, dst, {"qtype": qtype, "imatrix": im, "overrides": list(ov), "quantize": q})
        return row, dst

    def drop(x):
        if x:
            Path(x[1]).unlink(missing_ok=True)

    # 1. published files (re-measured on this panel) and uniform types.
    for f in cfg["published"]:
        p = fetch(cfg["repo"], f)
        ev.run(f"published:{f}", p)
        os.unlink(os.path.realpath(p))
    if model == "bgem3":
        conv16 = S / "bge-m3-heads-f16-conv.gguf"
        r, _ = sh([sys.executable, "models/convert-bert-to-gguf.py", "--model", hfpath, "--output", conv16,
                   "--dtype", "f16", "--crisp"])
        if r.returncode == 0:
            ev.run("convert-f16 (heads F32)", conv16, {"converter": "--dtype f16 --crisp"})
        for nm, ov in (("q8_0", ()), ("q8_0 colbert_linear=f16", ("colbert_linear.weight=f16",)),
                       ("q8_0 colbert_linear=f32", ("colbert_linear.weight=f32",))):
            drop(arm(nm, "q8_0", False, ov))
    for qt in ("q8_0", "q6_k", "q5_k") if model != "bgem3" else ("q6_k", "q5_k"):
        drop(arm(qt, qt))
    base_plain = arm("q4_k", "q4_k")
    base_im = arm("q4_k+imatrix", "q4_k", True)
    # 2. diagnosis: same Q4_K weights, F16 matmuls (no activation quantisation).
    for b in (base_plain, base_im):
        if b:
            drop(arm(f"{b[0]['arm']} -> f16 (dequantised Q4_K weights)", "f16", src_=b[1]))
    drop(arm("iq4_xs+imatrix", "iq4_xs", True))
    rows = [r for r in (base_plain, base_im) if r and r[0].get("score") is not None]
    if not rows:
        sys.exit("no Q4_K base arm")
    best = max(rows, key=lambda r: r[0]["score"])
    use_im = bool(best[0]["recipe"]["imatrix"])
    base_score = best[0]["score"]
    meta["base"] = best[0]["arm"]
    drop(base_plain)
    drop(base_im)
    classes, layers, other = tensor_groups(src)
    meta["groups"] = {"classes": classes, "layers": layers, "other": other}
    # 3. llama.cpp-style Q4_K_M analogue (attention V + FFN down at Q6_K).
    km = [f"*.{c}=q6_k" for c in classes if re.search(r"(attn[._]v|ffn[._]down|fc2)", c)]
    drop(arm("q4_k_m-like", "q4_k", use_im, km))
    # 4. sensitivity: one group at Q8_0, everything else at the base recipe.
    groups = [(f"class:{c}", f"*.{c}=q8_0") for c in classes]
    groups += [(f"tensor:{o}", f"{o}=q8_0") for o in other if "embd" not in o and "embed" not in o]
    groups += [(f"layer:{l}", f"{l}=q8_0") for l in layers]
    gains = []
    for g, ov in groups:
        x = arm(f"q4_k{'+im' if use_im else ''} {g}->q8_0", "q4_k", use_im, (ov,))
        if x and x[0].get("score") is not None:
            gains.append((x[0]["score"] - base_score, x[0]["bytes"], g, ov))
        drop(x)
    gains.sort(key=lambda t: -t[0])
    meta["gains"] = [{"group": g, "gain": d, "bytes": b} for d, b, g, _ in gains]
    # 5. greedy: add groups by gain until every metric clears bar + MARGIN.
    chosen, best_row = [], None
    for d, _, g, ov in gains[:14]:
        if d <= 0:
            break
        chosen.append(ov)
        x = arm(f"greedy-{len(chosen)}", "q4_k", use_im, tuple(chosen))
        if not x:
            continue
        row = x[0]
        bar = BARS.get(row["census"]["dominant"], 0.94)
        if row["score"] is not None and row["score"] > bar + MARGIN:
            best_row = row
            meta["greedy_cleared_at"] = len(chosen)
            drop(x)
            break
        drop(x)
    # 6. a coarse fallback that keeps the whole attention + FFN-down path at Q8_0.
    if best_row is None:
        fb = [f"*.{c}=q8_0" for c in classes if re.search(r"(attn|ffn[._]down|fc2)", c)]
        drop(arm("q4_k attn+ffn_down->q8_0", "q4_k", use_im, fb))
    meta["recipe_candidate"] = best_row and best_row["recipe"]
    (OUT / f"sweep-{model}.json").write_text(json.dumps(meta, indent=1))
    return ev, meta


# ───────────────────────────── publish ─────────────────────────────
def publish(model, cfg, cli, quant, P, C, ref, hfpath):
    sys.path.insert(0, str(ROOT / "models"))
    import upload_to_hf
    recipes = json.loads((ROOT / "tests/results/below-bar/recipes.json").read_text())[model]
    ev = Evaluator(model, cfg, cli, quant, P, ref)
    src = prepare_source(model, cfg, cli, hfpath)
    _, ev.f32 = ev.run("source-f32", src, keep=True)
    imat = S / f"{model}.imatrix"
    cal = calibrate(cli, src, C, cfg, imat)
    staged, report = [], {"model": model, "source_commit": os.environ.get("GITHUB_SHA"), "calibration": cal,
                          "artifacts": []}
    for rc in recipes["files"]:
        dst = S / rc["file"]
        if rc.get("converter"):
            r, _ = sh([sys.executable, "models/convert-bert-to-gguf.py", "--model", hfpath, "--output", dst]
                      + rc["converter"].split())
            if r.returncode:
                sys.exit(f"convert {rc['file']} failed")
        else:
            quantize(quant, src, dst, rc["qtype"], imat if rc.get("imatrix") else None, rc.get("overrides", ()))
        row = ev.run(f"candidate:{rc['file']}", dst, rc)
        if not row["qualifies"]:
            report["refused"] = row["arm"]
            (OUT / f"publication-{model}.json").write_text(json.dumps(report, indent=1))
            sys.exit(f"{rc['file']} does not clear its bar (score {row['score']}, bar {row['bar']}): not uploading")
        staged.append((rc, dst, row))
    extra = [(f"{cfg['stem']}-{recipes['imatrix_name']}", imat)] if recipes.get("imatrix_name") else []
    note = recipes["card_note"]
    token = os.environ.get("HF_TOKEN_2") or os.environ.get("HF_TOKEN")
    commit = upload_to_hf.publish_revision(upload_to_hf_name(model), [(rc["file"], d) for rc, d, _ in staged] + extra,
                                           note, token=token)
    from huggingface_hub import hf_hub_download
    for rc, dst, row in staged:
        fresh = hf_hub_download(cfg["repo"], rc["file"], revision=commit, cache_dir=str(S / "fresh"),
                                force_download=True)
        got = sha256(fresh)
        assert got == row["sha256"], f"readback of {rc['file']} differs"
        back = ev.run(f"readback:{rc['file']}", fresh, rc)
        report["artifacts"].append({"file": rc["file"], "url": f"https://huggingface.co/{cfg['repo']}/resolve/{commit}/{rc['file']}",
                                    "sha256": got, "bytes": row["bytes"], "census": row["census"],
                                    "vs_original": back["vs_original"], "vs_f32": back["vs_f32"],
                                    "score": back["score"], "bar": back["bar"], "qualifies": back["qualifies"]})
    report.update(repo=cfg["repo"], commit=commit, status="published and re-verified from an empty cache")
    (OUT / f"publication-{model}.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    return ev


def upload_to_hf_name(model):
    return {"f2llm": "f2llm-v2-0.6b", "clip": "clip-text-base", "bgem3": "bge-m3"}[model]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sweep", "publish"])
    ap.add_argument("model", choices=sorted(CFG))
    a = ap.parse_args()
    cfg = CFG[a.model]
    cli, quant = build()
    P, C = panel(cfg)
    ref, hfpath = reference(a.model, cfg, P)
    (OUT / f"reference-{a.model}.json").write_text(json.dumps(
        {k: v for k, v in ref.items() if k not in ("embed", "sparse", "colbert")}
        | {"n_docs": len(P["docs"]), "n_queries": len(P["queries"]), "n_audit_texts": P["n_audit"]}, indent=1))
    if a.phase == "sweep":
        ev, meta = sweep(a.model, cfg, cli, quant, P, C, ref, hfpath)
    else:
        ev = publish(a.model, cfg, cli, quant, P, C, ref, hfpath)
    lines = [f"### below-bar {a.phase} {a.model} ({os.environ.get('GITHUB_SHA', '')[:8]})", "",
             "| arm | MB | dominant | bar | score | " + " | ".join(f"{m} orig min/mean" for m in cfg["modes"]) + " |",
             "|---|---:|---|---:|---:|" + "---|" * len(cfg["modes"])]
    for r in ev.rows:
        if "bytes" not in r:
            lines.append(f"| {r['arm']} | | | | error | |")
            continue
        cells = [f"{r['vs_original'][m]['min']:.5f}/{r['vs_original'][m]['mean']:.5f}" if m in r["vs_original"]
                 else "-" for m in cfg["modes"]]
        sc = f"{r['score']:.5f}" if r["score"] is not None else "error"
        lines.append(f"| {r['arm']} | {r['bytes']/2**20:.1f} | {r['census']['dominant']} | {r['bar']} | {sc} | "
                     + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
