#!/usr/bin/env python3
"""CrispEmbed CUDA-vs-CPU engine audit on a Kaggle GPU (feat/cuda-audit).

    python tools/kaggle/cuda-audit/run.py BUILD_DIR BATCH      (normally via kernel.py)
    python tools/kaggle/cuda-audit/run.py --list BATCH          (print the plan, run nothing)

Counterpart of tests/results/gpu-audit/metal-m1/ab_embed.py + ab_cmd.py. Every
arm is a separate process of the SAME binary:

  cuda arms : CLI args + `--gpu-backend cuda` (+ arm env)
  cpu arm   : CLI args + `--gpu-backend cpu`, CUDA_VISIBLE_DEVICES="" (no GPU
              visible at all, so per-kind defaults resolve exactly as on a CPU box)

Diff binaries (test-*-diff) take no --gpu-backend flag: their CUDA arm is the
engine default with the GPU visible, their CPU arm hides it.

Each arm records whether it really touched the GPU (peak nvidia-smi memory.used
above the idle baseline, plus any CUDA0 marker on stderr), so a silently-CPU
"CUDA" arm is visible in the receipt rather than passing as parity. Outputs go
to files, every step has a timeout, and report.json is rewritten after every
job. Models are fetched from the HF URLs below and deleted after their last use.

Acceptance (docs/precision-acceptance.md): final-output cosine against the CPU
reference above F32 0.99, F16 0.98, Q8_0 0.97, Q4_K/IQ4 0.94. The precision of
an artifact is read from its GGUF (dominant 2-D weight type), not its name.
"""
import difflib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_VERSION = "cuda-audit-run-v1"
OUT = Path(os.environ.get("CUDA_AUDIT_OUT", "/kaggle/working/cuda-audit"))
SCRATCH = Path(os.environ.get("CUDA_AUDIT_SCRATCH", "/kaggle/temp/cuda-audit"))
IMG = "tests/regression/images"
TEXTS = "tests/results/gpu-audit/metal-m1/embed_texts.txt"
QUERY = "What did Einstein publish in 1905?"
BARS = {"f32": 0.99, "f16": 0.98, "bf16": 0.98, "q8_0": 0.97, "q6_k": 0.97, "q5_k": 0.94, "q4_k": 0.94,
        "q4_0": 0.94, "iq4_xs": 0.94}
LICENCE_ENV = {  # owner-approved acknowledgements for internal testing only; never defaults
    "CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE": "1", "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE": "1",
    "CRISPEMBED_ACCEPT_LFM_LICENSE": "1", "CRISPEMBED_ACCEPT_LICENSE": "all"}


def hf(repo, name, rev="main"):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{name}"


# ───────────────────────────── model table ─────────────────────────────
M = {}


def add(key, repo, name, rev="main"):
    M[key] = {"url": hf(repo, name, rev), "file": name}


EMBED_FAMILIES = [  # (family, repo, base file, [quant files], modes)
    ("minilm-l6", "cstr/all-MiniLM-L6-v2-GGUF", "all-MiniLM-L6-v2-f32.gguf",
     ["all-MiniLM-L6-v2-q8_0.gguf", "all-MiniLM-L6-v2-q4_k.gguf"], ["embed"]),
    ("bge-small", "cstr/bge-small-en-v1.5-GGUF", "bge-small-en-v1.5.gguf",
     ["bge-small-en-v1.5-q8_0.gguf", "bge-small-en-v1.5-q4_k.gguf"], ["embed"]),
    ("e5-small", "cstr/multilingual-e5-small-GGUF", "multilingual-e5-small.gguf",
     ["multilingual-e5-small-q8_0.gguf", "multilingual-e5-small-q4_k.gguf"], ["embed"]),
    ("arctic-m-v2", "cstr/arctic-embed-m-v2-GGUF", "arctic-embed-m-v2.gguf",
     ["arctic-embed-m-v2-q8_0.gguf", "arctic-embed-m-v2-q4_k-imatrix.gguf"], ["embed"]),
    ("bge-m3", "cstr/bge-m3-GGUF", "bge-m3.gguf",
     ["bge-m3-q8_0.gguf", "bge-m3-iq4_xs.gguf"], ["embed", "sparse", "colbert"]),
    ("qwen3-embed-0.6b", "cstr/qwen3-embed-0.6b-GGUF", "qwen3-embed-0.6b.gguf",
     ["qwen3-embed-0.6b-q8_0.gguf", "qwen3-embed-0.6b-q4_k-imatrix.gguf"], ["embed"]),
    ("f2llm-v2-0.6b", "cstr/f2llm-v2-0.6b-GGUF", "f2llm-v2-0.6b.gguf",
     ["f2llm-v2-0.6b-q8_0.gguf", "f2llm-v2-0.6b-q4_k-imatrix.gguf"], ["embed"]),
    ("jina-v5-nano", "cstr/jina-v5-nano-GGUF", "jina-v5-nano.gguf",
     ["jina-v5-nano-q8_0.gguf", "jina-v5-nano-q4_k.gguf"], ["embed"]),
    ("harrier-270m", "cstr/harrier-270m-GGUF", "harrier-270m.gguf",
     ["harrier-270m-q8_0.gguf", "harrier-270m-q4_k.gguf"], ["embed"]),
    ("embeddinggemma-300m", "cstr/embeddinggemma-300m-GGUF", "embeddinggemma-300m.gguf",
     ["embeddinggemma-300m-q8_0.gguf", "embeddinggemma-300m-q4_k.gguf"], ["embed"]),
    ("lfm2-embed", "cstr/lfm2-embed-GGUF", "lfm2-embed-f16.gguf",
     ["lfm2-embed-q8_0.gguf", "lfm2-embed-q4_k.gguf"], ["embed"]),
    ("clip-text-base", "cstr/clip-text-base-GGUF", "clip-text-base.gguf",
     ["clip-text-base-q8_0.gguf", "clip-text-base-q4_k.gguf"], ["embed"]),
    ("ms-marco-l6", "cstr/ms-marco-MiniLM-L-6-v2-GGUF", "ms-marco-MiniLM-L-6-v2-g7c.gguf",
     ["ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf", "ms-marco-MiniLM-L-6-v2-q4_k-g7c.gguf"], ["rerank"]),
    ("mxbai-rerank-xsmall", "cstr/mxbai-rerank-xsmall-v1-GGUF", "mxbai-rerank-xsmall-v1-g7c.gguf",
     ["mxbai-rerank-xsmall-v1-q8_0-g7c.gguf"], ["rerank"]),
    ("ettin-reranker-17m", "cstr/ettin-reranker-17m-v1-GGUF", "ettin-reranker-17m-v1-f32.gguf",
     ["ettin-reranker-17m-v1-q8_0.gguf", "ettin-reranker-17m-v1-q4_k.gguf"], ["rerank"]),
]
for fam, repo, base, quants, _ in EMBED_FAMILIES:
    for f in [base] + quants:
        add(f, repo, f)
EMBED_FAMILIES.append(("bge-m3-heads", None, "bge-m3-heads-f16.gguf", ["bge-m3-heads-q8_0.gguf"],
                       ["embed", "sparse", "colbert"]))

for key, repo, name in [
    ("dbnet", "cstr/dbnet-ic15-GGUF", "dbnet-ic15-q8_0.gguf"),
    ("trocr", "cstr/trocr-small-printed-GGUF", "trocr-small-printed-q8_0.gguf"),
    ("trocr-f32", "cstr/trocr-small-printed-GGUF", "trocr-small-printed-f32.gguf"),
    ("ppv6-det", "cstr/PP-OCRv6-small-det-GGUF", "PP-OCRv6_small_det-f16.gguf"),
    ("ppv6-rec", "cstr/PP-OCRv6_small_rec-GGUF", "PP-OCRv6_small_rec-q8-head.gguf"),
    ("ppv6-rec-ref", "cstr/PP-OCRv6_small_rec-GGUF", "PP-OCRv6_small_rec-fox-ref.gguf"),
    ("parseq", "cstr/parseq-GGUF", "parseq-q8_0.gguf"),
    ("parseq-tiny", "cstr/parseq-GGUF", "parseq-tiny-q8_0.gguf"),
    ("esrgan", "cstr/esrgan-sr-GGUF", "esrgan-x4-f32.gguf"),
    ("esrgan-ref", "cstr/esrgan-sr-GGUF", "esrgan-ref.gguf"),
    ("instructir", "cstr/instructir-GGUF", "instructir-f16.gguf"),
    ("instructir-f32", "cstr/InstructIR-GGUF", "instructir-f32.gguf"),
    ("instructir-ref", "cstr/InstructIR-GGUF", "instructir-ref.gguf"),
    ("qwen3-embed-ref", "cstr/qwen3-embed-0.6b-GGUF", "qwen3-embed-ref.gguf"),
    ("lfm2-ref", "cstr/lfm2-embed-GGUF", "lfm2-ref.gguf"),
    ("clip-text-ref", "cstr/clip-text-base-GGUF", "clip-text-ref.gguf"),
    # batch b
    ("got-q4", "cstr/got-ocr2-crispembed-GGUF", "got-ocr2-q4_k.gguf"),
    ("got-f16", "cstr/got-ocr2-crispembed-GGUF", "got-ocr2-f16.gguf"),
    ("got-ref", "cstr/got-ocr2-crispembed-GGUF", "got-ocr2-ref-full.gguf"),
    ("h2ovl-q8", "cstr/h2ovl-800m-crispembed-GGUF", "h2ovl-800m-q8_0.gguf"),
    ("granite-q4", "cstr/granite-vision-crispembed-GGUF", "granite-vision-3.3-2b-q4_k.gguf"),
    ("granite-q8", "cstr/granite-vision-crispembed-GGUF", "granite-vision-3.3-2b-q8_0.gguf"),
    ("granite-ref", "cstr/granite-vision-crispembed-GGUF", "granite-vision-ref.gguf"),
    ("lighton-q4", "cstr/lightonocr-GGUF", "lightonocr-1b-q4_k.gguf"),
    ("glm-q8", "cstr/glm-ocr-crispembed-GGUF", "glm-ocr-q8_0.gguf"),
    ("glm-f16", "cstr/glm-ocr-crispembed-GGUF", "glm-ocr-f16.gguf"),
    ("glm-ref", "cstr/glm-ocr-crispembed-GGUF", "glm-ocr-ref-full.gguf"),
    ("smoldocling-q8", "cstr/smoldocling-GGUF", "smoldocling-q8_0.gguf"),
    ("smoldocling-f16", "cstr/smoldocling-GGUF", "smoldocling-f16.gguf"),
    ("surya-det", "cstr/surya-det-GGUF", "surya-det-f16.gguf"),
    ("sam2-f32", "cstr/sam2.1-hiera-tiny-GGUF", "sam2.1-hiera-tiny-f32.gguf"),
    ("sam2-f16", "cstr/sam2.1-hiera-tiny-GGUF", "sam2.1-hiera-tiny-f16.gguf"),
    ("gliner-q8", "cstr/sauerkraut-gliner-lfm-GGUF", "gliner-lfm-q8_0.gguf"),
    ("gliner-deberta-f32", "cstr/gliner-deberta-GGUF", "gliner-deberta-f32.gguf"),
    ("gliner-deberta-q8", "cstr/gliner-deberta-GGUF", "gliner-deberta-q8_0.gguf"),
    ("lilt-f32", "cstr/lilt-base-GGUF", "lilt-base-f32.gguf"),
    ("lilt-ref", "cstr/lilt-base-GGUF", "lilt-ref.gguf"),
    ("firered-f32", "cstr/fireredpunc-GGUF", "fireredpunc.gguf"),
    ("firered-q8", "cstr/fireredpunc-GGUF", "fireredpunc-q8_0.gguf"),
    ("pcs-q8", "cstr/pcs-xlmr-base-GGUF", "pcs-xlmr-base-q8_0.gguf"),
    ("siglip-base", "cstr/siglip-base-GGUF", "siglip-base.gguf"),
    ("vit-ref", "cstr/siglip-base-GGUF", "vit-ref.gguf"),
    ("clip-vit-b16", "cstr/clip-vit-base-patch16-GGUF", "clip-vit-base-patch16.gguf"),
    ("scrfd", "cstr/scrfd-det-10g-GGUF", "scrfd-det-10g.gguf"),
    ("face-ref", "cstr/scrfd-det-10g-GGUF", "face-ref.gguf"),
    ("bidirlm-q8", "cstr/bidirlm-omni-2.5b-GGUF", "bidirlm-omni-2.5b-q8_0.gguf"),
    ("bidirlm-text-ref", "cstr/bidirlm-omni-2.5b-GGUF", "bidirlm-text-ref.gguf"),
    ("bidirlm-vision-ref", "cstr/bidirlm-omni-2.5b-GGUF", "bidirlm-vision-ref.gguf"),
    ("nafnet-f16", "cstr/nafnet-sidd-GGUF", "nafnet-sidd-w32-f16.gguf"),
    ("nafnet-q8", "cstr/nafnet-sidd-GGUF", "nafnet-sidd-w32-q8_0.gguf"),
    ("nafnet-ref", "cstr/nafnet-sidd-GGUF", "nafnet-ref.gguf"),
    ("smt", "cstr/smt-grandstaff-GGUF", "smt-grandstaff-q8_0.gguf"),
    ("smt-f32", "cstr/smt-grandstaff-GGUF", "smt-grandstaff-f32.gguf"),
    ("tromr", "cstr/tromr-GGUF", "tromr-q8_0.gguf"),
    ("tromr-f32", "cstr/tromr-GGUF", "tromr-f32.gguf"),
    ("transcoda", "cstr/transcoda-omr-GGUF", "transcoda-q8_0.gguf"),
    ("ppfnl-q8", "cstr/ppformulanet-l-gguf", "ppformulanet-l-q8_0.gguf"),
    ("ppfnl-f16", "cstr/ppformulanet-l-gguf", "ppformulanet-l-f16.gguf"),
    ("pan-f16", "cstr/text-super-resolution-gguf", "pan-x4-f16.gguf"),
    ("pan-ref", "cstr/text-super-resolution-gguf", "pan-ref.gguf"),
    ("restormer-f16", "cstr/text-super-resolution-gguf", "restormer-denoise-f16.gguf"),
    ("restormer-ref", "cstr/text-super-resolution-gguf", "restormer-ref.gguf"),
    ("swinir-f16", "cstr/text-super-resolution-gguf", "swinir-light-x4-f16.gguf"),
    ("swinir-ref", "cstr/text-super-resolution-gguf", "swinir-ref.gguf"),
    ("hat-f16", "cstr/text-super-resolution-gguf", "hat-sr-x4-f16.gguf"),
    ("hat-ref", "cstr/text-super-resolution-gguf", "hat-ref.gguf"),
    ("tbsrn-f16", "cstr/text-super-resolution-gguf", "tbsrn-telescope-f16.gguf"),
    ("tbsrn-ref", "cstr/text-super-resolution-gguf", "tbsrn-ref.gguf"),
    ("dat-f32", "cstr/text-super-resolution-gguf", "dat-light-x2-f32.gguf"),
    ("dat-ref", "cstr/text-super-resolution-gguf", "dat-ref.gguf"),
    ("scunet-f32", "cstr/scunet-GGUF", "scunet-color-f32.gguf"),
    ("scunet-ref", "cstr/scunet-GGUF", "scunet-ref.gguf"),
    ("adair-f32", "cstr/AdaIR-GGUF", "adair-5d-f32.gguf"),
    ("adair-ref", "cstr/text-super-resolution-gguf", "adair-ref.gguf"),
    ("modernbert-q8", "cstr/gte-modernbert-base-GGUF", "gte-modernbert-base-q8_0.gguf"),
    ("modernbert-ref", "cstr/gte-modernbert-base-GGUF", "modernbert-ref.gguf"),
]:
    add(key, repo, name)
# EasyOCR's english_g2 recognizer is not on HF as GGUF: converted on the worker
# from the upstream release checkpoint (see fetch_easyocr()).
M["easyocr"] = {"url": None, "file": "easyocr-english-g2-f16.gguf", "convert": "easyocr"}
# The published bge-m3 GGUFs carry no sparse/ColBERT heads ("model does not support sparse
# retrieval", run 1): convert BAAI/bge-m3 with its sparse_linear/colbert_linear .pt heads.
M["bge-m3-heads-f16.gguf"] = {"url": None, "file": "bge-m3-heads-f16.gguf", "convert": "bert",
                              "hf_model": "BAAI/bge-m3", "dtype": "f16"}
M["bge-m3-heads-q8_0.gguf"] = {"url": None, "file": "bge-m3-heads-q8_0.gguf", "convert": "quantize",
                               "src": "bge-m3-heads-f16.gguf", "qtype": "q8_0"}


# ───────────────────────────── job plan ─────────────────────────────
CUDA = ("cuda", "cuda", {})
CPU = ("cpu", "cpu", {})


def cmd(jid, argv, arms=(CUDA, CPU), repeat=2, timeout=1800, ref=None, note=""):
    return {"id": jid, "kind": "cmd", "argv": list(argv), "arms": [list(a) for a in arms], "repeat": repeat,
            "timeout": timeout, "ref": ref, "note": note}


def diff(jid, binary, argv, env=None, thresholds=None, timeout=1800):
    return {"id": jid, "kind": "diff", "binary": binary, "argv": list(argv), "env": env or {},
            "thresholds": thresholds or {}, "timeout": timeout}


def ocr_json(m_rec, det, img, extra=()):
    return ["-t", "@NT", "--json", "-m", f"@M:{m_rec}", "--ocr-det", f"@M:{det}", "--ocr-rec", f"@M:{m_rec}",
            "--ocr", f"{IMG}/{img}.png", *extra]


def batch_a():
    jobs = []
    for fam, repo, base, quants, modes in EMBED_FAMILIES:
        for mode in modes:
            jobs.append({"id": f"embed-{fam}-{mode}", "kind": "embed", "mode": mode, "base": base,
                         "files": [base] + quants, "repeat": 2, "timeout": 2400})
    # Default OCR pipeline: DBNet (per-kind: CUDA on a CUDA box) + TrOCR (GPU).
    det_arms = (CUDA, ("cuda-detcpu", "cuda", {"OCR_DETECT_USE_GPU": "0"}),
                ("cuda-deconvmm", "cuda", {"CRISPEMBED_DBNET_DECONV_MATMUL": "1"}), CPU)
    for img in ("fox", "scan_page_pd", "scan_strip"):
        jobs.append(cmd(f"dbnet-trocr.{img}", ocr_json("trocr", "dbnet", img), arms=det_arms))
    jobs.append(cmd("dbnet-trocr-f32.fox", ocr_json("trocr-f32", "dbnet", "fox")))
    for img in ("fox", "scan_page_pd"):
        jobs.append(cmd(f"ppocrv6-small.{img}", ocr_json("ppv6-rec", "ppv6-det", img),
                        arms=(CUDA, ("cuda-detcpu", "cuda", {"CRISPEMBED_PPOCRV6_DET_GPU": "0"}), CPU)))
    jobs.append(cmd("dbnet-easyocr.fox", ["-t", "@NT", "--json", "-m", "@M:easyocr", "--ocr-pipeline",
                                          f"{IMG}/fox.png", "--ocr-engine", "easyocr", "--ocr-det", "@M:dbnet",
                                          "--ocr-rec", "@M:easyocr"]))
    jobs.append(cmd("parseq.fox", ["-t", "@NT", "-m", "@M:parseq", "--ocr", f"{IMG}/fox.png"]))
    jobs.append(cmd("parseq-tiny.fox", ["-t", "@NT", "-m", "@M:parseq-tiny", "--ocr", f"{IMG}/fox.png"]))
    gpu_sr = lambda gate: (("cuda-gate1", "cuda", {gate: "1"}), CUDA, CPU)  # noqa: E731
    for img in ("fox", "scan_strip"):
        jobs.append(cmd(f"esrgan.{img}", ["-t", "@NT", "--esrgan-model", "@M:esrgan", "--esrgan-sr",
                                         f"{IMG}/{img}.png"], arms=gpu_sr("CRISPEMBED_ESRGAN_GPU")))
    jobs.append(cmd("instructir.scan_strip", ["-t", "@NT", "--instructir-model", "@M:instructir", "--instructir",
                                             f"{IMG}/scan_strip.png"], arms=gpu_sr("CRISPEMBED_INSTRUCTIR_GPU")))
    # Stage diffs against the committed Python references (worst stage cosine).
    jobs.append(diff("diff-ppocrv6-small-rec", "test-ppocrv6-rec", ["@M:ppv6-rec", f"{IMG}/fox.png"],
                     env={"PPOCRV6_REF": "@M:ppv6-rec-ref"}, thresholds={"*": 0.999}))
    jobs.append(diff("diff-qwen3-embed-q8", "test-decoder-embed-diff",
                     ["@M:qwen3-embed-0.6b-q8_0.gguf", "@M:qwen3-embed-ref"], thresholds={"final_embedding": 0.98}))
    jobs.append(diff("diff-lfm2-embed-q8", "test-lfm2-diff", ["@M:lfm2-embed-q8_0.gguf", "@M:lfm2-ref"],
                     thresholds={"*": 0.99}))
    jobs.append(diff("diff-clip-text", "test-clip-text-diff", ["@M:clip-text-base.gguf", "@M:clip-text-ref"],
                     thresholds={"final_embedding": 0.99}))
    jobs.append(diff("diff-esrgan", "test-esrgan-diff", ["@M:esrgan", "@M:esrgan-ref"], thresholds={"output": 0.95}))
    jobs.append(diff("diff-instructir", "test-instructir-diff", ["@M:instructir-f32", "@M:instructir-ref"],
                     thresholds={"output": 0.99}))
    jobs.append(diff("diff-modernbert-q8", "test-modernbert-diff", ["@M:modernbert-q8", "@M:modernbert-ref"],
                     thresholds={"final_hidden": 0.99}))
    return jobs


def vlm(jid, model, img="fox", extra=(), arms=(CUDA, CPU), timeout=3600, ref=None):
    return cmd(jid, ["-t", "@NT", "-m", f"@M:{model}", "--ocr", f"{IMG}/{img}.png", "--ocr-max-tokens", "128",
                     *extra], arms=arms, repeat=1, timeout=timeout, ref=ref)


def batch_b():
    jobs = [
        vlm("got-ocr2-q4k.fox", "got-q4", ref="got-ocr2-f16.fox"),
        vlm("got-ocr2-f16.fox", "got-f16"),
        diff("diff-got-ocr2-q4k", "test-got-ocr-diff", ["@M:got-q4", "@M:got-ref"], thresholds={"*": 0.95}),
        vlm("h2ovl-800m-q8.fox", "h2ovl-q8"),
        vlm("granite-vision-q4k.fox", "granite-q4",
            arms=(CUDA, ("cpu", "cpu", {"CRISPEMBED_GRANITE_CPU": "1"})), ref="granite-vision-q8.fox"),
        vlm("granite-vision-q8.fox", "granite-q8", arms=(CUDA, ("cpu", "cpu", {"CRISPEMBED_GRANITE_CPU": "1"}))),
        diff("diff-granite-vision-q8", "test-granite-vision-diff", ["@M:granite-q8", "@M:granite-ref"],
             thresholds={"*": 0.95}),
        vlm("lightonocr-1b-q4k.fox", "lighton-q4",
            arms=(("cuda-gate1", "cuda", {"CRISPEMBED_LIGHTONOCR_GPU": "1"}), CUDA, CPU)),
        vlm("glm-ocr-q8.fox", "glm-q8", ref="glm-ocr-f16.fox"),
        vlm("glm-ocr-f16.fox", "glm-f16"),
        vlm("glm-ocr-q8.receipt", "glm-q8", img="cc0/receipt_example"),
        diff("diff-glm-ocr-q8", "test-glm-ocr-diff", ["@M:glm-q8", "@M:glm-ref"], thresholds={"*": 0.99}),
        vlm("smoldocling-q8.fox", "smoldocling-q8", ref="smoldocling-f16.fox"),
        vlm("smoldocling-f16.fox", "smoldocling-f16"),
        cmd("surya-trocr.fox", ocr_json("trocr", "surya-det", "fox")),
        cmd("surya-trocr.scan_page_pd", ocr_json("trocr", "surya-det", "scan_page_pd")),
        cmd("sam2-f32.fox", ["-t", "@NT", "--sam2", f"{IMG}/fox.png", "--sam2-model", "@M:sam2-f32",
                             "--sam2-point", "200,60:1"]),
        cmd("sam2-f16.fox", ["-t", "@NT", "--sam2", f"{IMG}/fox.png", "--sam2-model", "@M:sam2-f16",
                             "--sam2-point", "200,60:1"], ref="sam2-f32.fox"),
        cmd("gliner-lfm-q8.ner", ["-t", "@NT", "-m", "@M:gliner-q8", "--ner",
                                  "Angela Merkel met Tim Cook at the Apple office in Berlin on Monday.",
                                  "--ner-labels", "person,organization,location,date", "--ner-threshold", "0.3"]),
        cmd("gliner-deberta-f32.ner", ["-t", "@NT", "-m", "@M:gliner-deberta-f32", "--ner",
                                       "Angela Merkel met Tim Cook at the Apple office in Berlin on Monday.",
                                       "--ner-labels", "person,organization,location,date", "--ner-threshold", "0.3"]),
        cmd("gliner-deberta-q8.ner", ["-t", "@NT", "-m", "@M:gliner-deberta-q8", "--ner",
                                      "Angela Merkel met Tim Cook at the Apple office in Berlin on Monday.",
                                      "--ner-labels", "person,organization,location,date", "--ner-threshold", "0.3"],
            ref="gliner-deberta-f32.ner"),
        diff("diff-lilt", "test-lilt-diff", ["@M:lilt-f32", "@M:lilt-ref"], thresholds={"*": 0.99}),
        cmd("trocr-fireredpunc-f32.fox", ["-t", "@NT", "-m", "@M:trocr", "--ocr", f"{IMG}/fox.png",
                                          "--punct-model", "@M:firered-f32"]),
        cmd("trocr-fireredpunc-q8.fox", ["-t", "@NT", "-m", "@M:trocr", "--ocr", f"{IMG}/fox.png",
                                         "--punct-model", "@M:firered-q8"]),
        cmd("trocr-pcs-q8.fox", ["-t", "@NT", "-m", "@M:trocr", "--ocr", f"{IMG}/fox.png",
                                 "--punct-model", "@M:pcs-q8"]),
        cmd("siglip-base.image", ["-t", "@NT", "--json", "-m", "@M:siglip-base", "--image", f"{IMG}/fox.png"]),
        diff("diff-vit-siglip", "test-vit-embed-diff", ["@M:siglip-base", "@M:vit-ref", f"{IMG}/fox.png"],
             thresholds={"final_embedding": 0.98}),
        cmd("clip-vit-b16.image", ["-t", "@NT", "--json", "-m", "@M:clip-vit-b16", "--image", f"{IMG}/fox.png"]),
        cmd("scrfd.detect", ["-t", "@NT", "-m", "@M:scrfd", "--detect", f"{IMG}/face.png"]),
        diff("diff-face-scrfd", "test-face-diff", ["@M:scrfd", "@M:face-ref", f"{IMG}/face.png"],
             thresholds={"detection": 0.5}),
        cmd("bidirlm-q8.embed", ["-t", "@NT", "--json", "-m", "@M:bidirlm-q8", "-f", TEXTS], timeout=2400),
        cmd("bidirlm-q8.image", ["-t", "@NT", "--json", "-m", "@M:bidirlm-q8", "--image", f"{IMG}/fox.png"],
            timeout=2400),
        diff("diff-bidirlm-text", "test-decoder-embed-diff", ["@M:bidirlm-q8", "@M:bidirlm-text-ref"],
             thresholds={"final_embedding": 0.98}),
        diff("diff-bidirlm-vision", "test-bidirlm-vision-diff", ["@M:bidirlm-q8", "@M:bidirlm-vision-ref"],
             thresholds={"*": 0.99}),
        cmd("nafnet-f16.scan_strip", ["-t", "@NT", "--nafnet-model", "@M:nafnet-f16", "--nafnet-denoise",
                                      f"{IMG}/scan_strip.png"]),
        cmd("nafnet-q8.scan_strip", ["-t", "@NT", "--nafnet-model", "@M:nafnet-q8", "--nafnet-denoise",
                                     f"{IMG}/scan_strip.png"], ref="nafnet-f16.scan_strip"),
        diff("diff-nafnet-f16", "test-nafnet-diff", ["@M:nafnet-f16", "@M:nafnet-ref"], thresholds={"output": 0.99}),
        cmd("smt-q8.staff", ["-t", "@NT", "-m", "@M:smt", "--ocr", f"{IMG}/staff_smt.png"], ref="smt-f32.staff"),
        cmd("smt-f32.staff", ["-t", "@NT", "-m", "@M:smt-f32", "--ocr", f"{IMG}/staff_smt.png"]),
        cmd("tromr-q8.staff", ["-t", "@NT", "-m", "@M:tromr", "--ocr", f"{IMG}/staff_tromr.jpg"],
            ref="tromr-f32.staff"),
        cmd("tromr-f32.staff", ["-t", "@NT", "-m", "@M:tromr-f32", "--ocr", f"{IMG}/staff_tromr.jpg"]),
        cmd("transcoda-q8.page", ["-t", "@NT", "-m", "@M:transcoda", "--ocr", f"{IMG}/page_transcoda.png"],
            timeout=3600),
        cmd("ppformulanet-l-q8.formula", ["-t", "@NT", "-m", "@M:ppfnl-q8", "--ocr", f"{IMG}/formula_quadratic.png"],
            ref="ppformulanet-l-f16.formula"),
        cmd("ppformulanet-l-f16.formula", ["-t", "@NT", "-m", "@M:ppfnl-f16", "--ocr",
                                           f"{IMG}/formula_quadratic.png"]),
    ]
    # Scope 3: new default-off CUDA paths for CPU-computed restoration graphs.
    jobs += [
        cmd("pan-f16.scan_strip", ["-t", "@NT", "--pan-model", "@M:pan-f16", "--pan-sr", f"{IMG}/scan_strip.png"],
            arms=(("cuda-gate1", "cuda", {"CRISPEMBED_PAN_GPU": "1"}), CUDA, CPU)),
        cmd("restormer-f16.scan_strip", ["-t", "@NT", "--restormer", "@M:restormer-f16", "--restormer-input",
                                         f"{IMG}/scan_strip.png"],
            arms=(("cuda-gate1", "cuda", {"CRISPEMBED_RESTORMER_GPU": "1"}), CUDA, CPU)),
        diff("diff-pan-gpu", "test-pan-diff", ["@M:pan-f16", "@M:pan-ref"], env={"CRISPEMBED_PAN_GPU": "1"},
             thresholds={"output": 0.99}),
        diff("diff-restormer-gpu", "test-restormer-diff", ["@M:restormer-f16", "@M:restormer-ref"],
             env={"CRISPEMBED_RESTORMER_GPU": "1"}, thresholds={"output": 0.99}),
    ]
    # Default (weights-on-GPU, compute-on-CPU) restoration engines against their references.
    for name, binary, w, r in [("swinir", "test-swinir-diff", "swinir-f16", "swinir-ref"),
                               ("hat", "test-hat-diff", "hat-f16", "hat-ref"),
                               ("tbsrn", "test-tbsrn-diff", "tbsrn-f16", "tbsrn-ref"),
                               ("dat", "test-dat-diff", "dat-f32", "dat-ref"),
                               ("scunet", "test-scunet-diff", "scunet-f32", "scunet-ref"),
                               ("adair", "test-adair-diff", "adair-f32", "adair-ref")]:
        jobs.append(diff(f"diff-{name}", binary, [f"@M:{w}", f"@M:{r}"], thresholds={"output": 0.99}))
    return jobs


def batch_rest():
    """Run 2: everything run 1 (8f781770, crashed at the EasyOCR fetch) had not measured,
    plus the clip-text JSON-lines re-parse and the bge-m3 artifact that carries its heads."""
    done1 = {"embed-minilm-l6-embed", "embed-bge-small-embed", "embed-e5-small-embed", "embed-arctic-m-v2-embed",
             "embed-bge-m3-embed", "embed-bge-m3-sparse", "embed-bge-m3-colbert", "embed-qwen3-embed-0.6b-embed",
             "embed-f2llm-v2-0.6b-embed", "embed-jina-v5-nano-embed", "embed-harrier-270m-embed",
             "embed-embeddinggemma-300m-embed", "embed-lfm2-embed-embed", "embed-ms-marco-l6-rerank",
             "embed-mxbai-rerank-xsmall-rerank", "embed-ettin-reranker-17m-rerank", "dbnet-trocr.fox",
             "dbnet-trocr.scan_page_pd", "dbnet-trocr.scan_strip", "dbnet-trocr-f32.fox", "ppocrv6-small.fox",
             "ppocrv6-small.scan_page_pd"}
    return [j for j in batch_a() + batch_b() if j["id"] not in done1]


BATCHES = {"a": batch_a, "b": batch_b, "all": lambda: batch_a() + batch_b(), "rest": batch_rest}


def jobs_for(batch):
    names = batch.split("+")
    out = []
    for n in names:
        out.extend(BATCHES[n]())
    # A job compared against another job's CPU output needs that job to run first.
    for _ in range(len(out) ** 2):
        moved = False
        for i, j in enumerate(out):
            k = next((x for x, jj in enumerate(out) if jj["id"] == j.get("ref")), None)
            if k is not None and k > i:
                out.insert(i, out.pop(k))
                moved = True
                break
        if not moved:
            break
    only = os.environ.get("CUDA_AUDIT_ONLY")
    if only:
        out = [j for j in out if re.search(only, j["id"])]
    return out


def build_targets(batch):
    t = ["crispembed-cli", "crispembed-quantize"]
    for j in jobs_for(batch):
        if j["kind"] == "diff" and j["binary"] not in t:
            t.append(j["binary"])
    return t


# ───────────────────────────── infrastructure ─────────────────────────────
REPORT = {"script_version": SCRIPT_VERSION, "jobs": {}}


def save(phase):
    REPORT["phase"] = phase
    REPORT["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1, ensure_ascii=False) + "\n")
    print(f"== {phase}", flush=True)


def models_in(job):
    keys = []
    if job["kind"] == "embed":
        keys = list(job["files"])
    for s in job.get("argv", []) + list(job.get("env", {}).values()):
        if isinstance(s, str) and s.startswith("@M:"):
            keys.append(s[3:])
    return keys


def sh_capture(command, timeout):
    p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def fetch_easyocr(dest):
    """Convert EasyOCR english_g2 to GGUF (charset read from the easyocr wheel's own config)."""
    work = SCRATCH / "easyocr-src"
    work.mkdir(parents=True, exist_ok=True)
    zp = work / "english_g2.zip"
    rc, _, err = sh_capture(["curl", "-sfL", "--retry", "3", "-o", str(zp),
                             "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"], 1200)
    if rc != 0:
        return f"download failed: {err[-300:]}"
    shutil.unpack_archive(str(zp), str(work))
    rc, _, err = sh_capture([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "easyocr", "-d",
                             str(work / "wheel")], 1200)
    if rc != 0:
        return f"easyocr wheel download failed: {err[-300:]}"
    whl = next((work / "wheel").glob("easyocr-*.whl"))
    shutil.unpack_archive(str(whl), str(work / "wheel" / "x"), format="zip")
    cfg_src = (work / "wheel" / "x" / "easyocr" / "config.py").read_text()
    ns = {"os": os, "__file__": str(work / "wheel" / "x" / "easyocr" / "config.py")}
    exec(compile(cfg_src, "easyocr_config", "exec"), ns)  # plain dict definitions
    chars = ns["recognition_models"]["gen2"]["english_g2"]["characters"]
    (work / "english_g2_chars.txt").write_text(chars, encoding="utf-8")
    rc, out, err = sh_capture([sys.executable, "models/convert-easyocr-to-gguf.py", "--checkpoint",
                               str(next(work.glob("english_g2.pth"))), "--charset",
                               str(work / "english_g2_chars.txt"), "--output", str(dest), "--fp16"], 1200)
    return None if rc == 0 and dest.exists() else f"convert failed rc={rc}: {(out + err)[-600:]}"


QUANTIZE = [None]


def fetch(key):
    spec = M[key]
    dest = SCRATCH / "models" / spec["file"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if spec.get("convert") == "easyocr":
        err = fetch_easyocr(dest)
    elif spec.get("convert") == "bert":
        rc, out, e = sh_capture([sys.executable, "models/convert-bert-to-gguf.py", "--model", spec["hf_model"],
                                 "--output", str(dest), "--dtype", spec.get("dtype", "f16")], 3600)
        err = None if rc == 0 and dest.exists() else f"convert rc={rc}: {(out + e)[-800:]}"
        REPORT.setdefault("conversions", {})[key] = (out + e)[-1500:]
    elif spec.get("convert") == "quantize":
        src, err = fetch(spec["src"])
        if not err:
            rc, out, e = sh_capture([str(QUANTIZE[0]), str(src), str(dest), spec["qtype"]], 3600)
            err = None if rc == 0 and dest.exists() else f"quantize rc={rc}: {(out + e)[-800:]}"
    else:
        rc, _, e = sh_capture(["curl", "-sfL", "--retry", "3", "-o", str(dest), spec["url"]], 3600)
        err = None if rc == 0 and dest.exists() and dest.stat().st_size > 0 else f"curl rc={rc} {e[-300:]}"
    REPORT.setdefault("downloads", {})[key] = {"seconds": round(time.time() - t0, 1), "error": err,
                                               "bytes": dest.stat().st_size if dest.exists() else 0}
    if err:
        dest.unlink(missing_ok=True)
        return None, err
    return dest, None


def gguf_precision(path):
    """Dominant type of the 2-D weight tensors (by element count), lower-case."""
    try:
        from gguf import GGUFReader
        r = GGUFReader(str(path))
        count = {}
        for t in r.tensors:
            if len(t.shape) >= 2 and int(t.n_elements) >= 4096:
                n = t.tensor_type.name.lower()
                count[n] = count.get(n, 0) + int(t.n_elements)
        if not count:
            return None, {}
        return max(count, key=count.get), count
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {e}"}


class GpuSampler:
    """Peak device memory.used (MiB) while a child runs; idle baseline subtracted by the caller."""

    def __init__(self):
        self.peak = 0
        self.proc = None
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                          "-lms", "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.peak = max(self.peak, int(line.strip().split()[0]))
            except (ValueError, IndexError):
                pass

    def stop(self):
        if not self.proc:
            return self.peak
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.peak


def gpu_idle_mib():
    try:
        rc, out, _ = sh_capture(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 60)
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return 0


def resolve(argv, paths, nthreads):
    out = []
    for a in argv:
        if a == "@NT":
            out.append(str(nthreads))
        elif a.startswith("@M:"):
            out.append(str(paths[a[3:]]))
        else:
            out.append(a)
    return out


def run_arm(binary, argv, backend, env_extra, timeout, cli=True, tag="arm"):
    env = dict(os.environ)
    env.update(LICENCE_ENV)
    env.update(env_extra)
    command = [str(binary)] + list(argv)
    if backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if cli:
        command += ["--gpu-backend", backend]
    out_p, err_p = SCRATCH / f"{tag}.out", SCRATCH / f"{tag}.err"
    idle = gpu_idle_mib()
    sampler = GpuSampler()
    t0 = time.perf_counter()
    with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
        try:
            rc = subprocess.run(command, stdout=fo, stderr=fe, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = time.perf_counter() - t0
    peak = sampler.stop()
    stdout = out_p.read_bytes()
    stderr = err_p.read_bytes().decode("utf-8", "replace")
    gpu_mib = max(0, peak - idle)
    marker = bool(re.search(r"CUDA0|using preferred GPU backend: CUDA|ggml_backend_cuda", stderr))
    return {"rc": rc, "wall_s": round(wall, 3), "gpu_mem_peak_mib": gpu_mib, "cuda_marker": marker,
            "cuda_used": gpu_mib >= 64 or (backend != "cpu" and marker), "stdout": stdout,
            "stderr_tail": stderr[-2500:]}


# ───────────────────────────── comparisons ─────────────────────────────
def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else (1.0 if not na and not nb else float("nan"))


def maxabs(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def flatten_json(obj, nums, strs):
    if isinstance(obj, bool):
        nums.append(float(obj))
    elif isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, str):
        strs.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            flatten_json(x, nums, strs)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            flatten_json(obj[k], nums, strs)


def decode_image(b):
    try:
        import numpy as np
        from PIL import Image
        if b[:8] == b"\x89PNG\r\n\x1a\n" or b[:2] in (b"P5", b"P6", b"BM") or b[:3] == b"\xff\xd8\xff":
            return np.asarray(Image.open(io.BytesIO(b))).astype(np.float64)
    except Exception:  # noqa: BLE001
        return None
    return None


def compare_outputs(a, b):
    """Compare stdout bytes of two arms: image, JSON (numbers + strings) or text."""
    r = {"byte_identical": a == b}
    ia, ib = decode_image(a), decode_image(b)
    if ia is not None and ib is not None:
        import numpy as np
        if ia.shape != ib.shape:
            r["image"] = {"shape_differs": [list(ia.shape), list(ib.shape)]}
            r["final_cosine"] = None
            return r
        d = np.abs(ia - ib)
        mse = float((d ** 2).mean())
        ca, cb = (ia - ia.mean()).ravel(), (ib - ib.mean()).ravel()
        den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
        r["image"] = {"shape": list(ia.shape), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                      "psnr_db": 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse),
                      "n_diff": int((d > 0).sum())}
        uniq = set(np.unique(ia).tolist()) | set(np.unique(ib).tolist())
        if uniq <= {0.0, 255.0}:  # binary mask (SAM2)
            ma, mb = ia > 0, ib > 0
            union = int((ma | mb).sum())
            r["image"]["mask_iou"] = 1.0 if union == 0 else float((ma & mb).sum()) / union
        if den:
            r["final_cosine"] = float(ca @ cb) / den
            r["final_metric"] = "centered pixel cosine"
        else:  # a flat image has no centered direction; fall back to the raw cosine
            ra, rb = ia.ravel(), ib.ravel()
            dr = float(np.linalg.norm(ra) * np.linalg.norm(rb))
            r["final_cosine"] = float(ra @ rb) / dr if dr else 1.0
            r["final_metric"] = "raw pixel cosine (flat image)"
        return r
    sa, sb = a.decode("utf-8", "replace"), b.decode("utf-8", "replace")
    try:
        ja, jb = json.loads(sa), json.loads(sb)
        na, nb, ta, tb = [], [], [], []
        flatten_json(ja, na, ta)
        flatten_json(jb, nb, tb)
        r["json"] = {"n_numbers": [len(na), len(nb)], "strings_equal": ta == tb, "n_strings": [len(ta), len(tb)]}
        if ta != tb:
            r["json"]["string_ratio"] = difflib.SequenceMatcher(None, "\n".join(ta), "\n".join(tb),
                                                                autojunk=False).ratio()
        if na and len(na) == len(nb):
            r["final_cosine"] = cos(na, nb)
            r["max_abs"] = maxabs(na, nb)
            r["final_metric"] = "cosine over all JSON numbers"
        else:
            r["final_cosine"] = None
        return r
    except (ValueError, TypeError):
        pass
    r["text_exact"] = sa.strip() == sb.strip()
    r["char_ratio"] = difflib.SequenceMatcher(None, sa, sb, autojunk=False).ratio() if len(sa) + len(sb) < 400000 \
        else None
    r["final_cosine"] = None
    r["final_metric"] = "decoded text agreement"
    return r


def ocr_boxes(b):
    """Per-box summary for OCR JSON output (list of {text,bbox,conf})."""
    try:
        j = json.loads(b.decode("utf-8", "replace"))
    except ValueError:
        return None
    if isinstance(j, dict):
        j = j.get("regions") or j.get("results") or j.get("lines") or j
    if not isinstance(j, list) or not all(isinstance(x, dict) and "bbox" in x for x in j):
        return None
    return [(x.get("text", ""), x.get("bbox"), x.get("conf")) for x in j]


def ocr_compare(a, b):
    ba, bb = ocr_boxes(a), ocr_boxes(b)
    if ba is None or bb is None:
        return None
    r = {"n_boxes": [len(ba), len(bb)]}
    if len(ba) == len(bb):
        r["texts_equal"] = sum(1 for x, y in zip(ba, bb) if x[0] == y[0])
        r["box_max_dpx"] = max((max(abs(p - q) for p, q in zip(x[1], y[1])) for x, y in zip(ba, bb)
                                if x[1] and y[1]), default=0)
        r["conf_max_d"] = max((abs((x[2] or 0) - (y[2] or 0)) for x, y in zip(ba, bb)), default=0)
    r["joined_text_ratio"] = difflib.SequenceMatcher(None, " ".join(x[0] for x in ba), " ".join(y[0] for y in bb),
                                                     autojunk=False).ratio()
    return r


def embed_vectors(mode, raw):
    text = raw.decode("utf-8", "replace")
    try:
        j = json.loads(text)
    except ValueError:  # JSON lines (clip-text prints one object per text)
        j = [json.loads(ln) for ln in text.splitlines() if ln.strip().startswith("{")]
    if mode == "rerank":
        res = j["results"] if isinstance(j, dict) else j
        key = "index" if res and "index" in res[0] else None
        ordered = sorted(res, key=lambda r: r[key]) if key else res
        return {"scores": [r["score"] for r in ordered], "order": [r.get("index") for r in res]}
    if mode == "embed":
        return {"vecs": [x["embedding"] for x in j]}
    if mode == "colbert":
        return {"tok": [x["vectors"] for x in j]}
    if mode == "sparse":
        return {"sp": [{e["token_id"]: e["weight"] for e in x["sparse"]} for x in j]}
    raise ValueError(mode)


def embed_compare(mode, x, y):
    if mode == "embed":
        cs = [cos(a, b) for a, b in zip(x["vecs"], y["vecs"])]
        return {"cos_min": min(cs), "cos_mean": sum(cs) / len(cs),
                "max_abs": max(maxabs(a, b) for a, b in zip(x["vecs"], y["vecs"]))}
    if mode == "colbert":
        cs, same = [], True
        for a, b in zip(x["tok"], y["tok"]):
            same &= len(a) == len(b)
            cs += [cos(p, q) for p, q in zip(a, b)]
        return {"cos_min": min(cs), "same_token_counts": same}
    if mode == "sparse":
        mx, same, cs = 0.0, True, []
        for a, b in zip(x["sp"], y["sp"]):
            keys = sorted(set(a) | set(b))
            va, vb = [a.get(k, 0.0) for k in keys], [b.get(k, 0.0) for k in keys]
            cs.append(cos(va, vb))
            mx = max(mx, maxabs(va, vb))
            same &= set(a) == set(b)
        return {"cos_min": min(cs), "max_abs": mx, "same_terms": same}
    if mode == "rerank":
        return {"cos_min": cos(x["scores"], y["scores"]), "max_abs": maxabs(x["scores"], y["scores"]),
                "same_order": x["order"] == y["order"]}


def bar_for(prec):
    return BARS.get((prec or "").lower())


# ───────────────────────────── job runners ─────────────────────────────
def strip(arm):
    a = dict(arm)
    out = a.pop("stdout", b"")
    a["stdout_sha256"] = hashlib.sha256(out).hexdigest()
    a["stdout_bytes"] = len(out)
    return a


def run_embed(job, binary, paths, nthreads):
    mode = job["mode"]
    argv0 = ["-t", str(nthreads), "--json", "-f", TEXTS]
    argv0 += {"embed": [], "sparse": ["--sparse"], "colbert": ["--colbert"], "rerank": ["--rerank", QUERY]}[mode]
    rec = {"kind": "embed", "mode": mode, "base": job["base"], "files": {}}
    outs = {}
    for f in job["files"]:
        prec, hist = gguf_precision(paths[f])
        entry = {"precision": prec, "type_hist": hist, "arms": {}}
        rec["files"][f] = entry
        for rep in range(job["repeat"]):
            for name, backend in (("cuda", "cuda"), ("cpu", "cpu")):
                arm = run_arm(binary, ["-m", str(paths[f])] + argv0, backend, {}, job["timeout"], tag=name)
                a = entry["arms"].setdefault(name, {"wall_s": [], "gpu_mem_peak_mib": 0, "cuda_used": False})
                a["wall_s"].append(arm["wall_s"])
                a["rc"] = arm["rc"]
                a["gpu_mem_peak_mib"] = max(a["gpu_mem_peak_mib"], arm["gpu_mem_peak_mib"])
                a["cuda_used"] = a["cuda_used"] or arm["cuda_used"]
                a["stderr_tail"] = arm["stderr_tail"][-1200:]
                if arm["rc"] == 0:
                    try:
                        outs[(f, name)] = embed_vectors(mode, arm["stdout"])
                    except Exception as e:  # noqa: BLE001
                        a["parse_error"] = f"{type(e).__name__}: {e}"
                        a["stdout_head"] = arm["stdout"][:600].decode("utf-8", "replace")
    ref = outs.get((job["base"], "cpu"))
    for f, entry in rec["files"].items():
        c, p = outs.get((f, "cuda")), outs.get((f, "cpu"))
        try:
            if c and p:
                entry["cuda_vs_cpu_same_artifact"] = embed_compare(mode, c, p)
            if c and ref:
                entry["cuda_vs_cpu_reference"] = embed_compare(mode, c, ref)
            if p and ref and f != job["base"]:
                entry["cpu_vs_cpu_reference"] = embed_compare(mode, p, ref)
        except Exception as e:  # noqa: BLE001
            entry["compare_error"] = f"{type(e).__name__}: {e}"
        bar = bar_for(entry["precision"])
        got = (entry.get("cuda_vs_cpu_reference") or {}).get("cos_min")
        entry["bar"] = bar
        entry["qualifies"] = None if got is None or bar is None else bool(got > bar)
    return rec


def run_cmd(job, binary, paths, nthreads, results):
    argv = resolve(job["argv"], paths, nthreads)
    rec = {"kind": "cmd", "argv": [os.path.basename(x) if "/" in x else x for x in argv], "arms": {},
           "precision": None}
    for k in models_in(job):
        rec["precision"] = rec["precision"] or gguf_precision(paths[k])[0]
    outs = {}
    for rep in range(job["repeat"]):
        for name, backend, env in job["arms"]:
            arm = run_arm(binary, argv, backend, env, job["timeout"], tag=name)
            a = rec["arms"].setdefault(name, {"backend": backend, "env": env, "wall_s": [], "gpu_mem_peak_mib": 0,
                                              "cuda_used": False})
            a["wall_s"].append(arm["wall_s"])
            a["rc"] = arm["rc"]
            a["gpu_mem_peak_mib"] = max(a["gpu_mem_peak_mib"], arm["gpu_mem_peak_mib"])
            a["cuda_used"] = a["cuda_used"] or arm["cuda_used"]
            a["stderr_tail"] = arm["stderr_tail"][-1500:]
            a["stdout_sha256"] = hashlib.sha256(arm["stdout"]).hexdigest()
            if decode_image(arm["stdout"]) is None:
                a["stdout"] = arm["stdout"][:20000].decode("utf-8", "replace")
            if arm["rc"] == 0:
                outs[name] = arm["stdout"]
    rec["_outs"] = outs
    if "cpu" in outs:
        for name in outs:
            if name == "cpu":
                continue
            c = compare_outputs(outs[name], outs["cpu"])
            o = ocr_compare(outs[name], outs["cpu"])
            if o:
                c["ocr"] = o
            rec["arms"][name]["vs_cpu"] = c
    ref = results.get(job.get("ref") or "")
    if ref and "cpu" in ref.get("_outs", {}):
        rec["reference_job"] = job["ref"]
        for name in outs:
            c = compare_outputs(outs[name], ref["_outs"]["cpu"])
            o = ocr_compare(outs[name], ref["_outs"]["cpu"])
            if o:
                c["ocr"] = o
            rec["arms"][name]["vs_reference_cpu"] = c
    bar = bar_for(rec["precision"])
    rec["bar"] = bar
    return rec


def run_diff(job, build, paths, nthreads):
    sys.path.insert(0, "tests/regression")
    from run_one import parse_diff_stdout, evaluate_stage_thresholds  # noqa: E402
    binary = next((p for p in build.rglob(job["binary"]) if p.is_file() and os.access(p, os.X_OK)), None)
    rec = {"kind": "diff", "binary": job["binary"], "arms": {}}
    if not binary:
        rec["error"] = "binary not built"
        return rec
    argv = resolve(job["argv"], paths, nthreads)
    env = {k: (str(paths[v[3:]]) if v.startswith("@M:") else v) for k, v in job["env"].items()}
    for name, backend in (("cuda", "cuda"), ("cpu", "cpu")):
        arm = run_arm(binary, argv, backend, env, job["timeout"], cli=False, tag=name)
        text = arm["stdout"].decode("utf-8", "replace") + "\n" + arm["stderr_tail"]
        full_err = (SCRATCH / f"{name}.err").read_text(errors="replace")
        stages = parse_diff_stdout(arm["stdout"].decode("utf-8", "replace") + "\n" + full_err)
        fails, missing = evaluate_stage_thresholds(stages, job["thresholds"]) if stages else ([], [])
        worst = min(stages.items(), key=lambda kv: kv[1]) if stages else None
        rec["arms"][name] = {"rc": arm["rc"], "wall_s": arm["wall_s"], "gpu_mem_peak_mib": arm["gpu_mem_peak_mib"],
                             "cuda_used": arm["cuda_used"], "n_stages": len(stages), "stages": stages,
                             "worst_stage": worst, "threshold_fails": fails, "threshold_missing": missing,
                             "stdout_tail": text[-3000:]}
    return rec


def scrub(obj):
    """Drop private bytes and anything that would name the Kaggle account."""
    acct = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_ACCOUNT")
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if acct:
        s = s.replace(acct, "${KAGGLE_ACCOUNT}")
    s = s.replace(str(SCRATCH), "$SCRATCH")
    return json.loads(s)


def cpu_model():
    try:
        return next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None)
    except OSError:
        return None


def main(build, batch, commit=None, gpu=None):
    build = Path(build)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "receipts").mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("CUDA_AUDIT_NO_PIP"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "numpy", "pillow"], timeout=900)
    binary = next(p for p in build.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))
    QUANTIZE[0] = next((p for p in build.rglob("crispembed-quantize") if p.is_file()), None)
    nthreads = os.cpu_count() or 4
    REPORT.update({"commit": commit, "gpu": gpu, "batch": batch, "threads": nthreads,
                   "cpu_model": cpu_model(),
                   "date": time.strftime("%Y-%m-%d")})
    jobs = jobs_for(batch)
    last_use = {}
    for i, j in enumerate(jobs):
        for k in models_in(j):
            last_use[k] = i
    save("setup")
    results = {}
    for i, job in enumerate(jobs):
        t0 = time.time()
        paths, missing = {}, {}
        try:
            for k in models_in(job):
                p, err = fetch(k)
                if p:
                    paths[k] = p
                else:
                    missing[k] = err
            if missing:
                rec = {"kind": job["kind"], "error": "model fetch failed", "missing": missing}
            elif job["kind"] == "embed":
                rec = run_embed(job, binary, paths, nthreads)
            elif job["kind"] == "cmd":
                rec = run_cmd(job, binary, paths, nthreads, results)
            else:
                rec = run_diff(job, build, paths, nthreads)
        except Exception as e:  # noqa: BLE001  (a failed job is evidence too; keep going)
            import traceback
            rec = {"kind": job["kind"], "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        rec["seconds"] = round(time.time() - t0, 1)
        results[job["id"]] = rec
        public = scrub({k: v for k, v in rec.items() if k != "_outs"})
        public.update({"id": job["id"], "commit": commit, "gpu": gpu, "date": REPORT["date"]})
        (OUT / "receipts" / f"{job['id']}.json").write_text(json.dumps(public, indent=1, ensure_ascii=False) + "\n")
        REPORT["jobs"][job["id"]] = summarize(job, public)
        for k in list(paths):
            if last_use.get(k, -1) <= i:
                paths[k].unlink(missing_ok=True)
        for name, r in results.items():  # keep reference outputs only while a later job names them
            if "_outs" in r and not any(jj.get("ref") == name for jj in jobs[i + 1:]):
                r.pop("_outs")
        save(f"{i + 1}/{len(jobs)} {job['id']}")
    save("done")
    return 0


def summarize(job, rec):
    s = {"kind": rec.get("kind"), "seconds": rec.get("seconds")}
    if rec.get("error"):
        s["error"] = rec["error"]
    if rec.get("kind") == "embed":
        for f, e in rec.get("files", {}).items():
            s[f] = {"precision": e.get("precision"), "bar": e.get("bar"), "qualifies": e.get("qualifies"),
                    "cos_cuda_vs_ref": (e.get("cuda_vs_cpu_reference") or {}).get("cos_min"),
                    "cos_cuda_vs_cpu": (e.get("cuda_vs_cpu_same_artifact") or {}).get("cos_min"),
                    "cos_cpu_vs_ref": (e.get("cpu_vs_cpu_reference") or {}).get("cos_min"),
                    "wall": {k: a.get("wall_s") for k, a in e.get("arms", {}).items()},
                    "cuda_used": (e.get("arms", {}).get("cuda") or {}).get("cuda_used")}
    elif rec.get("kind") == "cmd":
        s["precision"] = rec.get("precision")
        for name, a in rec.get("arms", {}).items():
            v = a.get("vs_cpu") or {}
            s[name] = {"rc": a.get("rc"), "wall": a.get("wall_s"), "cuda_used": a.get("cuda_used"),
                       "identical": v.get("byte_identical"), "cos": v.get("final_cosine"),
                       "char_ratio": v.get("char_ratio"), "ocr": v.get("ocr"),
                       "image": {k: v["image"].get(k) for k in ("max_abs", "psnr_db", "mask_iou")
                                 if k in v.get("image", {})} if v.get("image") else None,
                       "ref_cos": (a.get("vs_reference_cpu") or {}).get("final_cosine")}
    elif rec.get("kind") == "diff":
        for name, a in rec.get("arms", {}).items():
            s[name] = {"rc": a.get("rc"), "n": a.get("n_stages"), "worst": a.get("worst_stage"),
                       "fails": a.get("threshold_fails"), "wall": a.get("wall_s"), "cuda_used": a.get("cuda_used")}
    return s


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--list":
        for j in jobs_for(sys.argv[2]):
            print(j["id"], j["kind"], models_in(j))
        print("targets:", build_targets(sys.argv[2]))
        sys.exit(0)
    sys.exit(main(sys.argv[1], sys.argv[2]))
