#!/usr/bin/env python3
"""Reconvert a Qwen3-VL-family GGUF from upstream, verify it, quantize it.

The published Uni-MuMER F16 GGUF had three zeroed tensors in LLM layer 18
(ffn_norm, attn_q_norm, attn_k_norm: cos 0.0 vs upstream, every other tensor
exact) and all its quants were derived from it. This rebuilds the set on a
clean runner and gates it:

  1. models/convert-qwen3vl-to-gguf.py --dtype f16
  2. gguf_vs_safetensors.py: every LLM tensor within F16 rounding of upstream
  3. qwen3vl_stage_diff.py on the new file: exact OCR text + stage gate
  4. crispembed-quantize to q8_0 q6_k q4_k q3_k q2_k
  5. --upload: push the set to the HF repo (needs HF_TOKEN write access to it)

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/reconvert_qwen3vl.py \\
        -f args="--profile unimumer" \\
        -f pip="torch torchvision transformers>=4.57 accelerate pillow gguf safetensors"
"""
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILES = {
    "unimumer": dict(hf="phxember/Uni-MuMER-Qwen3-VL-2B", stem="uni-mumer-qwen3-vl-2b",
                     repo="cstr/uni-mumer-qwen3-vl-2b-GGUF"),
}
prof = PROFILES[sys.argv[sys.argv.index("--profile") + 1]]
QUANTS = ["q8_0", "q6_k", "q4_k", "q3_k", "q2_k"]
res = {"profile": prof, "steps": {}}


def step(name, cmd, env=None):
    print(f"== {name}: {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd], env=dict(os.environ, **(env or {})))
    res["steps"][name] = r.returncode
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    return r.returncode == 0


f16 = SCR / f"{prof['stem']}-f16.gguf"
ok = step("convert", [sys.executable, REPO / "models/convert-qwen3vl-to-gguf.py", "--model", prof["hf"],
                      "--dtype", "f16", "--output", f16])
sub_out = lambda n: {"HEAVY_OUT": str(OUT / n)}
ok = ok and step("verify_tensors", [sys.executable, HERE / "gguf_vs_safetensors.py", "--hf", prof["hf"],
                                    "--gguf-local", f16], sub_out("verify"))
ok = ok and step("stage_diff", [sys.executable, HERE / "qwen3vl_stage_diff.py", "--model", "unimumer",
                                "--gguf-local", f16], sub_out("stage_diff"))
files = [f16]
if ok:
    b = SCR / "build"  # configured by qwen3vl_stage_diff.py
    step("build_quantize", ["cmake", "--build", b, "--target", "crispembed-quantize", f"-j{os.cpu_count() or 4}"])
    q = next(p for p in b.rglob("crispembed-quantize") if p.is_file() and os.access(p, os.X_OK))
    for t in QUANTS:
        dst = SCR / f"{prof['stem']}-{t}.gguf"
        if step(f"quantize_{t}", [q, f16, dst, t]):
            files.append(dst)
    res["sizes_gb"] = {p.name: round(p.stat().st_size / 1e9, 2) for p in files}
if ok and "--upload" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        for p in files:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=prof["repo"],
                            commit_message=f"Reconvert {p.name} from {prof['hf']} (the previous F16 had zeroed "
                                           f"layer-18 norm tensors; every quant derived from it)")
        res["uploaded"] = [p.name for p in files]
    except Exception as e:  # typically: the CI token is not scoped to this repo
        res["upload_error"] = repr(e)[:500]
        ok = False
(OUT / "result.json").write_text(json.dumps(res, indent=1))
summary = [f"### reconvert {prof['hf']}\n", f"steps: {res['steps']}", f"sizes: {res.get('sizes_gb')}",
           f"uploaded: {res.get('uploaded')}  error: {res.get('upload_error')}"]
for sub in ("verify", "stage_diff"):
    s = OUT / sub / "summary.md"
    if s.exists():
        summary += ["", s.read_text()]
(OUT / "summary.md").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
sys.exit(0 if ok else 1)
