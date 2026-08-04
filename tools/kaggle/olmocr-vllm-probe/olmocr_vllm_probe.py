#!/usr/bin/env python3
"""Does the olmOCR toolkit's own serving stack run on Turing? (brief A3, follow-up)

The main A3 kernel produced its gold through transformers because
`olmocr[gpu]` pins `vllm==0.11.2` and Kaggle's cards are sm_75.  Two earlier
attempts to answer the question failed for reasons that had nothing to do with
Turing, and both are worth stating because they are the traps:

1. Probing at the end of the gold kernel: vLLM installed and accepted the
   cards, then refused to start — "Free memory on device (7.14/14.56 GiB) ...
   less than desired GPU memory utilization". That is the transformers model
   still resident in the same process, not a platform limit.
2. Probing in its own kernel, but with `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   and `torch.cuda` touched before `vllm.LLM(...)`: the parent's CUDA context
   forces vLLM off fork, spawn re-imports the kernel script, and the re-import
   runs the whole file again — "An attempt has been made to start a new process
   before the current process has finished its bootstrapping phase". Also not a
   platform limit.
3. Fixed both, and got the first genuinely platform-shaped failure: Turing has
   no FlashAttention-2 (`FA2 is only supported on devices with compute
   capability >= 8`), so vLLM chose FLASHINFER, which JIT-builds its prefill
   kernels — and the link failed with `cannot find -lcuda`, because Kaggle
   ships the driver as `libcuda.so.1` with no linker symlink. Environmental.
4. Forcing `TRITON_ATTN` to sidestep the JIT hit a hard model-side refusal:
   `Qwen2.5-VL does not support AttentionBackendEnum.TRITON_ATTN backend now`.

So sm_75 offers exactly three backends — FLASHINFER, TRITON_ATTN,
FLEX_ATTENTION — one of which this model rejects outright and one of which
needs a JIT that the image's linker paths broke. This version installs the
`libcuda.so` symlink and then tries the remaining backends **each in its own
subprocess**, so one refusal does not end the run and every outcome is
recorded. If an engine starts, it replays the toolkit's own requests so the
serving-layer deviation in the gold can be quantified rather than asserted.

Outputs under /kaggle/working: vllm_probe.json, run.log, and on success
gold_vllm/<corpus>/{<stem>.raw.txt,<stem>.txt,pages.json}.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/parity-olmocr")
MODEL = os.environ.get("OLMOCR_MODEL", "allenai/olmOCR-2-7B-1025")
REPO = Path("/kaggle/temp/CrispEmbed")
PDFS = Path("/tmp/olmocr_pdfs")

os.environ["HF_HOME"] = "/tmp/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            try:
                st.write(x)
            except Exception:
                pass

    def flush(self):
        for st in self.s:
            try:
                st.flush()
            except Exception:
                pass


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def gpu_info():
    """Identify the cards without importing torch.cuda.

    Touching torch.cuda here would create a CUDA context in this process, which
    pushes vLLM off fork and into spawn — see the module docstring.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    cards = []
    for line in out.strip().splitlines():
        name, cap, mem = [x.strip() for x in line.split(",")]
        cards.append({"name": name, "compute_cap": cap, "memory": mem})
    return cards


def find_dataset(name):
    root = Path("/kaggle/input")
    for cand in (root.rglob(name) if root.exists() else []):
        if cand.is_dir():
            return cand
    return None


# sm_75's whole menu, in the order worth trying.  FLASHINFER is what vLLM picks
# on its own; FLEX_ATTENTION is the only other one Qwen2.5-VL has not already
# refused.  TRITON_ATTN is skipped: the model raises on it by name.
BACKENDS = os.environ.get("OLMOCR_VLLM_BACKENDS", "FLASHINFER,FLEX_ATTENTION").split(",")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    worker_backend = os.environ.get("OLMOCR_VLLM_BACKEND", "")
    tag = f"-{worker_backend}" if worker_backend else ""
    log = open(WORK / f"run{tag}.log", "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)

    result = {"model": MODEL, "pin": "vllm==0.11.2 (olmocr[gpu])",
              "backend": worker_backend or None}
    out_json = WORK / (f"vllm_probe{tag}.json" if worker_backend else "vllm_probe.json")

    def save():
        out_json.write_text(json.dumps(result, indent=2) + "\n")

    if not worker_backend:
        sh("nvidia-smi || true", check=False)
        # ~10 minutes on a cold Kaggle image; the per-backend workers inherit the
        # installed package and must not pay for it again.
        sh("apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y poppler-utils "
           ">/dev/null 2>&1 || true", check=False)

    cards = gpu_info()
    result["gpus"] = cards
    save()
    print(f"cards={cards}")
    if not cards:
        result["verdict"] = "no GPU"
        save()
        return 1
    if any(float(c["compute_cap"]) < 7.0 for c in cards):
        result["verdict"] = (f"wrong accelerator draw: {cards}; re-push pinning "
                             f"machine_shape NvidiaTeslaT4")
        save()
        return 1

    if not worker_backend:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.11.2"],
                           capture_output=True, text=True)
        result["install_rc"] = p.returncode
        result["install_s"] = round(time.time() - t0, 1)
        result["install_tail"] = (p.stderr or p.stdout)[-1500:]
        save()
        if p.returncode != 0:
            result["verdict"] = f"vllm install failed rc={p.returncode}"
            save()
            return 1
        sh("pip install -q olmocr==0.4.27 img2pdf 2>&1 | tail -3", check=False)

        # FLASHINFER JIT-compiles its prefill kernels with ninja and links against
        # -lcuda.  Kaggle ships the driver as libcuda.so.1 with no linker symlink,
        # so that link dies with "cannot find -lcuda" after ~4 minutes of nvcc.
        libcuda = subprocess.run("ldconfig -p | grep -m1 'libcuda\\.so\\.1'", shell=True,
                                 capture_output=True, text=True).stdout.strip()
        result["libcuda_so_1"] = libcuda
        src = libcuda.split("=>")[-1].strip() if "=>" in libcuda else ""
        if src:
            r = subprocess.run(["ln", "-sf", src, "/usr/lib/x86_64-linux-gnu/libcuda.so"],
                               capture_output=True, text=True)
            result["libcuda_symlink_rc"] = r.returncode
            result["libcuda_symlink_present"] = Path(
                "/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
        save()
        print(f"libcuda={libcuda} symlink={result.get('libcuda_symlink_present')}")

        # Each backend in its own process: a refusal or a JIT failure ends that
        # attempt, not the run, and the next one still gets measured.
        result["attempts"] = []
        for backend in BACKENDS:
            print(f"=== trying VLLM_ATTENTION_BACKEND={backend} ===", flush=True)
            env = dict(os.environ, OLMOCR_VLLM_BACKEND=backend,
                       VLLM_ATTENTION_BACKEND=backend)
            t = time.time()
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                                env=env).returncode
            sub = WORK / f"vllm_probe-{backend}.json"
            entry = json.loads(sub.read_text()) if sub.exists() else {}
            entry["rc"] = rc
            entry["wall_s"] = round(time.time() - t, 1)
            result["attempts"].append({k: v for k, v in entry.items() if k != "pages"})
            save()
            if entry.get("init") == "ok":
                result["verdict"] = (
                    f"vLLM {entry.get('vllm')} served {MODEL} on {len(cards)}x "
                    f"{cards[0]['name']} (cc {cards[0]['compute_cap']}) with "
                    f"VLLM_ATTENTION_BACKEND={backend}")
                save()
                print(result["verdict"])
                return 0
        result["verdict"] = (
            f"vLLM 0.11.2 did not serve {MODEL} on cc {cards[0]['compute_cap']}: "
            f"no available attention backend started "
            f"({', '.join(BACKENDS)}; TRITON_ATTN is refused by the model itself)")
        save()
        print(json.dumps(result, indent=2))
        return 1

    if not REPO.exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
    sys.path.insert(0, str(REPO / "tests"))
    from ocr_external_parity import load_fixtures

    corpora = {"synth": find_dataset("crispembed-ocr-synth"),
               "cc0": REPO / "tests" / "regression" / "images" / "cc0"}
    if corpora["synth"] is None:
        result["verdict"] = "crispembed-ocr-synth dataset not mounted"
        save()
        return 1
    fixtures = {c: [f for f in load_fixtures(p) if f["truth"]] for c, p in corpora.items()}

    PDFS.mkdir(parents=True, exist_ok=True)
    pdf_of = {}
    for corpus, fx_list in fixtures.items():
        (PDFS / corpus).mkdir(exist_ok=True)
        for fx in fx_list:
            pdf = PDFS / corpus / (Path(fx["name"]).stem + ".pdf")
            r = subprocess.run(["img2pdf", "--output", str(pdf), str(fx["path"])],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            pdf_of[(corpus, fx["name"])] = pdf
    print(f"wrapped {len(pdf_of)} fixtures")

    from olmocr.pipeline import build_page_query
    from olmocr.prompts import PageResponse
    from olmocr.train.front_matter import FrontMatterParser

    parser = FrontMatterParser(front_matter_class=PageResponse)

    import vllm

    result["vllm"] = vllm.__version__
    save()
    t1 = time.time()
    try:
        llm = vllm.LLM(model=MODEL, dtype="half",
                       tensor_parallel_size=len(cards),
                       max_model_len=16384, gpu_memory_utilization=0.90,
                       limit_mm_per_prompt={"image": 1}, enforce_eager=True)
        result["init"] = "ok"
        result["init_s"] = round(time.time() - t1, 1)
    except BaseException as e:
        result["init"] = "failed"
        result["init_s"] = round(time.time() - t1, 1)
        result["error"] = f"{type(e).__name__}: {e}"[:4000]
        result["traceback"] = traceback.format_exc()[-8000:]
        result["verdict"] = ("vLLM 0.11.2 installed and accepted the cards but the "
                             "engine did not start; see error")
        save()
        print(json.dumps(result, indent=2))
        return 1
    save()
    print(f"vLLM engine up in {result['init_s']}s")

    # Greedy, to be comparable with the transformers gold pass; the toolkit's own
    # attempt-0 temperature (0.1) is measured in the main kernel's sampled pass.
    sp = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8000)
    gold = WORK / "gold_vllm"
    for corpus, fx_list in fixtures.items():
        d = gold / corpus
        d.mkdir(parents=True, exist_ok=True)
        pages = []
        for fx in fx_list:
            q = asyncio.run(build_page_query(str(pdf_of[(corpus, fx["name"])]), 1, 1288,
                                             model_name=MODEL))
            t = time.time()
            out = llm.chat(q["messages"], sampling_params=sp)
            gen_s = round(time.time() - t, 2)
            raw = out[0].outputs[0].text
            finish = out[0].outputs[0].finish_reason
            nat, err = "", None
            try:
                fm, text = parser._extract_front_matter_and_text(raw)
                nat = parser._parse_front_matter(fm, text).natural_text or ""
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            stem = Path(fx["name"]).stem
            (d / f"{stem}.raw.txt").write_text(raw)
            (d / f"{stem}.txt").write_text(nat)
            pages.append({"fixture": fx["name"], "stem": stem, "gen_s": gen_s,
                          "finish_reason": finish, "parse_error": err,
                          "prompt_tokens": len(out[0].prompt_token_ids or []),
                          "completion_tokens": len(out[0].outputs[0].token_ids),
                          "natural_text_chars": len(nat), "raw_chars": len(raw)})
            (d / "pages.json").write_text(json.dumps(pages, indent=2) + "\n")
            print(f"  [{corpus}] {fx['name']:32} {finish} {gen_s}s chars={len(nat)}",
                  flush=True)
        result.setdefault("pages", {})[corpus] = pages
        save()

    result["verdict"] = (f"vLLM {result['vllm']} served {MODEL} on "
                         f"{len(cards)}x {cards[0]['name']} (cc {cards[0]['compute_cap']})")
    save()
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))
    shutil.rmtree(REPO, ignore_errors=True)
    print("done")
    return 0


if __name__ == "__main__":
    # A vLLM worker started with spawn re-imports this file; the guard makes that
    # re-import inert instead of recursively re-running the kernel.
    raise SystemExit(main())
