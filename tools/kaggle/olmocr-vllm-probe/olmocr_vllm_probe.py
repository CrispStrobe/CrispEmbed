#!/usr/bin/env python3
"""Does the olmOCR toolkit's own serving stack run on Turing? (brief A3, follow-up)

The main A3 kernel produced its gold through transformers because
`olmocr[gpu]` pins `vllm==0.11.2` and Kaggle's cards are sm_75.  Its end-of-run
probe was inconclusive: vLLM installed and accepted the cards, then refused to
start because the *same process* still held ~7-9 GiB of VRAM from the
transformers model.  "Free memory on device (7.14/14.56 GiB) ... less than
desired GPU memory utilization" is a statement about that leftover, not about
Turing.

So this kernel loads nothing else.  It installs vLLM, starts it on the same
checkpoint, and — if it starts — replays the toolkit's own requests through it,
so the serving-layer deviation in the gold can be quantified rather than
asserted.

Outputs under /kaggle/working: vllm_probe.json, and on success
gold_vllm/<corpus>/{<stem>.raw.txt,<stem>.txt,pages.json} plus run.log.
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
WORK.mkdir(parents=True, exist_ok=True)
_LOG = open(WORK / "run.log", "w", buffering=1)


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


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _hook(t, e, tb):
    traceback.print_exception(t, e, tb, file=_LOG)
    _LOG.flush()
    traceback.print_exception(t, e, tb, file=sys.__stderr__)


sys.excepthook = _hook

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/parity-olmocr")
MODEL = os.environ.get("OLMOCR_MODEL", "allenai/olmOCR-2-7B-1025")
REPO = Path("/kaggle/temp/CrispEmbed")
PDFS = Path("/tmp/olmocr_pdfs")
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

result = {"model": MODEL, "pin": "vllm==0.11.2 (olmocr[gpu])"}


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def save():
    (WORK / "vllm_probe.json").write_text(json.dumps(result, indent=2) + "\n")


sh("nvidia-smi || true", check=False)
sh("apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y poppler-utils "
   ">/dev/null 2>&1 || true", check=False)
# vLLM first: it pins torch, and olmocr's base install must not fight it.
t0 = time.time()
p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.11.2"],
                   capture_output=True, text=True)
result["install_rc"] = p.returncode
result["install_s"] = round(time.time() - t0, 1)
result["install_tail"] = (p.stderr or p.stdout)[-1500:]
save()
if p.returncode != 0:
    raise SystemExit(f"vllm install failed rc={p.returncode}")
sh("pip install -q olmocr==0.4.27 img2pdf 2>&1 | tail -3", check=False)

import torch  # noqa: E402

caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
result["torch"] = torch.__version__
result["arch_list"] = torch.cuda.get_arch_list()
result["caps"] = caps
result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
result["free_gib_before_init"] = [round(torch.cuda.mem_get_info(i)[0] / 2 ** 30, 2)
                                  for i in range(torch.cuda.device_count())]
print(json.dumps({k: result[k] for k in
                  ("torch", "arch_list", "caps", "gpus", "free_gib_before_init")}, indent=2))
save()
missing = [f"sm_{c[0]}{c[1]}" for c in caps if f"sm_{c[0]}{c[1]}" not in result["arch_list"]]
if missing:
    result["verdict"] = f"wrong accelerator draw: torch has no kernels for {missing}"
    save()
    raise SystemExit(result["verdict"])

if not REPO.exists():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
sys.path.insert(0, str(REPO / "tests"))
from ocr_external_parity import load_fixtures  # noqa: E402


def find_dataset(name):
    root = Path("/kaggle/input")
    for cand in root.rglob(name) if root.exists() else []:
        if cand.is_dir():
            return cand
    return None


CORPORA = {"synth": find_dataset("crispembed-ocr-synth"),
           "cc0": REPO / "tests" / "regression" / "images" / "cc0"}
if CORPORA["synth"] is None:
    raise SystemExit("crispembed-ocr-synth dataset not mounted")
fixtures = {c: [f for f in load_fixtures(p) if f["truth"]] for c, p in CORPORA.items()}

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

from olmocr.pipeline import build_page_query  # noqa: E402
from olmocr.prompts import PageResponse  # noqa: E402
from olmocr.train.front_matter import FrontMatterParser  # noqa: E402

parser = FrontMatterParser(front_matter_class=PageResponse)

import vllm  # noqa: E402

result["vllm"] = vllm.__version__
save()
t1 = time.time()
try:
    llm = vllm.LLM(model=MODEL, dtype="half",
                   tensor_parallel_size=torch.cuda.device_count(),
                   max_model_len=16384, gpu_memory_utilization=0.90,
                   limit_mm_per_prompt={"image": 1}, enforce_eager=True)
    result["init"] = "ok"
    result["init_s"] = round(time.time() - t1, 1)
except BaseException as e:
    result["init"] = "failed"
    result["init_s"] = round(time.time() - t1, 1)
    result["error"] = f"{type(e).__name__}: {e}"[:4000]
    result["traceback"] = traceback.format_exc()[-8000:]
    result["verdict"] = ("vLLM 0.11.2 installed and accepted sm_75 but the engine "
                         "did not start on a clean box; see error")
    save()
    print(json.dumps(result, indent=2))
    raise SystemExit(1)

save()
print(f"vLLM engine up in {result['init_s']}s")

# Greedy, to be comparable with the transformers gold pass; the toolkit's own
# attempt-0 temperature (0.1) is recorded in the main kernel's sampled pass.
sp = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8000)
GOLD = WORK / "gold_vllm"
for corpus, fx_list in fixtures.items():
    d = GOLD / corpus
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
            page = parser._parse_front_matter(fm, text)
            nat = page.natural_text or ""
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
        print(f"  [{corpus}] {fx['name']:32} {finish} {gen_s}s chars={len(nat)}", flush=True)
    result.setdefault("pages", {})[corpus] = pages
    save()

result["verdict"] = "vLLM 0.11.2 served olmOCR-2-7B on 2x Tesla T4 (sm_75)"
save()
print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))
shutil.rmtree(REPO, ignore_errors=True)
print("done")
