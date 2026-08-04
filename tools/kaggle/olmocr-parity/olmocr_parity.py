#!/usr/bin/env python3
"""olmOCR toolkit parity arm on a Kaggle GPU (brief A3).

Produces two things the native olmOCR lane needs and cannot produce itself:

1. **The page-request contract, captured at runtime from the toolkit's own
   code.**  Nothing here re-implements `build_page_query`: the request is built
   by importing it, so the prompt string, the message order, the render
   geometry and the sampling parameters are whatever olmocr actually does on
   the day, not what a source reading concluded.
2. **Document-level gold**: the raw model output per page (front matter
   included) and the parsed `natural_text`.

The toolkit consumes PDFs, so every fixture is wrapped into a single-page PDF
with img2pdf and handed to the toolkit's renderer.  That wrap is lossless: a
PNG is embedded as a Flate stream and a JPEG passes through as DCT, so no
resampling happens before the toolkit's own pdftoppm render.

Serving layer: the toolkit drives a vLLM OpenAI server.  vLLM 0.11.2 (the pin
in `olmocr[gpu]`) is V1-only and Kaggle's cards are Turing (sm_75), so the
serving layer is replaced with transformers generation while the *request* is
still constructed by the toolkit's own code.  Whether vLLM would have started
is measured rather than assumed - see the probe at the end, which runs in a
subprocess after every artifact is already on disk so it cannot cost anything.

Outputs under /kaggle/working: contract.json, gold/<corpus>/*.{raw.txt,txt},
gold/<corpus>/{manifest,pages}.json, sampled/<corpus>/*.raw.txt, summary.json,
vllm_probe.json, run.log.
"""
import asyncio
import base64
import io
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

# kernels_output does not expose stderr (kaggle_usage gotcha #15) - tee
# everything, including a fatal traceback, into a downloadable file.
_LOG = open(WORK / "run.log", "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _excepthook(exc_type, exc, tb):
    _LOG.write("\n=== FATAL ===\n")
    traceback.print_exception(exc_type, exc, tb, file=_LOG)
    _LOG.flush()
    traceback.print_exception(exc_type, exc, tb, file=sys.__stderr__)


sys.excepthook = _excepthook

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/parity-olmocr")
# The toolkit's own default is the FP8 checkpoint, which needs sm_89+ for the
# quantised kernels; the unquantised sibling is the same fine-tune and is what
# runs here.  Recorded as a deviation.
MODEL = os.environ.get("OLMOCR_MODEL", "allenai/olmOCR-2-7B-1025")
TOOLKIT_DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
OLMOCR_PIN = "0.4.27"
TRANSFORMERS_PIN = "4.57.3"  # the pin in olmocr[gpu]
TARGET_LONGEST = 1288
MAX_ATTEMPTS = int(os.environ.get("OLMOCR_MAX_ATTEMPTS", "4"))
# A degenerate repeat loop would spend the whole 8000-token budget; the budget
# itself stays at the contract value, but a page that runs longer than this is
# cut off and flagged rather than eating the kernel's wall clock.
PAGE_TIME_LIMIT_S = float(os.environ.get("OLMOCR_PAGE_TIME_LIMIT", "420"))

REPO = Path("/kaggle/temp/CrispEmbed")
PDFS = Path("/tmp/olmocr_pdfs")
GOLD = WORK / "gold"
SAMPLED = WORK / "sampled"

os.environ["HF_HOME"] = "/tmp/hf"           # 15.5 GiB must not land in the 20 GB output mount
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find_dataset(*names) -> Path | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for name in names:
        for cand in root.rglob(name):
            if cand.is_dir():
                return cand
    return None


print("=" * 70)
print("Step 1: environment")
print("=" * 70)
sh("nvidia-smi || true", check=False)
sh("apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y poppler-utils "
   ">/dev/null 2>&1 || true", check=False)
sh(f"pip install -q olmocr=={OLMOCR_PIN} img2pdf 2>&1 | tail -3", check=False)
sh(f"pip install -q transformers=={TRANSFORMERS_PIN} accelerate 2>&1 | tail -3", check=False)
sh("pdftoppm -v 2>&1 | head -1 || true", check=False)
sh("img2pdf --version || true", check=False)

import torch  # noqa: E402
import transformers  # noqa: E402

n_gpu = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(n_gpu)]
names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
vram = [torch.cuda.get_device_properties(i).total_memory / 2 ** 30 for i in range(n_gpu)]
arch_list = torch.cuda.get_arch_list()
# bf16 arithmetic needs Ampere or newer; on an older card torch accepts the
# dtype and then runs an emulated path, so pick fp16 explicitly and record it.
dtype = "bfloat16" if caps and all(c[0] >= 8 for c in caps) else "float16"
hardware = f"Kaggle {n_gpu}x {names[0] if names else 'CPU'} ({sum(vram):.0f} GiB VRAM total)"
print(f"torch={torch.__version__} transformers={transformers.__version__}")
print(f"gpus={names} caps={caps} vram={[round(v, 1) for v in vram]}")
print(f"arch_list={arch_list}")
print(f"dtype={dtype}  hardware={hardware}")

# `torch.cuda.is_available()` is True on a card the installed wheel has no SASS
# for; the failure surfaces only at the first kernel launch, ~90 s in.  Compare
# the card against the wheel's compiled arch list up front so a bad accelerator
# draw costs one line instead of a GPU-hour. (Measured on A2: a P100 draw died
# exactly this way - cu128 wheels ship no sm_60.)
missing = [f"sm_{c[0]}{c[1]}" for c in caps if f"sm_{c[0]}{c[1]}" not in arch_list]
if missing:
    raise SystemExit(
        f"torch {torch.__version__} has no kernels for {missing} "
        f"(compiled for {arch_list}); re-push pinning a supported accelerator "
        f"via kernel-metadata machine_shape")
WEIGHTS_GIB = 15.45
if sum(vram) < WEIGHTS_GIB + 2:
    raise SystemExit(
        f"{sum(vram):.1f} GiB VRAM cannot hold {WEIGHTS_GIB} GiB of weights; "
        f"re-push pinning a larger accelerator")

print("=" * 70)
print("Step 2: repo, fixtures, single-page PDFs")
print("=" * 70)
if not REPO.exists():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
sh("git log --oneline -1", cwd=REPO)
repo_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()

synth = find_dataset("crispembed-ocr-synth")
cc0 = REPO / "tests" / "regression" / "images" / "cc0"
print(f"synth fixtures: {synth}")
print(f"cc0 fixtures:   {cc0}")
if synth is None:
    raise SystemExit("crispembed-ocr-synth dataset not mounted")

sys.path.insert(0, str(REPO / "tests"))
from ocr_external_parity import load_fixtures  # noqa: E402

CORPORA = {"synth": synth, "cc0": cc0}
fixtures = {c: [f for f in load_fixtures(p) if f["truth"]] for c, p in CORPORA.items()}
for c, fx in fixtures.items():
    print(f"{c}: {len(fx)} labelled fixtures")

img2pdf_ver = subprocess.run(["img2pdf", "--version"], capture_output=True,
                             text=True).stdout.strip()
pdftoppm_ver = subprocess.run(["pdftoppm", "-v"], capture_output=True,
                              text=True).stderr.splitlines()[0]

PDFS.mkdir(parents=True, exist_ok=True)
pdf_of = {}
for corpus, fx_list in fixtures.items():
    (PDFS / corpus).mkdir(exist_ok=True)
    for fx in fx_list:
        pdf = PDFS / corpus / (Path(fx["name"]).stem + ".pdf")
        cmd = ["img2pdf", "--output", str(pdf), str(fx["path"])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"img2pdf failed for {fx['name']}: {r.stderr}")
        pdf_of[(corpus, fx["name"])] = pdf
print(f"wrapped {len(pdf_of)} fixtures into single-page PDFs with {img2pdf_ver}")

print("=" * 70)
print("Step 3: capture the toolkit's request contract at runtime")
print("=" * 70)
import olmocr  # noqa: E402
from olmocr.pipeline import TEMPERATURE_BY_ATTEMPT, build_page_query  # noqa: E402
from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt  # noqa: E402
from olmocr.train.front_matter import FrontMatterParser  # noqa: E402
from PIL import Image  # noqa: E402

PROMPT = build_no_anchoring_v4_yaml_prompt()
import hashlib  # noqa: E402

prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
print(f"prompt sha256={prompt_sha} bytes={len(PROMPT.encode())}")
print(f"TEMPERATURE_BY_ATTEMPT={TEMPERATURE_BY_ATTEMPT}")


def page_query(pdf: Path, rotation: int = 0) -> dict:
    """The toolkit's own request builder - not a re-implementation."""
    return asyncio.run(build_page_query(str(pdf), 1, TARGET_LONGEST,
                                        image_rotation=rotation, model_name=MODEL))


def query_image(q: dict) -> Image.Image:
    url = q["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,"), url[:40]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


renders = []
skeleton = None
for (corpus, name), pdf in pdf_of.items():
    q = page_query(pdf)
    img = query_image(q)
    with Image.open([f for f in fixtures[corpus] if f["name"] == name][0]["path"]) as src:
        src_wh = list(src.size)
    renders.append({"corpus": corpus, "fixture": name, "pdf": pdf.name,
                    "source_wh": src_wh, "rendered_wh": list(img.size),
                    "rendered_longest": max(img.size)})
    if skeleton is None:
        skeleton = json.loads(json.dumps(q))
        skeleton["messages"][0]["content"][1]["image_url"]["url"] = \
            "<data:image/png;base64, PNG bytes of the pdftoppm render>"

contract = {
    "captured_at_runtime": True,
    "olmocr_version": olmocr.version.VERSION,
    "olmocr_builder": "olmocr.pipeline.build_page_query (imported, not reimplemented)",
    "prompt_builder": "olmocr.prompts.build_no_anchoring_v4_yaml_prompt",
    "prompt": PROMPT,
    "prompt_sha256": prompt_sha,
    "prompt_bytes": len(PROMPT.encode()),
    "request_skeleton": skeleton,
    "message_structure": "one user message; content[0] = text prompt, content[1] = "
                         "image_url data URI; no system message in the request "
                         "(the chat template injects 'You are a helpful assistant.')",
    "max_tokens": 8000,
    "temperature_first_attempt": TEMPERATURE_BY_ATTEMPT[0],
    "temperature_by_attempt": TEMPERATURE_BY_ATTEMPT,
    "temperature_in_skeleton": 0.0,
    "temperature_note": "build_page_query sets 0.0 and try_single_page then "
                        "overwrites it with TEMPERATURE_BY_ATTEMPT[attempt]; "
                        "attempt 0 therefore runs at 0.1, i.e. sampling.",
    "target_longest_image_dim": TARGET_LONGEST,
    "render": "pdftoppm -png -r (target*72/longest_mediabox_point) via "
              "olmocr.data.renderpdf.render_pdf_to_base64png",
    "max_model_len": 16384,
    "max_page_retries_default": 8,
    "guided_decoding_default": False,
    "img2pdf": img2pdf_ver,
    "pdftoppm": pdftoppm_ver,
    "img2pdf_recipe": "img2pdf --output <name>.pdf <fixture image>",
    "renders": renders,
}
(WORK / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
print(json.dumps({k: v for k, v in contract.items() if k != "renders"}, indent=2))
print(f"render dims: {[r['rendered_wh'] for r in renders]}")

print("=" * 70)
print("Step 4: load the checkpoint")
print("=" * 70)
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,  # noqa: E402
                          StoppingCriteria)
from huggingface_hub import model_info  # noqa: E402

revision = model_info(MODEL).sha
print(f"model={MODEL} revision={revision}")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
# `torch_dtype` rather than the newer `dtype` alias: transformers is pinned to
# the version olmocr[gpu] pins, and the alias is not in every 4.5x release.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=getattr(torch, dtype), device_map="auto",
    attn_implementation="sdpa")
model.eval()
load_s = time.time() - t0
print(f"loaded in {load_s:.1f}s")
placements = sorted({str(v) for v in (model.hf_device_map or {}).values()})
print(f"placements={placements}")
for i in range(n_gpu):
    print(f"  cuda:{i} reserved={torch.cuda.memory_reserved(i) / 2**30:.2f} GiB "
          f"allocated={torch.cuda.memory_allocated(i) / 2**30:.2f} GiB")

# The render is not what the vision tower sees: Qwen2.5-VL's smart_resize snaps
# each side to a multiple of 28 (patch 14 x merge 2), so a 1288x904 render is fed
# as 1288x896.  The native lane emulates the 1288 render, so record the second
# resize too - it is the dimension the model actually consumes.
_ip = processor.image_processor
for r in renders:
    corpus, name = r["corpus"], r["fixture"]
    q = page_query(pdf_of[(corpus, name)])
    grid = _ip(images=[query_image(q)], return_tensors="pt")["image_grid_thw"][0].tolist()
    r["image_grid_thw"] = [int(x) for x in grid]
    r["model_input_wh"] = [int(grid[2]) * 14, int(grid[1]) * 14]
contract["renders"] = renders
contract["image_processor"] = type(_ip).__name__
contract["image_processor_note"] = (
    f"transformers {transformers.__version__} loads the FAST image processor by "
    "default even though the checkpoint saved a slow one; left at the default "
    "because that is what the toolkit's own serving stack gets")
contract["preprocessor_config"] = {
    k: getattr(_ip, k, None) for k in
    ("min_pixels", "max_pixels", "patch_size", "merge_size", "image_mean", "image_std")}
contract["chat_prompt_applied"] = processor.apply_chat_template(
    page_query(next(iter(pdf_of.values())))["messages"],
    tokenize=False, add_generation_prompt=True)
contract["chat_template_note"] = (
    "the request carries no system message; the checkpoint's chat template "
    "injects '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>' and "
    "renders content in order, so the instruction text precedes "
    "<|vision_start|><|image_pad|><|vision_end|>")
(WORK / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
print(f"model_input_wh: {[(r['fixture'], r['rendered_wh'], r['model_input_wh']) for r in renders]}")

eos_ids = {model.generation_config.eos_token_id} if isinstance(
    model.generation_config.eos_token_id, int) else set(model.generation_config.eos_token_id or [])
eos_ids |= {processor.tokenizer.eos_token_id}
im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
if im_end is not None:
    eos_ids.add(im_end)
print(f"eos ids={sorted(eos_ids)}")

parser = FrontMatterParser(front_matter_class=PageResponse)


class _Deadline(StoppingCriteria):
    def __init__(self, limit_s):
        self.until = time.time() + limit_s
        self.fired = False

    def __call__(self, input_ids, scores, **kw):
        if time.time() > self.until:
            self.fired = True
            return True
        return False


def generate(q: dict, temperature: float | None):
    """Run one request.  temperature None => greedy."""
    msgs = q["messages"]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = query_image(q)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    n_in = int(inputs["input_ids"].shape[-1])
    dl = _Deadline(PAGE_TIME_LIMIT_S)
    kw = dict(max_new_tokens=q["max_tokens"], stopping_criteria=[dl])
    if temperature is None:
        kw.update(do_sample=False)
    else:
        # vLLM's defaults for the fields the toolkit does not set.
        kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    gen_s = time.time() - t
    new = out[0][n_in:]
    n_out = int(new.shape[-1])
    raw = processor.tokenizer.decode(new, skip_special_tokens=True)
    stopped = bool(new.numel() and int(new[-1]) in eos_ids)
    finish = "stop" if stopped else ("length" if n_out >= q["max_tokens"] else "cutoff")
    return {
        "raw": raw, "prompt_tokens": n_in, "completion_tokens": n_out,
        "total_tokens": n_in + n_out, "finish_reason": finish,
        "gen_s": round(gen_s, 2), "deadline_fired": dl.fired,
        "chat_prompt_head": text[:220], "image_wh": list(img.size),
        "processor_grid_thw": [int(x) for x in inputs["image_grid_thw"][0].tolist()]
        if "image_grid_thw" in inputs else None,
    }


def evaluate(res: dict):
    """try_single_page's validity rules, applied to a local generation."""
    valid = True
    if res["total_tokens"] > contract["max_model_len"]:
        valid = False
    if res["finish_reason"] != "stop":
        valid = False
    page = None
    try:
        fm, text = parser._extract_front_matter_and_text(res["raw"])
        page = parser._parse_front_matter(fm, text)
    except Exception as e:
        valid = False
        return valid, None, f"{type(e).__name__}: {e}"
    return valid, page, None


def run_page(pdf: Path, sampled_only: bool):
    """The pipeline's retry ladder: attempt 0, then TEMPERATURE_BY_ATTEMPT."""
    attempts = []
    rotation = 0
    for attempt in range(MAX_ATTEMPTS):
        temp = TEMPERATURE_BY_ATTEMPT[min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)]
        # Attempt 0 of the gold pass is greedy on purpose: gold must be
        # reproducible, and the toolkit's own 0.1 is sampling.  The faithful
        # 0.1 attempt-0 is measured separately (sampled pass).
        use_temp = temp if (sampled_only or attempt > 0) else None
        q = page_query(pdf, rotation)
        res = generate(q, use_temp)
        valid, page, err = evaluate(res)
        rec = {"attempt": attempt,
               "temperature": use_temp, "decoding": "greedy" if use_temp is None else "sample",
               "valid": valid, "parse_error": err,
               **{k: v for k, v in res.items() if k != "raw"}}
        if page is not None:
            rec["front_matter"] = {
                "primary_language": page.primary_language,
                "is_rotation_valid": page.is_rotation_valid,
                "rotation_correction": page.rotation_correction,
                "is_table": page.is_table, "is_diagram": page.is_diagram,
            }
        rec["rotation_in"] = rotation
        attempts.append(rec)
        print(f"    attempt={attempt} temp={use_temp} valid={valid} "
              f"finish={res['finish_reason']} out_tok={res['completion_tokens']} "
              f"{res['gen_s']}s", flush=True)
        if valid and page is not None and page.is_rotation_valid:
            return res["raw"], page, attempts
        if page is not None and not page.is_rotation_valid:
            rotation = (rotation + (page.rotation_correction or 0)) % 360
    return res["raw"], page, attempts


def do_pass(tag: str, out_root: Path, sampled_only: bool):
    out_root.mkdir(parents=True, exist_ok=True)
    all_pages = {}
    for corpus, fx_list in fixtures.items():
        d = out_root / corpus
        d.mkdir(parents=True, exist_ok=True)
        pages = []
        for fx in fx_list:
            print(f"  [{tag}/{corpus}] {fx['name']}", flush=True)
            raw, page, attempts = run_page(pdf_of[(corpus, fx["name"])], sampled_only)
            stem = Path(fx["name"]).stem
            (d / f"{stem}.raw.txt").write_text(raw)
            nat = (page.natural_text if page is not None and page.natural_text else "")
            (d / f"{stem}.txt").write_text(nat)
            pages.append({
                "fixture": fx["name"], "stem": stem,
                "attempts": attempts, "n_attempts": len(attempts),
                "final_attempt": attempts[-1]["attempt"],
                "final_temperature": attempts[-1]["temperature"],
                "deterministic": len(attempts) == 1 and attempts[0]["temperature"] is None,
                "gen_s": sum(a["gen_s"] for a in attempts),
                "natural_text_chars": len(nat),
                "raw_chars": len(raw),
            })
            # Checkpoint after every page: a later crash must not cost gold
            # that the GPU already paid for.
            (d / "pages.json").write_text(json.dumps(pages, indent=2) + "\n")
        all_pages[corpus] = pages
        manifest = {
            "brief": "A3 - parity arm + gold: olmOCR toolkit",
            "pass": tag,
            "model_id": MODEL, "revision": revision,
            "toolkit_default_model": TOOLKIT_DEFAULT_MODEL,
            "olmocr_version": olmocr.version.VERSION,
            "prompt": PROMPT, "prompt_sha256": prompt_sha,
            "prompt_builder": contract["prompt_builder"],
            "message_structure": contract["message_structure"],
            "render": contract["render"],
            "target_longest_image_dim": TARGET_LONGEST,
            "max_tokens": 8000,
            "temperature_by_attempt": TEMPERATURE_BY_ATTEMPT,
            "dtype": dtype,
            "serving_stack": f"transformers {transformers.__version__} "
                             f"(torch {torch.__version__}), device_map=auto, sdpa",
            "serving_deviation": "the toolkit serves through vLLM; see vllm_probe.json",
            "hardware": hardware, "placements": placements,
            "date": time.strftime("%Y-%m-%d"),
            "images": str(CORPORA[corpus]),
            "repo_commit": repo_commit,
            "img2pdf": img2pdf_ver, "pdftoppm": pdftoppm_ver,
            "img2pdf_recipe": contract["img2pdf_recipe"],
            "renders": [r for r in renders if r["corpus"] == corpus],
            "pages": pages,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return all_pages


print("=" * 70)
print("Step 5: gold pass (greedy attempt 0, toolkit retry ladder on failure)")
print("=" * 70)
gold_pages = do_pass("gold", GOLD, sampled_only=False)

print("=" * 70)
print("Step 6: contract-faithful pass (attempt 0 at temperature 0.1)")
print("=" * 70)
try:
    sampled_pages = do_pass("sampled-t0.1", SAMPLED, sampled_only=True)
except Exception:
    traceback.print_exc()
    sampled_pages = {}

summary = {
    "model": MODEL, "revision": revision, "dtype": dtype, "hardware": hardware,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "olmocr": olmocr.version.VERSION, "branch": BRANCH, "repo_commit": repo_commit,
    "load_s": round(load_s, 1), "placements": placements,
    "gold": {c: [{"fixture": p["fixture"], "n_attempts": p["n_attempts"],
                  "final_temperature": p["final_temperature"],
                  "gen_s": p["gen_s"], "chars": p["natural_text_chars"]}
                 for p in ps] for c, ps in gold_pages.items()},
}
(WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

print("=" * 70)
print("Step 7: vLLM-on-Turing probe (subprocess; gold is already on disk)")
print("=" * 70)
# Everything above is written.  Installing vLLM rewrites torch, so it runs in a
# subprocess that cannot take the main process down, and its result is a
# recorded measurement either way.
del model
torch.cuda.empty_cache()
probe = WORK / "vllm_probe.json"
probe_src = Path("/tmp/vllm_probe.py")
probe_src.write_text(f'''
import json, subprocess, sys, time, traceback
out = {{"pin": "vllm==0.11.2 (olmocr[gpu])", "model": {MODEL!r}}}
t0 = time.time()
p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.11.2"],
                   capture_output=True, text=True)
out["install_rc"] = p.returncode
out["install_s"] = round(time.time() - t0, 1)
out["install_tail"] = (p.stderr or p.stdout)[-2000:]
if p.returncode == 0:
    try:
        import torch
        out["torch_after_install"] = torch.__version__
        out["arch_list"] = torch.cuda.get_arch_list()
        out["caps"] = [torch.cuda.get_device_capability(i)
                       for i in range(torch.cuda.device_count())]
        import vllm
        out["vllm"] = vllm.__version__
        t1 = time.time()
        # The parent process still owns a CUDA context on both cards, so the
        # utilisation budget is deliberately below what a clean box would take.
        llm = vllm.LLM(model={MODEL!r}, dtype="half",
                       tensor_parallel_size=torch.cuda.device_count(),
                       max_model_len=16384, gpu_memory_utilization=0.85,
                       limit_mm_per_prompt={{"image": 1}}, enforce_eager=True)
        out["init_s"] = round(time.time() - t1, 1)
        out["init"] = "ok"
        del llm
    except BaseException as e:
        out["init"] = "failed"
        out["error"] = f"{{type(e).__name__}}: {{e}}"[:4000]
        out["traceback"] = traceback.format_exc()[-6000:]
json.dump(out, open({str(probe)!r}, "w"), indent=2)
''')
try:
    subprocess.run([sys.executable, str(probe_src)], timeout=3600, check=False)
except subprocess.TimeoutExpired:
    probe.write_text(json.dumps(
        {"install_rc": None, "init": "timeout",
         "error": "vLLM install+init exceeded 3600 s"}, indent=2) + "\n")
if probe.exists():
    print(probe.read_text()[:4000])

shutil.rmtree(REPO, ignore_errors=True)
print("done")
