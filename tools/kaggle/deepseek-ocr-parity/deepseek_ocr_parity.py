#!/usr/bin/env python3
"""DeepSeek-OCR reference arm on a Kaggle GPU (brief A4).

Produces the reference transcripts and the request contract that the native
`deepseek_ocr2` lane needs and cannot produce for itself.

Nothing here re-implements the model's own entry point.  The checkpoint ships
`modeling_deepseekocr*.py` with an `infer()` helper, and that helper is what
runs: the conversation template, the `<image>` token accounting, the crop
geometry, the sampling parameters and the post-processing are whatever the
checkpoint's own code does on the day.  What this script adds is a recorder
wrapped around `model.generate`, so the prompt token count, the preprocessed
image tensor shapes, the generation kwargs and the raw output ids are captured
*from the call the checkpoint makes*, not from a source reading.

Two checkpoints run, because the brief names one and the native lane runs the
other:

  * `deepseek-ai/DeepSeek-OCR`   - the brief's reference (Gundam preset:
    base_size=1024, image_size=640, crop_mode=True).
  * `deepseek-ai/DeepSeek-OCR-2` - the checkpoint `src/deepseek_ocr2.cpp`
    actually loads, so the only one whose CER is a gate for T14
    (base_size=1024, image_size=768, crop_mode=True).

Two prompts run per checkpoint, both taken verbatim from the model cards:

  * plain OCR      `<image>\\nFree OCR. `
  * document/markdown `<image>\\n<|grounding|>Convert the document to markdown. `

Deviations from the card, all forced by the hardware and all recorded in
contract.json rather than papered over:

  * `_attn_implementation='flash_attention_2'` is impossible on Turing (sm_75).
    The checkpoint's `ATTENTION_CLASSES` offers only `eager` and
    `flash_attention_2` for the language model - there is no `sdpa` entry - so
    `eager` is not a preference, it is the only remaining key.  The vision
    towers never used flash attention (v1's ViT is built with
    `use_flash_attn=False`, v2's Qwen2 encoder with `attn_implementation="sdpa"`),
    so the substitution touches the decoder only.
  * bf16 is probed on the card rather than assumed.  If the probe fails the
    hardcoded `torch.bfloat16` in the checkpoint's `infer()` is rewritten to
    `torch.float16` and the exact patched lines are recorded.

Outputs under /kaggle/working: contract.json, summary.json, run.log, and
gold/<model>/<corpus>/{<stem>.<mode>.txt, <stem>.<mode>.mmd, manifest.json,
pages.json}.
"""
import contextlib
import io
import json
import os
import re
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
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/parity-deepseek")
REPO = Path("/kaggle/temp/CrispEmbed")
MODELS_DIR = Path("/tmp/dsmodels")
OUTDIR = Path("/tmp/dsout")
GOLD = WORK / "gold"

# The card's requirement block, verbatim.  torch is NOT touched (gotcha #11):
# Kaggle ships a working CUDA build and the card's torch==2.6.0 is only a
# "tested on" note, whereas transformers==4.46.3 is load-bearing - the remote
# code predates the GenerationMixin split and loses `.generate()` on 4.50+.
TRANSFORMERS_PIN = "4.46.3"
TOKENIZERS_PIN = "0.20.3"

# A degenerate repeat loop would spend the whole 8192-token budget; the budget
# itself stays at the checkpoint's value, but a page running longer than this
# is cut off and flagged rather than eating the kernel's wall clock.
PAGE_TIME_LIMIT_S = float(os.environ.get("DS_PAGE_TIME_LIMIT", "240"))
# 100 generations x a 240 s ceiling is 6.7 h in the worst case and Kaggle kills
# the session at 12 h with nothing retrievable.  Past this point the remaining
# pages are recorded as skipped instead of losing everything already produced.
GLOBAL_BUDGET_S = float(os.environ.get("DS_GLOBAL_BUDGET", "25200"))
RUN_T0 = time.time()

# base_size/image_size/crop_mode exactly as each model card writes them.
MODELS = [
    {
        "key": "deepseek-ocr",
        "id": "deepseek-ai/DeepSeek-OCR",
        "modeling_file": "modeling_deepseekocr.py",
        "infer_kwargs": {"base_size": 1024, "image_size": 640, "crop_mode": True,
                         "save_results": True, "test_compress": True},
        "card_preset": "Gundam (base_size=1024, image_size=640, crop_mode=True)",
    },
    {
        "key": "deepseek-ocr2",
        "id": "deepseek-ai/DeepSeek-OCR-2",
        "modeling_file": "modeling_deepseekocr2.py",
        "infer_kwargs": {"base_size": 1024, "image_size": 768, "crop_mode": True,
                         "save_results": True, "test_compress": False},
        "card_preset": "dynamic resolution (0-6)x768x768 + 1x1024x1024 "
                       "(base_size=1024, image_size=768, crop_mode=True)",
    },
]

# Verbatim from the "Usage" block of each card, trailing space included.  The
# checkpoint's own `format_messages` calls `.strip()` on the content, so the
# trailing space never reaches the tokenizer - recorded, not silently dropped.
PROMPTS = {
    "free_ocr": "<image>\nFree OCR. ",
    "grounding_markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
}
PRIMARY_MODE = "free_ocr"

os.environ["HF_HOME"] = "/tmp/hf"          # weights must not land in the 20 GB output mount
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find_dataset(*names):
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
sh(f"pip install -q transformers=={TRANSFORMERS_PIN} tokenizers=={TOKENIZERS_PIN} "
   f"einops addict easydict 2>&1 | tail -5", check=False)
sh("pip install -q huggingface_hub hf_transfer 2>&1 | tail -3", check=False)

import torch  # noqa: E402
import transformers  # noqa: E402

if transformers.__version__ != TRANSFORMERS_PIN:
    raise SystemExit(f"transformers is {transformers.__version__}, need {TRANSFORMERS_PIN}: "
                     "the checkpoint's remote code predates the GenerationMixin split "
                     "and silently loses .generate() on newer releases")

n_gpu = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(n_gpu)]
names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
vram = [torch.cuda.get_device_properties(i).total_memory / 2 ** 30 for i in range(n_gpu)]
arch_list = torch.cuda.get_arch_list()
hardware = f"Kaggle {n_gpu}x {names[0] if names else 'CPU'} ({sum(vram):.0f} GiB VRAM total)"
print(f"torch={torch.__version__} transformers={transformers.__version__}")
print(f"gpus={names} caps={caps} vram={[round(v, 1) for v in vram]}")
print(f"arch_list={arch_list}")

# `torch.cuda.is_available()` is True on a card the installed wheel has no SASS
# for; the failure surfaces only at the first kernel launch, ~90 s in.  Compare
# the card against the wheel's compiled arch list up front so a bad accelerator
# draw costs one line instead of a GPU-hour (a P100 draw died exactly this way
# on A2 - cu128 wheels ship no sm_60).
missing = [f"sm_{c[0]}{c[1]}" for c in caps if f"sm_{c[0]}{c[1]}" not in arch_list]
if missing:
    raise SystemExit(
        f"torch {torch.__version__} has no kernels for {missing} (compiled for "
        f"{arch_list}); re-push with machine_shape=NvidiaTeslaT4")
if not n_gpu:
    raise SystemExit("no GPU: infer() hardcodes .cuda()")


def probe_bf16():
    """Does this card actually run bf16, or does it only accept the dtype?

    cuBLAS exposes CUDA_R_16BF gemm from sm_80; on Turing the dtype is
    constructible and the matmul is not.  Probe the three shapes the checkpoint
    uses - gemm, conv (SAM stem) and SDPA (v2 encoder) - rather than trusting
    `is_bf16_supported()`, which has meant different things across releases.
    """
    out = {}
    for name, fn in (
        ("matmul", lambda: torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            @ torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)),
        ("conv2d", lambda: torch.nn.functional.conv2d(
            torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(8, 3, 3, 3, device="cuda", dtype=torch.bfloat16))),
        ("sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(
            *[torch.randn(1, 2, 32, 16, device="cuda", dtype=torch.bfloat16)] * 3)),
    ):
        try:
            r = fn()
            torch.cuda.synchronize()
            out[name] = "ok" if bool(torch.isfinite(r).all()) else "non-finite"
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"[:200]
    return out


bf16_probe = probe_bf16()
bf16_ok = all(v == "ok" for v in bf16_probe.values())
DTYPE = torch.bfloat16 if bf16_ok else torch.float16
DTYPE_NAME = "bfloat16" if bf16_ok else "float16"
print(f"bf16 probe: {bf16_probe} -> dtype={DTYPE_NAME}")

print("=" * 70)
print("Step 2: repo + fixtures")
print("=" * 70)
if not REPO.exists():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
sh("git log --oneline -1", cwd=REPO)
repo_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()

synth = find_dataset("crispembed-ocr-synth")
cc0 = REPO / "tests" / "regression" / "images" / "cc0"
if synth is None:
    raise SystemExit("crispembed-ocr-synth dataset not mounted")
print(f"synth fixtures: {synth}\ncc0 fixtures:   {cc0}")

sys.path.insert(0, str(REPO / "tests"))
from ocr_external_parity import load_fixtures  # noqa: E402

CORPORA = {"synth": synth, "cc0": cc0}
fixtures = {c: [f for f in load_fixtures(p) if f["truth"]] for c, p in CORPORA.items()}
for c, fx in fixtures.items():
    print(f"{c}: {len(fx)} labelled fixtures")

print("=" * 70)
print("Step 3: fetch checkpoints")
print("=" * 70)
# Both checkpoints are public, so the token is not an access requirement - it
# is only there so a shared-IP anonymous rate limit cannot fail the download
# half an hour in (kaggle_usage gotcha #26: attaching the dataset does nothing
# on its own, the file has to be read).
for _tok in Path("/kaggle/input").rglob("hf_token.txt") if Path("/kaggle/input").exists() else []:
    os.environ["HF_TOKEN"] = _tok.read_text().strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print(f"HF token from {_tok}")
    break

from huggingface_hub import model_info, snapshot_download  # noqa: E402
from transformers import AutoModel, AutoTokenizer, StoppingCriteria, StoppingCriteriaList  # noqa: E402

ALLOW = ["*.py", "*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)
for m in MODELS:
    m["revision"] = model_info(m["id"]).sha
    t0 = time.time()
    m["local"] = Path(snapshot_download(m["id"], local_dir=str(MODELS_DIR / m["key"]),
                                        allow_patterns=ALLOW))
    print(f"{m['id']} rev={m['revision']} -> {m['local']} in {time.time() - t0:.0f}s", flush=True)

    # The checkpoint hardcodes bf16 inside infer(): the preprocessed image
    # tensors are cast with `.to(torch.bfloat16)` and generation runs under
    # `torch.autocast("cuda", dtype=torch.bfloat16)`.  Neither is reachable
    # through an argument, so on a card without bf16 the file is rewritten and
    # every changed line is recorded.
    src = m["local"] / m["modeling_file"]
    text = src.read_text()
    if bf16_ok:
        m["dtype_patch"] = None
    else:
        changed = [(i + 1, ln.strip()) for i, ln in enumerate(text.split("\n"))
                   if "torch.bfloat16" in ln]
        src.write_text(text.replace("torch.bfloat16", "torch.float16"))
        m["dtype_patch"] = {
            "file": m["modeling_file"],
            "rule": "s/torch.bfloat16/torch.float16/g",
            "lines_changed": changed,
        }
        print(f"  patched {len(changed)} bf16 lines in {m['modeling_file']}", flush=True)


class _Deadline(StoppingCriteria):
    def __init__(self, limit_s):
        self.until = time.time() + limit_s
        self.fired = False

    def __call__(self, input_ids, scores, **kw):
        if time.time() > self.until:
            self.fired = True
            return True
        return False


STOP_STR = "<｜end▁of▁sentence｜>"
_COMPRESS_RE = {
    "image_size": re.compile(r"^image size:\s*(.*)$", re.M),
    "valid_img_tokens": re.compile(r"^valid image tokens:\s*(.*)$", re.M),
    "output_text_tokens": re.compile(r"^output texts tokens \(valid\):\s*(.*)$", re.M),
    "compression_ratio": re.compile(r"^compression ratio:\s*(.*)$", re.M),
}


def run_model(m):
    """Load one checkpoint and run every fixture x mode through its own infer()."""
    print("=" * 70)
    print(f"Step 4[{m['key']}]: load {m['id']}")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(str(m["local"]), trust_remote_code=True)
    t0 = time.time()
    # The card passes `_attn_implementation='flash_attention_2'`; sm_75 has no
    # FA2, and the checkpoint's ATTENTION_CLASSES has no sdpa entry, so `eager`
    # is the only remaining key rather than a choice between two.
    #
    # `torch_dtype` is the second forced change.  The card writes
    # `.eval().cuda().to(dtype)`, which materialises 3B parameters in fp32
    # (12 GiB) on the card *before* casting - 18 GiB peak on a 15 GiB T4.
    # Materialising at the target dtype gives the same numbers (the stored
    # weights are bf16, and bf16 -> fp32 -> fp16 equals bf16 -> fp16 because
    # the fp32 step is exact) with half the peak.
    model = AutoModel.from_pretrained(str(m["local"]), _attn_implementation="eager",
                                      torch_dtype=DTYPE,
                                      trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda()
    load_s = time.time() - t0
    # A bf16 weight above 65504 becomes inf in fp16 and the failure would read
    # as "the model is bad at this page".  Cheap to rule out, expensive to
    # bisect later.
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise SystemExit(f"non-finite parameters after {DTYPE_NAME} load: {bad[:5]}")
    attn_cfg = getattr(model.config, "_attn_implementation", None)
    attn_cls = type(model.model.layers[0].self_attn).__name__
    print(f"loaded in {load_s:.0f}s  _attn_implementation={attn_cfg}  "
          f"decoder attention class={attn_cls}")
    # Proof of work: the config field can be set and ignored; the instantiated
    # class is what runs.
    if "Flash" in attn_cls:
        raise SystemExit(f"decoder still on flash attention ({attn_cls}) on sm_75")
    gen_cfg = json.loads(model.generation_config.to_json_string())
    print(f"generation_config={gen_cfg}")
    print(f"vram reserved={torch.cuda.memory_reserved(0) / 2 ** 30:.2f} GiB")

    captured = {}
    _orig_generate = model.generate

    def recording_generate(*args, **kwargs):
        deadline = _Deadline(PAGE_TIME_LIMIT_S)
        images = kwargs.get("images")
        seq_mask = kwargs.get("images_seq_mask")
        crop = kwargs.get("images_spatial_crop")
        captured.clear()
        captured["input_ids"] = args[0] if args else kwargs.get("input_ids")
        captured["n_prompt_tokens"] = int(captured["input_ids"].shape[-1])
        captured["n_image_tokens"] = int(seq_mask.sum()) if seq_mask is not None else None
        captured["images_crop_shape"] = list(images[0][0].shape) if images else None
        captured["images_ori_shape"] = list(images[0][1].shape) if images else None
        captured["images_spatial_crop"] = crop.tolist() if crop is not None else None
        captured["generate_kwargs"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__)
            for k, v in kwargs.items()
            if k not in ("images", "images_seq_mask", "images_spatial_crop")}
        kwargs["stopping_criteria"] = StoppingCriteriaList([deadline])
        t = time.time()
        out = _orig_generate(*args, **kwargs)
        captured["gen_s"] = round(time.time() - t, 2)
        captured["deadline_fired"] = deadline.fired
        captured["output_ids"] = out
        return out

    model.generate = recording_generate

    def one_page(fx, mode, prompt, gd, stem, write=True):
        out_path = OUTDIR / m["key"] / mode / stem
        shutil.rmtree(out_path, ignore_errors=True)
        out_path.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        t0 = time.time()
        err = None
        # A failure before generate() would otherwise leave the PREVIOUS page's
        # output_ids in `captured` and silently write it out as this page's
        # transcript.
        captured.clear()
        try:
            # The card's own call, argument for argument.  Everything
            # informative is read back out of `captured`, so nothing about the
            # request is second-guessed here.
            with contextlib.redirect_stdout(buf):
                model.infer(tokenizer, prompt=prompt, image_file=str(fx["path"]),
                            output_path=str(out_path), **m["infer_kwargs"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        wall_s = round(time.time() - t0, 2)
        printed = buf.getvalue()

        raw = ""
        n_out = 0
        if captured.get("output_ids") is not None:
            new = captured["output_ids"][0, captured["n_prompt_tokens"]:]
            raw = tokenizer.decode(new, skip_special_tokens=False)
            if raw.endswith(STOP_STR):
                raw = raw[:-len(STOP_STR)]
            raw = raw.strip()
            n_out = int(new.shape[-1])
        mmd_chars = None
        if write:
            (gd / f"{stem}.{mode}.txt").write_text(raw)
            mmd = out_path / "result.mmd"
            if mmd.exists():
                shutil.copy(mmd, gd / f"{stem}.{mode}.mmd")
                mmd_chars = len((gd / f"{stem}.{mode}.mmd").read_text())
        entry = {
            "prompt": prompt,
            "error": err,
            "gen_s": captured.get("gen_s"),
            "wall_s": wall_s,
            "deadline_fired": captured.get("deadline_fired"),
            "n_prompt_tokens": captured.get("n_prompt_tokens"),
            "n_image_tokens": captured.get("n_image_tokens"),
            "n_output_tokens": n_out,
            "images_ori_shape": captured.get("images_ori_shape"),
            "images_crop_shape": captured.get("images_crop_shape"),
            "images_spatial_crop": captured.get("images_spatial_crop"),
            "generate_kwargs": captured.get("generate_kwargs"),
            "raw_chars": len(raw),
            "post_processed_chars": mmd_chars,
        }
        for k, rx in _COMPRESS_RE.items():
            hit = rx.search(printed)
            if hit:
                entry[k] = hit.group(1).strip()
        shutil.rmtree(out_path, ignore_errors=True)
        return entry, raw

    all_pages = {}
    skipped = []
    for corpus, fx_list in fixtures.items():
        gd = GOLD / m["key"] / corpus
        gd.mkdir(parents=True, exist_ok=True)
        pages = []
        for fx in fx_list:
            stem = Path(fx["name"]).stem
            rec = {"fixture": fx["name"], "stem": stem, "modes": {}}
            for mode, prompt in PROMPTS.items():
                if time.time() - RUN_T0 > GLOBAL_BUDGET_S:
                    skipped.append(f"{corpus}/{fx['name']}/{mode}")
                    continue
                entry, raw = one_page(fx, mode, prompt, gd, stem)
                rec["modes"][mode] = entry
                print(f"  [{m['key']}/{corpus}] {fx['name']} {mode}: "
                      f"{entry['n_output_tokens']} tok, {entry['gen_s']}s, "
                      f"{entry['raw_chars']} chars"
                      + (f" ERROR {entry['error']}" if entry["error"] else ""), flush=True)
            pages.append(rec)
            # Checkpoint after every page: a later crash must not cost gold the
            # GPU has already paid for.
            (gd / "pages.json").write_text(json.dumps(pages, indent=2) + "\n")

        manifest = {
            "brief": "A4 - parity arm + gold: HF DeepSeek-OCR reference",
            "model_id": m["id"],
            "revision": m["revision"],
            "card_preset": m["card_preset"],
            "infer_kwargs": m["infer_kwargs"],
            "prompts": PROMPTS,
            "primary_mode": PRIMARY_MODE,
            "entry_point": f"{m['modeling_file']}::infer (the checkpoint's own helper, "
                           "called with the model card's arguments)",
            "conversation_template": "plain (sft_format='plain', system_prompt=''); "
                                     "SeparatorStyle.PLAIN emits the user content alone, "
                                     "and format_messages .strip()s it, so the applied "
                                     "prompt is the card string minus its trailing space",
            "dtype": DTYPE_NAME,
            "dtype_deviation": (
                None if bf16_ok
                else "card runs bfloat16; sm_75 has no bf16 gemm, so float16 (see dtype_patch)"),
            "dtype_patch": m["dtype_patch"],
            "bf16_probe": bf16_probe,
            "attn_implementation": attn_cfg,
            "decoder_attention_class": attn_cls,
            "attn_deviation": "card passes _attn_implementation='flash_attention_2'; "
                              "sm_75 has no FA2 and the checkpoint's ATTENTION_CLASSES "
                              "has no sdpa entry, so eager is the only remaining key. "
                              "Vision towers are unaffected (v1 ViT use_flash_attn=False, "
                              "v2 Qwen2 encoder attn_implementation='sdpa')",
            "generation_config": gen_cfg,
            "page_time_limit_s": PAGE_TIME_LIMIT_S,
            "page_time_limit_note": "injected as a StoppingCriteria; max_new_tokens stays "
                                    "at the checkpoint's 8192",
            "serving_stack": f"transformers {transformers.__version__} "
                             f"(torch {torch.__version__}), single CUDA device",
            "hardware": hardware,
            "date": time.strftime("%Y-%m-%d"),
            "images": str(CORPORA[corpus]),
            "repo_commit": repo_commit,
            "load_s": round(load_s, 1),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "skipped_over_budget": skipped,
            "pages": pages,
        }
        (gd / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        all_pages[corpus] = pages

    # Is this gold reproducible?  The checkpoint passes temperature=0.0 with no
    # do_sample, so decoding should be greedy and a re-run byte-identical.
    # "Should be" is not a measurement: re-run two pages and compare, because
    # everything downstream will treat these files as fixed.
    determinism = []
    probe_fx = (fixtures["synth"][:1] + fixtures["cc0"][:1])
    for fx in probe_fx:
        stem = Path(fx["name"]).stem
        first = (GOLD / m["key"] / ("synth" if fx in fixtures["synth"] else "cc0")
                 / f"{stem}.{PRIMARY_MODE}.txt")
        if not first.exists():
            continue
        _, raw2 = one_page(fx, PRIMARY_MODE, PROMPTS[PRIMARY_MODE], None, stem, write=False)
        determinism.append({"fixture": fx["name"],
                            "identical": first.read_text() == raw2,
                            "chars": [len(first.read_text()), len(raw2)]})
    print(f"determinism re-run: {determinism}", flush=True)

    m["runtime"] = {"attn_implementation": attn_cfg, "decoder_attention_class": attn_cls,
                    "generation_config": gen_cfg, "load_s": round(load_s, 1),
                    "tokenizer_eos_token_id": tokenizer.eos_token_id,
                    "determinism_rerun": determinism, "skipped_over_budget": skipped}
    (GOLD / m["key"] / "runtime.json").write_text(
        json.dumps(m["runtime"], indent=2, default=str) + "\n")

    model = None
    torch.cuda.empty_cache()
    return all_pages


results = {}
for m in MODELS:
    try:
        results[m["key"]] = run_model(m)
    except Exception:
        traceback.print_exc()
        results[m["key"]] = {}

contract = {
    "captured_at_runtime": True,
    "brief": "A4 - parity arm + gold: HF DeepSeek-OCR reference",
    "entry_point": "the checkpoint's own modeling_deepseekocr*.py::infer(); this kernel "
                   "wraps model.generate to record the request it builds and does not "
                   "rebuild any part of it",
    "prompts": PROMPTS,
    "primary_mode": PRIMARY_MODE,
    "prompt_note": "verbatim from each model card's Usage block, trailing space included; "
                   "format_messages() .strip()s the content so the trailing space never "
                   "reaches the tokenizer",
    "conversation_template": "plain",
    "image_token": "<image>", "image_token_id": 128815,
    "bos_id": 0, "stop_str": STOP_STR,
    "dtype": DTYPE_NAME, "bf16_probe": bf16_probe,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "hardware": hardware, "arch_list": arch_list,
    "repo_commit": repo_commit,
    "date": time.strftime("%Y-%m-%d"),
    "models": [{k: v for k, v in m.items() if k != "local"} for m in MODELS],
}
(WORK / "contract.json").write_text(json.dumps(contract, indent=2, default=str) + "\n")

summary = {
    "contract": {k: v for k, v in contract.items() if k != "models"},
    "models": {},
}
for m in MODELS:
    summary["models"][m["key"]] = {
        "id": m["id"], "revision": m["revision"],
        "corpora": {c: [{"fixture": p["fixture"],
                         **{f"{mo}_chars": p["modes"][mo]["raw_chars"] for mo in PROMPTS},
                         **{f"{mo}_gen_s": p["modes"][mo]["gen_s"] for mo in PROMPTS}}
                        for p in ps]
                    for c, ps in results.get(m["key"], {}).items()},
    }
(WORK / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary["contract"], indent=2, default=str))

n_files = sum(1 for _ in WORK.rglob("*") if _.is_file())
print(f"/kaggle/working files: {n_files} (page cap is 500)")
shutil.rmtree(REPO, ignore_errors=True)
print("done")
