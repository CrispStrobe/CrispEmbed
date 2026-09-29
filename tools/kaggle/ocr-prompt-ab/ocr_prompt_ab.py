#!/usr/bin/env python3
"""Real Qwen3-VL decoded-output proof for CrispEmbed issue #56."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/ocr_prompt_ab")
EMBED = SCRATCH / "CrispEmbed"
ASR = Path("/tmp/ocr_prompt_harness/CrispASR")
BUILD = EMBED / "build"
RESULT = WORK / "ocr_prompt_ab.json"
BRANCH = "main"  # the fix merged; re-runs verify main
MODEL_URL = (
    "https://huggingface.co/cstr/qwen3-vl-2b-crispembed-gguf/resolve/main/"
    "qwen3-vl-2b-q4_k.gguf"
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(argv, *, check=True, capture=True, env=None):
    log("$ " + " ".join(map(str, argv)))
    result = subprocess.run(argv, text=True, capture_output=capture, env=env)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
    return result


SCRATCH.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
ASR.parent.mkdir(parents=True, exist_ok=True)
run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(ASR)])
sys.path.insert(0, str(ASR / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
hf_token = kh.resolve_hf_token(require=False)
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
kh.step("harness-ready", script_version="v5")
run([
    "timeout", "180", "git", "clone", "--depth", "1",
    "-b", BRANCH, "https://github.com/CrispStrobe/CrispEmbed.git", str(EMBED),
])
kh.step("repo-cloned")
run(["timeout", "180", "git", "-C", str(EMBED), "submodule", "update", "--init", "--depth", "1"])
commit = run(["git", "-C", str(EMBED), "rev-parse", "HEAD"]).stdout.strip()
log(f"CrispEmbed {commit}")
kh.step("repo-ready", commit=commit)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
unused_sibling = "/nonexistent/crispasr"
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags() + [
    f"-DCRISP_AUDIO_DIR={unused_sibling}/crisp_audio",
    f"-DCRISP_PUNC_DIR={unused_sibling}/crisp_punc",
    f"-DCRISP_LID_DIR={unused_sibling}/crisp_lid",
    f"-DCRISP_TRUECASE_DIR={unused_sibling}/crisp_truecase",
]
run(["cmake", "-S", str(EMBED), "-B", str(BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *flags])
with kh.build_heartbeat("ocr-prompt-build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispembed crispembed-server "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
kh.step("build-complete")
cli = BUILD / "crispembed"
server = BUILD / "crispembed-server"

model = SCRATCH / "qwen3-vl-2b-q4_k.gguf"
run(["curl", "-fL", "--retry", "4", "-o", str(model), MODEL_URL], capture=False)
run([sys.executable, "-m", "pip", "install", "-q", "Pillow"], capture=False)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

image = SCRATCH / "prompt.png"
canvas = Image.new("RGB", (1000, 420), "white")
draw = ImageDraw.Draw(canvas)
font = None
for font_path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
):
    try:
        font = ImageFont.truetype(font_path, 110)
        break
    except OSError:
        pass
if font is None:
    font = ImageFont.load_default(size=110)
draw.text((70, 45), "ALPHA", fill="black", font=font)
draw.text((70, 210), "OMEGA", fill="black", font=font)
canvas.save(image)

prompt_first = "Read the image and output only the first visible word, with no explanation."
prompt_second = "Read the image and output only the second visible word, with no explanation."


def cli_decode(prompt=None, pipeline=False):
    argv = [str(cli)]
    if pipeline:
        argv += ["--ocr-pipeline", str(image), "--ocr-engine", "qwen3vl", "--ocr-rec", str(model)]
    else:
        argv += ["-m", str(model), "--ocr", str(image)]
    argv += ["--ocr-max-tokens", "48"]
    if prompt:
        argv += ["--ocr-prompt", prompt]
    result = run(argv)
    return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "rc": result.returncode}


results = {"commit": commit, "model": MODEL_URL, "fixture": str(image)}
results["direct_default"] = cli_decode()
results["direct_first"] = cli_decode(prompt_first)
results["direct_second"] = cli_decode(prompt_second)
results["pipeline_first"] = cli_decode(prompt_first, pipeline=True)
results["pipeline_second"] = cli_decode(prompt_second, pipeline=True)

# One shared server context: prove both prompt and max-token overrides are
# request-scoped by returning to the baseline after each override.
proc = subprocess.Popen([
    str(server), "--ocr", str(model), "--ocr-max-tokens", "48",
    "--host", "127.0.0.1", "--port", "18086",
], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:18086/health", timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("server did not become ready")

    def post(body):
        req = urllib.request.Request(
            "http://127.0.0.1:18086/ocr/model",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)

    baseline_1 = post({"image": str(image)})
    custom = post({"image": str(image), "prompt": prompt_second})
    baseline_2 = post({"image": str(image)})
    one_token = post({"image": str(image), "max_tokens": 1})
    baseline_3 = post({"image": str(image)})
    results["server"] = {
        "baseline_1": baseline_1,
        "custom": custom,
        "baseline_2": baseline_2,
        "one_token": one_token,
        "baseline_3": baseline_3,
    }
finally:
    proc.terminate()
    try:
        _, server_stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, server_stderr = proc.communicate()
    results["server_stderr"] = server_stderr[-12000:]

direct_first = results["direct_first"]["stdout"].upper()
direct_second = results["direct_second"]["stdout"].upper()
pipeline_first = results["pipeline_first"]["stdout"].upper()
pipeline_second = results["pipeline_second"]["stdout"].upper()
server_result = results["server"]
checks = {
    "direct_outputs_differ": direct_first != direct_second,
    "pipeline_outputs_differ": pipeline_first != pipeline_second,
    "server_prompt_applied": server_result["custom"].get("prompt_applied") is True,
    "server_prompt_changes_output": server_result["custom"]["latex"] != server_result["baseline_1"]["latex"],
    "server_prompt_isolated": server_result["baseline_1"]["latex"] == server_result["baseline_2"]["latex"],
    "server_token_cap_isolated": server_result["baseline_2"]["latex"] == server_result["baseline_3"]["latex"],
    "server_one_token_shorter": server_result["one_token"]["len"] < server_result["baseline_3"]["len"],
}
results["checks"] = checks
results["passed"] = all(checks.values())
# The checks above prove prompt ROUTING (output changes with the prompt, overrides
# are request-scoped). They pass on garbage too: a model that emits scrambled
# letters still "differs". Report reading quality separately and loudly, so a
# routing pass is never mistaken for correct OCR. Not gating: it is a separate
# bug from issue #56 (seen 2026-09-28: default output "A L E P H A H ...").
default_text = results["direct_default"]["stdout"].upper()
results["quality"] = {
    "default_reads_alpha": "ALPHA" in default_text,
    "default_reads_omega": "OMEGA" in default_text,
    "first_prompt_says_alpha": "ALPHA" in direct_first,
    "second_prompt_says_omega": "OMEGA" in direct_second,
}
log("OCR reading quality (not gating): " + json.dumps(results["quality"]))
RESULT.write_text(json.dumps(results, indent=2))
log(json.dumps(checks, indent=2))
kh.step("verdict", passed=results["passed"], checks=checks)
if not results["passed"]:
    raise SystemExit("OCR prompt A/B failed; see ocr_prompt_ab.json")
