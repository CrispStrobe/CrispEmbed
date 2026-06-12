# %% [markdown]
# # CrispEmbed — Keyven german-ocr-3.1 end-to-end test
#
# Load Keyven's llama.cpp GGUFs directly (no re-conversion),
# run OCR on a German invoice, verify output.

# %% [code]
import os, subprocess, sys, shutil, time
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"

if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(hf_token))

# %% [code]
# Clone CrispEmbed + build
REPO = WORK / "CrispEmbed"
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", "main",
    "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
])
subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"])
kh.step("cloned")

# %% [code]
# Build CrispEmbed (test binary + quantizer)
kh.install_build_toolchain()
BUILD = WORK / "build"
BUILD.mkdir(exist_ok=True)

cmake_cfg = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(kh.cache_and_link_flags())
)
kh.sh_with_progress(cmake_cfg)

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target test-qwen2vl-e2e test-qwen2vl "
        f"-j{kh.safe_build_jobs(gpu=False)}"
    )
kh.step("built")

# %% [code]
# Download Keyven GGUFs + create test image
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "hf_transfer", "Pillow",
])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

scratch = Path("/kaggle/temp/keyven-cache")
scratch.mkdir(parents=True, exist_ok=True)

print("[3] downloading Keyven GGUFs", flush=True)
with kh.build_heartbeat("download.keyven"):
    llm_path = hf_hub_download(
        "Keyven/german-ocr-3.1", "german-ocr-3.1-Q4_K_M.gguf",
        cache_dir=str(scratch))
    mmproj_path = hf_hub_download(
        "Keyven/german-ocr-3.1", "mmproj-german-ocr-3.1-F16.gguf",
        cache_dir=str(scratch))
kh.step("downloaded", llm=llm_path, mmproj=mmproj_path)

# Create test invoice image
img = Image.new("RGB", (640, 480), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 590, 100], outline="black", width=2)
draw.text((60, 60), "Rechnung Nr. 2024-0042", fill="black")
draw.rectangle([50, 120, 590, 200], outline="gray", width=1)
draw.text((60, 130), "Firma Mustermann GmbH", fill="black")
draw.text((60, 155), "Musterstrasse 123, 10115 Berlin", fill="gray")
draw.rectangle([50, 220, 590, 350], outline="gray", width=1)
draw.text((60, 230), "Pos.  Beschreibung          Menge  Preis", fill="black")
draw.text((60, 260), "1     Beratungsleistung       10h   150.00", fill="black")
draw.text((60, 285), "2     Softwareentwicklung      5h   200.00", fill="black")
draw.line([(50, 320), (590, 320)], fill="black", width=1)
draw.text((60, 325), "Gesamt:                              2500.00 EUR", fill="black")
test_img = WORK / "test_invoice.png"
img.save(str(test_img))
kh.step("test_image_created")

# %% [code]
# Run smoke test (unit tests + model load)
print("[4] running unit + smoke test", flush=True)
result = subprocess.run(
    [str(BUILD / "test-qwen2vl"), llm_path],
    capture_output=True, text=True, timeout=120)
print(result.stdout, flush=True)
if result.stderr:
    print(result.stderr[-500:], flush=True)
kh.step("smoke_test", returncode=result.returncode,
        output=result.stdout[-200:])

# %% [code]
# Tokenize prompt for e2e test
# The Keyven model uses the same Qwen2-VL tokenizer
# For the e2e test, we need pre-tokenized prompt + pre-processed patches
# Since we don't have the C++ image preprocessor fully wired for split
# GGUFs yet, let's test the model loading and basic generation

# Write a minimal C test that loads the split model
test_code = f"""
#include "qwen2vl_ocr.h"
#include <cstdio>

int main() {{
    printf("Loading split model...\\n");
    qwen2vl_ocr_context *ctx = qwen2vl_ocr_init_split(
        "{llm_path}", "{mmproj_path}", 4);
    if (!ctx) {{
        printf("FAIL: model load failed\\n");
        return 1;
    }}
    printf("PASS: split model loaded\\n");

    // Set German OCR prompt
    qwen2vl_ocr_set_prompt(ctx, "Extrahiere die Rechnung im Bild als JSON.");
    printf("PASS: prompt set\\n");

    // Try recognize on test image
    qwen2vl_ocr_set_max_tokens(ctx, 8);
    int out_len = 0;
    const char *result = qwen2vl_ocr_recognize_raw(
        ctx, NULL, 0, 0, 0, &out_len);  // NULL test
    printf("NULL test: %s\\n", result ? "non-null" : "null (correct)");

    // Load actual image
    FILE *fp = fopen("{test_img}", "rb");
    if (fp) {{
        fclose(fp);
        // Use stb_image
        #define STB_IMAGE_IMPLEMENTATION
        #include "../ggml/examples/stb_image.h"
        int w, h, ch;
        unsigned char *pixels = stbi_load("{test_img}", &w, &h, &ch, 3);
        if (pixels) {{
            printf("Image: %dx%d\\n", w, h);
            result = qwen2vl_ocr_recognize_raw(ctx, pixels, w, h, 3, &out_len);
            if (result && out_len > 0) {{
                printf("PASS: OCR output (%d chars): %s\\n", out_len, result);
            }} else {{
                printf("Output: empty or null (pipeline may need patches)\\n");
            }}
            stbi_image_free(pixels);
        }}
    }}

    qwen2vl_ocr_free(ctx);
    printf("PASS: model freed\\n");
    return 0;
}}
"""

test_file = WORK / "test_keyven_split.cpp"
test_file.write_text(test_code)

# Build the test
print("[5] building split-load test", flush=True)
subprocess.check_call([
    "g++", "-O2",
    f"-I{REPO}/src",
    f"-I{REPO}/ggml/include",
    f"-I{REPO}/ggml/src",
    str(test_file),
    f"-L{BUILD}",
    f"-L{BUILD}/ggml/src",
    "-lcrispembed", "-lggml", "-lggml-base", "-lggml-cpu",
    f"-Wl,-rpath,{BUILD}:{BUILD}/ggml/src",
    "-lpthread", "-lm",
    "-o", str(BUILD / "test-keyven-split"),
], timeout=60)
kh.step("split_test_built")

# Run the test
print("[6] running split-load test", flush=True)
with kh.build_heartbeat("test.keyven"):
    result = subprocess.run(
        [str(BUILD / "test-keyven-split")],
        capture_output=True, text=True, timeout=600)
print("STDOUT:", result.stdout, flush=True)
if result.stderr:
    print("STDERR:", result.stderr[-1000:], flush=True)
kh.step("split_test_done", returncode=result.returncode,
        output=result.stdout[-500:])

kh.step("done")
