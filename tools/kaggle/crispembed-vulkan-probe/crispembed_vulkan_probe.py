"""O10: Vulkan bring-up probe on the Kaggle GPU image.

Answers ONE question — does Vulkan compute work at all here — with a yes/no
and evidence. No optimization claims. Three stages, each reported even when
a later one fails:

  1. loader+ICD: apt vulkan-tools/libvulkan and `vulkaninfo --summary`
     (the NVIDIA driver must expose an ICD inside the container — the
     unknown this probe exists to resolve);
  2. build: the ggml fork with -DGGML_VULKAN=ON (needs glslc; apt
     glslc/shaderc) plus its test binaries;
  3. compute: ggml's own test-backend-ops on a few core ops, which
     cross-checks every result against the CPU backend — that IS the
     "one tiny graph computed and verified".

Everything lands in /kaggle/working/vkprobe.log.
"""
import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "vkprobe.log", "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "probe/vulkan-bringup")
EMBED_DIR = TEMP / "CrispEmbed"
GGML_DIR = EMBED_DIR / "ggml"
BUILD_DIR = GGML_DIR / "build-vk"

_CRISPASR = TEMP / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR)])
        sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


def sh_cap(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p


gpu = sh_cap("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)

# --- stage 1: loader + ICD ---
kh.step("vulkan.loader")
sh_cap("apt-get update -qq", timeout=300)
inst = sh_cap("apt-get install -y -qq vulkan-tools libvulkan-dev glslc 2>&1 | tail -2", timeout=600)
print(inst.stdout, flush=True)
if "glslc" in (inst.stdout + inst.stderr) and sh_cap("which glslc").returncode != 0:
    sh_cap("apt-get install -y -qq shaderc || pip install -q shaderc", timeout=600)
icd = sh_cap("ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null; ls /usr/lib/x86_64-linux-gnu/libvulkan* 2>/dev/null")
print(f"ICD/loader files:\n{icd.stdout}", flush=True)
vki = sh_cap("vulkaninfo --summary 2>&1 | head -40")
print(vki.stdout, flush=True)
has_device = "deviceName" in vki.stdout or "GPU id" in vki.stdout
note(f"stage1 loader+ICD: vulkaninfo_rc={vki.returncode} device_visible={has_device}")

# --- stage 2: build ggml with Vulkan ---
kh.step("clone+build")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
kh.install_build_toolchain()
glslc = sh_cap("which glslc")
note(f"stage2 glslc: {glslc.stdout.strip() or 'MISSING'}")
BUILD_DIR.mkdir(exist_ok=True)
cfg = sh_cap(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON "
             f"-DGGML_BUILD_TESTS=ON .. 2>&1 | tail -5", timeout=600)
print(cfg.stdout, flush=True)
built = False
if cfg.returncode == 0:
    with kh.build_heartbeat("ninja-vk", 30):
        b = sh_cap(f"cd {BUILD_DIR} && ninja -j4 test-backend-ops 2>&1 | tail -5", timeout=2400)
    print(b.stdout, flush=True)
    built = b.returncode == 0
note(f"stage2 build: configure_rc={cfg.returncode} test-backend-ops_built={built}")

# --- stage 3: compute with CPU cross-check ---
kh.step("compute")
if built:
    for op in ("MUL_MAT", "IM2COL", "SOFT_MAX", "NORM"):
        t = sh_cap(f"cd {BUILD_DIR} && ./bin/test-backend-ops test -b Vulkan0 -o {op} 2>&1 | tail -3",
                   timeout=1200)
        tail = " / ".join(t.stdout.strip().splitlines()[-2:])
        note(f"stage3 {op}: rc={t.returncode} {tail}")
else:
    note("stage3 compute: SKIPPED (no build)")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
