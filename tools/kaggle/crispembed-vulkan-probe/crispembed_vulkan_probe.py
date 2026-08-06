"""O10: Vulkan bring-up probe on the Kaggle GPU image — v2.

v1 postmortem (honesty first): every stage-2/3 "success" was rc-laundering
through `... 2>&1 | tail`, the HARD-RULE-8 trap — the apt install had
actually failed (`vulkaninfo: not found`) and cmake's FindVulkan errored.
v2 captures every rc directly, reports the package-install outcome
verbatim, tries a glslc fallback (shaderc prebuilt), probes/repairs the
NVIDIA ICD json, and — regardless of the Vulkan outcome — exports a proper
single-file ccache.tar so the (currently useless: wrong layout, 23 MB,
2026-06-21) chr1s4/crispembed-ccache dataset can finally be refreshed.

Stages, each reported even when a later one fails:
  1. loader+ICD: apt vulkan packages, vulkaninfo, nvidia_icd.json repair;
  2. build: ggml fork with -DGGML_VULKAN=ON + test-backend-ops;
  3. compute: test-backend-ops (CPU cross-checked) on a few core ops;
  4. ccache export: kh.export_ccache_tar() -> /kaggle/working/ccache.tar.

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


def run(cmd, timeout=600):
    """Direct rc, no pipes — v1's `| tail` laundered every failure to rc=0."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p


def show(p, n=6):
    lines = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
    for ln in lines[-n:]:
        print("  | " + ln, flush=True)


gpu = run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)

# --- stage 1: loader + ICD ---
kh.step("vulkan.loader")
apt = run("apt-get update -q && apt-get install -y --no-install-recommends "
          "vulkan-tools libvulkan-dev libvulkan1 glslc glslang-tools", timeout=900)
note(f"stage1 apt: rc={apt.returncode}")
show(apt, 8)
icd_dirs = run("ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null")
print(f"ICD dirs:\n{icd_dirs.stdout or '  (none)'}", flush=True)
if "nvidia" not in icd_dirs.stdout.lower():
    # Containers frequently ship the driver libs without the ICD manifest;
    # write the standard one and let vulkaninfo judge it.
    lib = run("ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null")
    Path("/usr/share/vulkan/icd.d").mkdir(parents=True, exist_ok=True)
    Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0",'
        '"api_version":"1.3.194"}}\n')
    note(f"stage1 icd-repair: wrote nvidia_icd.json (driver lib present={lib.returncode == 0})")
vki = run("vulkaninfo --summary", timeout=120)
show(vki, 14)
has_device = "deviceName" in (vki.stdout or "")
note(f"stage1 vulkaninfo: rc={vki.returncode} device_visible={has_device}")

# --- glslc fallback ---
if run("which glslc").returncode != 0:
    fb = run("curl -sL https://storage.googleapis.com/shaderc/artifacts/prod/"
             "graphics_shader_compiler/shaderc/linux/continuous_clang_release/"
             "latest/install.tgz -o /tmp/shaderc.tgz && "
             "tar -xzf /tmp/shaderc.tgz -C /tmp && "
             "cp /tmp/install/bin/glslc /usr/local/bin/glslc", timeout=600)
    note(f"stage1 glslc-fallback: rc={fb.returncode} which={run('which glslc').stdout.strip() or 'MISSING'}")
else:
    note(f"stage1 glslc: {run('which glslc').stdout.strip()}")

# --- stage 2: build ggml with Vulkan ---
kh.step("clone+build")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
kh.install_build_toolchain()
BUILD_DIR.mkdir(exist_ok=True)
cfg = run(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON "
          "-DGGML_BUILD_TESTS=ON " + " ".join(kh.cache_and_link_flags()) + " ..", timeout=600)
note(f"stage2 configure: rc={cfg.returncode}")
show(cfg, 6)
built = False
if cfg.returncode == 0:
    with kh.build_heartbeat("ninja-vk", 30):
        b = run(f"cd {BUILD_DIR} && ninja -j4 test-backend-ops", timeout=3600)
    note(f"stage2 ninja: rc={b.returncode}")
    show(b, 6)
    built = b.returncode == 0

# --- stage 3: compute with CPU cross-check ---
kh.step("compute")
if built:
    binq = run(f"find {BUILD_DIR} -name test-backend-ops -type f")
    tb = binq.stdout.strip().splitlines()
    note(f"stage3 binary: {tb[0] if tb else 'NOT FOUND'}")
    if tb:
        for op in ("MUL_MAT", "IM2COL", "SOFT_MAX", "NORM"):
            t = run(f"{tb[0]} test -o {op}", timeout=1800)
            tail = " / ".join((t.stdout or "").strip().splitlines()[-2:])
            note(f"stage3 {op}: rc={t.returncode} {tail[:160]}")
else:
    note("stage3 compute: SKIPPED (no build)")

# --- stage 4: ccache export for the dataset refresh ---
kh.step("ccache.export")
tar = kh.export_ccache_tar()
note(f"stage4 ccache export: {tar or 'nothing to export'}")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
