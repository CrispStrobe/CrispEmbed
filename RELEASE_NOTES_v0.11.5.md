# CrispEmbed v0.11.5

Completes v0.11.4 (which shipped no Windows bundle) + the high-level KIE binding.

## 🛠 Fixes
- **Windows build** (`windows-x86_64` + vulkan + cuda were failing → no Windows
  asset in v0.11.4): `swinir_sr.cpp` used `M_PI` without a guard. Fixed, and —
  to stop this recurring with every new SR/restoration file — `_USE_MATH_DEFINES`
  is now defined globally for MSVC in CMake.

## ✨ Added
- **High-level KIE Rust binding** (`CrispKie`) — image → structured fields,
  with `new_lilt` for **LiLT layout-aware** extraction (`crispembed_kie_init_lilt`
  wires the pipeline's Phase-2 LiLT path through the C API).
- **SCUNet** denoising (Swin-Conv-UNet) + **SwinIR** SR engines.

## 📦 Consuming
- CrispSorter: bump `CRISPEMBED_REF` → `v0.11.5` for `CrispKie` (LiLT KIE),
  `CrispScunet` (restore engine), `tps_auto_dewarp` (TPS dewarp).
