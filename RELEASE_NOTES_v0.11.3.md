# CrispEmbed v0.11.3

Repairs the v0.11.2 release breakage (Windows build + Python wheels) caused by
the `feat/restormer` merge. All v0.11.2 features (SR/Restormer/SAFMN/Real-ESRGAN,
TPS dewarp, KIE/LiLT, table structure, GPU-for-all-engines) are intact.

## 🛠 Fixes
- **Windows build** (`windows-x86_64` + `-cuda` were failing → no Windows bundle
  published in v0.11.2): `text_sr.cpp` / `pan_sr.cpp` / `restormer.cpp` used
  `M_PI` without the `#ifndef M_PI` guard the other files carry — MSVC's
  `<cmath>` doesn't define it. Added the portable guard.
- **Python wheels** (all platforms failed with a `SyntaxError`): the merge
  duplicated the TBSRN block and tangled it into the SAFMN setup (unterminated
  docstring). Repaired `_binding.py`; exported the missing `CrispSafmnSr` +
  `CrispEsrganSr`.

## 📦 Consuming this release
- CrispSorter: bump `CRISPEMBED_REF` → `v0.11.3` (first **complete** post-SR
  release — v0.11.2 lacked the Windows bundle).
