# Actual installed Linux wheel — 9 November 2026

[Packaging receipt](receipt.json): **40/40** checks pass.
[Installed real-PDF receipt](real-pdf/receipt.json): **51/51** checks pass.
The supervised native process exits successfully. The build uses the already
validated native source `891ea3d3632c6a3c3155e25b34705da6a02ec807`; source hashes
bind the Python packaging and probe changes executed from the worktree based on
`9c2ac509fbefa20a2069307daecf1ce678f9e69b`.

The earlier packaging proof treated its native file as opaque bytes. This
additional proof builds a platform wheel from a relocatable CMake installation,
reads back every CrispEmbed/GGML library and all three notices, installs that
wheel into a new consumer venv offline, and executes outside the checkout.
`LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH` and `PYTHONHOME` are removed. Python
runs in isolated mode. Automatic library discovery must select the installed
package, and `/proc/self/maps` must show that all four loaded native library
families come from that package.

The real-PDF probe exercises conversion, page labels and source maps, partial
checkpoints, cancellation, a fresh-context resume, changed-source rejection and
four-format processed-artifact reopen without usable PDFium. Every earlier
owned real-PDF check is retained, with three additional package-origin checks.
The venv inherits interpreter system packages for NumPy and pypdfium2; it does
not inherit a checkout or an installed CrispEmbed package.

| Identity | Value |
|---|---|
| Python | 3.12.14 |
| Build tooling | setuptools 84.2.1, wheel 0.48.0 |
| Dependencies | NumPy 2.3.5, pypdfium2 5.14.0 |
| Wheel | `crispembed-0.17.16-cp312-cp312-linux_x86_64.whl` |
| Wheel size | 19,404,409 bytes |
| Wheel SHA-256 | `ffc210022566e4f797d0674e7c7784076d6a599b6a5be6ddc19fa082795195d2` |
| Installed native SHA-256 | `6d3a813e08862ce280265310e2d1d1b8c039dee02d42415c271d3b287838b32a` |
| PDFium SHA-256 | `fd4de9f91f183476d04b1bc3fd17dacd95aba349c55d2c376512d95e082b441e` |

CMake installation changes the native RPATH to `$ORIGIN`, so its hash differs
from the previously recorded build-tree library. All three GGML dependencies
and their SONAME filenames are included and hashed. This provider version is
separate from the earlier pinned pypdfium2 5.10.1 lane and the frozen OCR corpus
protocol; neither lane's input pins were changed.

Reproduce with `tests/test_marker_installed_wheel.py --native-prefix PREFIX
--pdfium PDFIUM_LIBRARY --work-dir NEW_WORK_DIRECTORY --output NEW_REPORT_DIRECTORY`.
The input environment needs the documented Python build/runtime packages.
The driver does not download packages or model weights and does not publish.
This is a local Linux wheel, with no manylinux compatibility or release claim.
Its binary archive stays outside Git; exact artifact hashes, logs and decoded
owned-fixture outputs are retained here.
