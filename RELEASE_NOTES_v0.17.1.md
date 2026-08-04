# CrispEmbed v0.17.1

A packaging patch. **Every Linux user of every prior release should upgrade** —
the Linux archives could not start at all, on any machine that did not already
have OpenBLAS installed.

No engine, graph or model behaviour changes in this release.

## Linux archives were unlaunchable (SubtitleEdit#13205)

`crispembed-server` and `crispembed` exited with **code 127, printing nothing**,
on any Linux without OpenBLAS. 127 from a dynamically-linked binary is the
loader failing to resolve a library — the process dies before `main()`, so no
diagnostic of ours could ever appear. The chain was:

```
crispembed-server -> libggml.so.0      (bundled, RUNPATH $ORIGIN)
libggml.so.0      -> libggml-blas.so.0 (bundled)
libggml-blas.so.0 -> libopenblas.so.0  <-- never shipped in the archive
```

`-DGGML_BLAS=ON` on the Linux release legs made OpenBLAS a hard link-time
dependency the tarball did not carry. Both `crispembed-linux-x86_64.tar.gz` and
`crispembed-linux-arm64.tar.gz` were affected, in every release up to and
including v0.17.0. The build workflow apt-installed `libopenblas-dev` so the
CMake BLAS probe would succeed — so the runner always had the library and the
artifact never did, and the failure was unreachable from CI by construction.

It was also buying nothing: this repo's own `PERFORMANCE.md` "BLAS
Acceleration" table measures OpenBLAS at 0.9–1.0x on these models, because
quantized kernels use ggml's SIMD paths rather than BLAS. So the fix is to drop
the dependency rather than bundle it.

- Linux legs now build `-DGGML_BLAS=OFF`. LLAMAFILE's tinyBLAS stays on for
  x86_64 and needs no external library. macOS is unaffected (Accelerate is a
  system framework).
- `libopenblas-dev` is no longer installed on the Linux build runners, so a
  re-enabled `GGML_BLAS` now fails loudly at configure time instead of quietly
  producing a broken archive.
- New `scripts/check-bundled-deps.py` runs on the staged package of every Linux
  leg: it parses each ELF's dynamic section and fails the release if a
  `DT_NEEDED` entry is neither bundled nor part of a base glibc system.

The Linux archives now depend only on the base C/C++ runtime — `libc`, `libm`,
`libstdc++`, `libgcc_s` and the loader. Nothing to install.

## crispembed-sys could not link a prebuilt library without the sources

`build.rs` resolved the C/C++ source tree as the first statement of `main()`,
before it looked for a prebuilt library, and that resolution panics when the
sources are absent. Both documented escape hatches — `CRISPEMBED_SYS_LIB_DIR`
and the `build/` probe — were therefore unusable without a full checkout
including the ggml submodule, so a consumer holding a working
`libcrispembed.so` still needed ~1 GB of sources or the build failed with
*"crispembed sources not found"*. A regression against v0.16.1.

The prebuilt probe now runs first, and the sources are resolved only when a
cmake build is actually going to happen. A `CRISPEMBED_SYS_LIB_DIR` that is set
but holds no library emits a `cargo:warning` instead of silently falling
through to a source build.

## Build and CI robustness

- **Windows builds** — `-DNOMINMAX` (the `<windows.h>` `min`/`max` macros were
  mangling `std::min`/`std::max` and cascading into errors far from the cause)
  and `-DWIN32_LEAN_AND_MEAN` (legacy `<winsock.h>` colliding with
  `<winsock2.h>`, ~100 errors inside the Windows SDK). Several POSIX-only
  `setenv` / `mkstemp` calls replaced with portable equivalents.
- **Release dry runs** — `release.yml` now also accepts `workflow_dispatch`,
  with every publish step guarded on `refs/tags/`. The packaged artifacts can
  be built and inspected without cutting a tag, which is what made the
  OpenBLAS defect visible.
- **Model pin checks** — a HuggingFace rate limit was being reported as
  `model_hashes.h is stale`. Transient failures are now distinguished from real
  drift, and regenerating the header refuses to run while any repo is
  unreachable (it would have rewritten 117 good pins as unpinned).

## Known limitation

The Linux archives are built on Ubuntu 24.04 and require **glibc ≥ 2.38** and
**GLIBCXX ≥ 3.4.32**. They run on Ubuntu 24.04+, Debian 13+, RHEL/EL 9+ and
current Arch, but **not** on Ubuntu 22.04 or Debian 12 — build from source
there. Lowering this floor needs a manylinux-container build and is tracked
separately.

Full technical detail in `HISTORY.md` and `LEARNINGS.md`.
