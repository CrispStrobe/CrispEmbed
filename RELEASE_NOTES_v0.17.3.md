# CrispEmbed v0.17.3

**The ggml submodule moves from v0.10.2 to v0.17.0**, and the Flutter pod
stops pinning itself to a four-releases-old set of prebuilt libraries. Both
matter most to anyone embedding CrispEmbed next to CrispASR in one
application. Also here: Linux artifacts built against an older glibc, the
olmOCR engine, and a CUDA archive that actually carries NVIDIA's EULA.

## ggml: v0.10.2 → v0.17.0

CrispEmbed and CrispASR have always built on the same `CrispStrobe/ggml`
fork, but CrispEmbed tracked `crispstrobe-ops` (v0.10.2 + our ops) while
CrispASR moved on to `sync/upstream-v0.17` (v0.17.0). That divergence is
invisible until an application bundles both.

On macOS it stops being invisible. Both libraries ship `libggml*.0.dylib`
into one flat `Contents/Frameworks/`, and both reference them by the same
`@rpath` install names — so only one set can survive, and `libcrispembed`
compiled against v0.10.2 was being loaded against v0.17.0 at runtime. The
pin now names the same commit CrispASR records (`a0f7289d`), so there is one
ggml in the bundle and it is the one everything was built against.

This is a pin move, not a port: the CrispStrobe ops — `COL2IM_1D`,
`NORM_AFFINE`, `AA_SNAKE_BETA`, the siglu variants, and the WebGPU
`arange` / `pool2d` / `conv_transpose_2d` / NORM shaders — were already
forward-ported on the newer branch.

Nothing about embedding output changes. `LFM2.5-Embedding-350M-Q8_0` scores
identically to four decimals before and after (related pair 0.5812,
unrelated 0.0542), `test-core-cpu-ops` passes 172/172, and the Metal backend
smoke test still reports `correct=1`. All nine CI legs — macOS, iOS, Linux,
Windows, Android ×3, WASM, static guards — are green on the merge.

## The Flutter pod fetched the wrong release

`prepare_command` downloads the prebuilt natives from
`releases/download/v#{s.version}/`, so the podspec's version literal decides
which release every `pod install` pulls — and it had been left at `0.16.0`
while v0.17.0, v0.17.1 and v0.17.2 shipped. macOS and iOS consumers kept
getting the July libraries no matter which CrispEmbed ref they pinned, and
nothing failed loudly enough to say so.

The version is now read from the plugin's `pubspec.yaml`, which ships beside
the podspec on pub.dev and in a checkout alike, so it can no longer drift
from the release it points at.

Note when upgrading: `prepare_command` skips the download when `Libs/`
already contains dylibs, so an existing local drop must be cleared by hand
before the new libraries are fetched.

## Linux artifacts

Tarballs and wheels are now compiled inside `manylinux_2_28`, putting the
glibc floor at 2.28 (measured: the previous artifacts needed 2.38, the new
ones 2.27). The staged libraries also carry an `$ORIGIN` RUNPATH so
`auditwheel repair` can do its job instead of failing on them.

## Also in this release

- **olmOCR lane** — engine id 18, reachable as `--ocr-engine olmocr`, with
  `olmocr-2-7b` q4_k as the registry default (SHA-pinned). Front matter
  closed by a blank line instead of `---` is now tolerated rather than
  treated as a parse failure.
- **CUDA archive ships NVIDIA's EULA text**, and a failure to include it is
  no longer swallowed.
- Reference-parity work for DeepSeek-OCR / DeepSeek-OCR-2 (gold data,
  runtime contract capture, harness rows) landed alongside; it affects the
  test corpus rather than the shipped runtime.
