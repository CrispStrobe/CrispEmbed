# CrispEmbed v0.17.4

A repair release for v0.17.3, which shipped **without either Linux CUDA
archive**. If you use CrispEmbed on Linux with an NVIDIA GPU, v0.17.3 has
nothing for you to download; this release does.

Everything else in v0.17.3 — the ggml 0.17.0 pin, the Flutter podspec fix, the
glibc 2.27 Linux floor, the olmOCR lane — is unchanged and carried forward.

## The Linux CUDA archives are back

The `linux-x86_64-cuda` build failed at the link step, after a full hour of
compiling:

```
libggml-cuda.so.0.17.0: undefined reference to `cuGetErrorString'
```

`cuGetErrorString` belongs to the CUDA *driver* API. ggml 0.17 calls it;
ggml 0.10.2 did not. So `libggml-cuda` now has to link `libcuda`, and a build
machine has no driver installed — which is exactly what the CUDA toolkit's
`lib64/stubs/libcuda.so` exists for. CMake only finds that stub when the stubs
directory is on the library path, and it was not.

The build now locates the stub and passes `-DCMAKE_LIBRARY_PATH`. The lookup
runs **before** the compile and fails immediately if the stub is absent, rather
than an hour later at link time — that hour is what turned this into a missing
artifact instead of a red build somebody noticed.

Linking the stub gives `libggml-cuda` a `DT_NEEDED` on `libcuda.so.1`, which is
the same driver contract both CUDA archives already declared. Nothing about
what a user needs installed has changed:

| archive | host must provide |
|---------|-------------------|
| `crispembed-linux-x86_64-cuda.tar.gz` | NVIDIA driver **and** CUDA 12.x toolkit runtime |
| `crispembed-linux-x86_64-cuda-bundled.tar.gz` | NVIDIA driver only |

Windows was never affected — `cuda.lib` sits in the toolkit's `lib/x64`, where
the linker finds it without help.

## Why v0.17.3 could not simply be re-run

The release workflow runs from the tagged commit, so a fix on `main` cannot
retroactively complete a published tag. v0.17.3 keeps its (incomplete) asset
set; this release supersedes it.

Full technical detail in `HISTORY.md` and `LEARNINGS.md`.
