# Kaggle ccache seed

Kaggle script kernels start from an empty disk, so every CrispEmbed build is
cold unless a compiler cache is attached as a dataset. This page covers how the
seed is produced, consumed and refreshed. Account names never appear in the
repository: `${KAGGLE_ACCOUNT}` is rendered by `tools/kaggle/kpush.py`.

## Consumer side

`kaggle_harness.install_build_toolchain()` (repository root) installs
ninja/ccache/mold, puts `CCACHE_DIR` on `/kaggle/temp`, and copies the seed
from any attached dataset that holds `.ccache/` or `ccache.tar` (owner-agnostic
scan; `crispembed-ccache` is probed first). It sets:

| variable | value | why |
|---|---|---|
| `CCACHE_BASEDIR` | `/kaggle` | absolute `-I`/source paths become relative to the build dir, so a seed made in `/kaggle/temp/<a>` hits in `/kaggle/working/<b>` |
| `CCACHE_NOHASHDIR` | `1` | the build directory path is not hashed |
| `CCACHE_COMPILERCHECK` | `content` | Kaggle image refreshes touch compiler mtimes; `mtime` (the default) would miss every object |
| `CCACHE_SLOPPINESS` | `include_file_mtime,include_file_ctime,time_macros` | a fresh clone gives all headers a new mtime |

Hits need the same relative layout (`<clone>/<build-dir>`, any names at the same
depth), the same compiler flags and the same CUDA arch (`detect_cuda_arch()`).
`cache_and_link_flags()` adds the C, C++ and CUDA compiler launchers.

Kaggle uploads only the kernel's `code_file`; a `kaggle_harness.py` placed next
to it never reaches the kernel. Kernels should import the harness from their
pinned clone (`sys.path.insert(0, <clone>)`), as `crispembed-ccache-seed` and
`ovisocr2-speed-ab` do, so the settings above apply.

## Producing and refreshing the seed

1. Render and push `tools/kaggle/crispembed-ccache-seed` with `__SHA__` set to a
   full commit SHA (GPU kernel: it builds a portable CPU tree and a CUDA tree).
   Drop the dataset from `dataset_sources` for the very first (cold) run.
2. When it completes: `python3 tools/kaggle/ccache_refresh.py`. It downloads only
   `ccache.tar` + `summary.json`, refuses a near-empty tar, and creates
   `${KAGGLE_ACCOUNT}/crispembed-ccache` (private) or adds a version.
3. Attach `${KAGGLE_ACCOUNT}/crispembed-ccache` in other kernels' metadata.

`summary.json` records per-build wall time and ccache hits/misses, so a seed run
on an attached dataset doubles as the warm-cache measurement. Refresh after
large `ggml` or CMake flag changes; stale entries only cost misses.
