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
to it never reaches the kernel. Every CrispEmbed build kernel under
`tools/kaggle/` therefore carries a small `# ce-ccache-bootstrap` block just
before its first CMake configure. It loads `kaggle_harness.py` from the
CrispEmbed clone it just made (current directory, `/kaggle/working/*/` or
`/kaggle/temp/*/`) and calls `ensure_build_cache()`. That function is idempotent.
It installs the toolchain, warms from the seed and exports
`CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER=ccache`, so kernels whose own CMake lines
carry no launcher flags also use the cache. It also registers the automatic
export described below. Each build kernel's metadata attaches
`${KAGGLE_ACCOUNT}/crispembed-ccache`. Kernels pinned to a commit older than this
mechanism print `ce-ccache: bootstrap skipped` and build as before. The frozen
guard replay kernels are excluded, because their sources are hash-pinned.
`crispembed-ccache-seed` imports the harness directly.

## Automatic refresh

At interpreter exit (or when a kernel calls `kh.maybe_export_ccache()` itself),
the harness decides whether the seed is stale. It writes the decision to
`/kaggle/working/ccache-export.json`. Only when the seed is stale does it also
write one `ccache.tar`, trimmed with `ccache -c` to the size cap. Stale means:

| condition | default | override |
|---|---|---|
| hit rate since warm-up below | 0.95 | `CE_CCACHE_MIN_HIT` |
| seed older than (commits, via `git rev-list` in the clone) | 150 | `CE_CCACHE_MAX_AGE_COMMITS` |
| seed older than (days) | 21 | `CE_CCACHE_MAX_AGE_DAYS` |
| seed has no `crispembed-seed.json` (cold or legacy seed) | always stale | - |

Fewer than 20 compilations count as no evidence, and the seed is kept.
`CE_CCACHE_EXPORT=always|never` forces the decision. `CE_CCACHE_MAXSIZE`
(default `3G`) caps the cache. The tar holds `.ccache/crispembed-seed.json`
(source SHA, time, reasons, CUDA arch), which the next warm-up reads.
CPU and CUDA objects share one cache and one tar. The seed is copied in whole
before a build, so a CPU-only kernel that refreshes it keeps the CUDA entries.
CUDA entries hit only for the same arch: the seed kernel builds for a T4 (75).

Versioning the dataset happens on the pushing machine (no GitHub workflow):

```bash
set -a; . <private env file>; set +a        # KAGGLE_TOKEN
export KAGGLE_ACCOUNT=<account>
python3 tools/kaggle/kpush.py tools/kaggle/<kernel> --refresh-ccache [--sha <40-hex>]
# or, for a kernel pushed earlier:
python3 tools/kaggle/ccache_refresh.py --from-kernel <slug or owner/slug> [--wait]
```

`--refresh-ccache` waits for the pushed kernel to finish. It then runs
`ccache_refresh.py --from-kernel <id>`, which downloads only
`ccache-export.json`, `ccache.tar` and `summary.json`. A warm kernel exits 0 and
leaves the dataset unchanged. If the kernel exported a tar, the script refuses a
tar under 50 MB. Otherwise it creates the private dataset or adds a version.
`--sha` fills the `__SHA__` placeholder (the seed kernel's clone pin).

## Seed kernel

`tools/kaggle/crispembed-ccache-seed` (GPU worker) builds a portable CPU tree and
a CUDA tree from an exact SHA. It passes their combined hit counts to
`maybe_export_ccache()`, and `summary.json` records per-build wall time and
hits/misses. For a first cold run, remove the dataset from `dataset_sources`.
Push it with `--sha <commit> --refresh-ccache` after large `ggml` or CMake flag
changes. Any other build kernel pushed with `--refresh-ccache` keeps the seed
current as well. Stale entries only cost misses.
