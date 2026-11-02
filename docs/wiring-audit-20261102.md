# Wiring audit — recently added engines (2 November 2026)

This audit checks the layers in the [contribution checklist](contributing.md)
for Unlimited-OCR-2, MiniCPM-OCR, MoEdernColBERT and Josev SystemOne. It is a
static source audit: nothing was built or run. Line numbers refer to the commit
that adds this file. Runtime gates are listed in
[environment-variables.md](environment-variables.md).

Legend: **wired**, **fixed** (added by this audit), **missing**, **N/A**
(not applicable to this kind of engine), **external** (lives in another repository
and was not verified here).

## Engines and aliases

| Engine | Registry aliases | Kind | Orchestrator id | Required gate(s) |
|---|---|---|---|---|
| Unlimited-OCR-2 | `unlimited-ocr2`, `unlimited-ocr2-q8` | whole-page VLM OCR | 13 (shared with Unlimited-OCR v1; variant from GGUF) | `CRISPEMBED_UNLIMITED_OCR2` |
| MiniCPM-OCR 1.0 | `minicpm-ocr`, `minicpm-ocr-q8` | whole-page VLM OCR | 23 | `CRISPEMBED_MINICPM_OCR` + `CRISPEMBED_QWEN35_OCR` |
| MoEdernColBERT | `moedern-1.5m`, `moedern-27m`, `moedern-115m` | multivector text retrieval | N/A | `CRISPEMBED_MOEDERN_*` (+ `_FAMILY` for 27M/115M) |
| Josev-2B SystemOne | `josev2b-systemone` | joint-schema text decisions | N/A | `CRISPEMBED_JOSEV_*` + `CRISPEMBED_QWEN35_OCR` |

## Engine × layer

| Layer | Unlimited-OCR-2 | MiniCPM-OCR | MoEdern | Josev SystemOne |
|---|---|---|---|---|
| Engine source | wired `src/unlimited_ocr.cpp` (variant 2) | wired `src/minicpm_ocr.cpp`, `src/minicpm_ocr_vision.cpp` | wired `src/moedern_*.cpp` | wired `src/josev2b_*.cpp`, `src/systemone_api.cpp` |
| C ABI | wired: arch detect `src/crispembed.cpp:4656` (`crispembed_ocr_model_*`) | wired: arch detect `src/crispembed.cpp:4651` | wired: separate public header `src/moedern_text_api.h:22` (installed with `crispembed.h`) | wired `src/crispembed.h:1597` |
| Orchestrator enum | wired `src/ocr_orchestrator.h:57` | wired `src/ocr_orchestrator.h:65` | N/A | N/A |
| `map_engine` | wired `src/crispembed.cpp:5826` | wired `src/crispembed.cpp:5846` | N/A | N/A |
| `ocr_stage.engine` header comment | wired `src/crispembed.h:1418` | wired `src/crispembed.h:1424` | N/A | N/A |
| Named stage constant (C) | **fixed** `src/crispembed.h:1405` | **fixed** `src/crispembed.h:1406` | N/A | N/A |
| `ocr_orchestrator.cpp` dispatch | wired `src/ocr_orchestrator.cpp:1951` | wired `src/ocr_orchestrator.cpp:1738` | N/A | N/A |
| VLM cap lists (`is_vlm_engine`, set_max_tokens) | wired (`tests/test_ocr_max_tokens_surfaces.py` passes) | wired (same test) | N/A | N/A |
| CMake | wired `CMakeLists.txt:376` | wired `CMakeLists.txt:390` | wired `CMakeLists.txt:424` | wired `CMakeLists.txt:355` |
| CLI | **fixed**: `--ocr-engine` accepted only `unlimited_ocr` while help and logs print `unlimited-ocr`; now also `unlimited-ocr`, `unlimited-ocr2`, `unlimited_ocr2` (`examples/cli/main.cpp:1432`), and the `unlimited-ocr2` alias defaults to the v2 registry model instead of v1 (`examples/cli/main.cpp:1588`) | **fixed** (follow-up): id 23 now defaults to the `minicpm-ocr` alias like the server; gates still required (`tests/test_cli_engine_names.py` checks both surfaces) | wired `examples/cli/main.cpp:10` | wired `examples/cli/main.cpp:115` |
| HTTP server | **fixed** the same aliases and v2 default (`examples/server/server.cpp:1578`, `:1690`) | wired `examples/server/server.cpp:1586` (id 23, default `minicpm-ocr`) | wired `examples/server/server.cpp:39` | wired `examples/server/server.cpp:6` (`POST /systemone`) |
| Model registry | wired `examples/cli/model_mgr.cpp:1896` | wired `examples/cli/model_mgr.cpp:1885` | wired `examples/cli/model_mgr.cpp:1641` | wired `examples/cli/model_mgr.cpp:1636` |
| SHA-256 pins | wired `examples/cli/model_hashes.h:579` (immutable revision) | wired `examples/cli/model_hashes.h:575` | wired `examples/cli/model_hashes.h:501` | wired `examples/cli/model_hashes.h:499` |
| Python | wired through `CrispOcrModel` / orchestrator stages; **fixed** `OCR_ENGINE_UNLIMITED_OCR` (`python/crispembed/_binding.py:4205`) | wired; **fixed** `OCR_ENGINE_MINICPM_OCR` (`python/crispembed/_binding.py:4206`) | wired `CrispMoEdern` (`python/crispembed/_moedern_text.py:16`) | wired `CrispSystemOne` (`python/crispembed/_systemone.py:7`) |
| Rust | wired (stage int); **fixed** `OCR_ENGINE_UNLIMITED_OCR` (`crispembed-sys/src/lib.rs:211`, `crispembed/src/lib.rs:2721`) | **fixed** `OCR_ENGINE_MINICPM_OCR` (`crispembed-sys/src/lib.rs:213`, `crispembed/src/lib.rs:2722`) | wired `MoEdern` (`crispembed/src/lib.rs:40`) | wired `SystemOne` (`crispembed/src/lib.rs:29`) |
| Dart / Flutter | **fixed** `crispOcrEngineUnlimitedOcr` (`flutter/crispembed/lib/src/crispembed.dart:4073`) | **fixed** `crispOcrEngineMinicpmOcr` (`flutter/crispembed/lib/src/crispembed.dart:4076`) | wired `flutter/crispembed/lib/crispembed.dart:9` | wired `flutter/crispembed/lib/crispembed.dart:17` |
| Env vars documented | wired (this audit) | wired (incl. PR #109 speed gates) | wired | wired |
| CrispSorter / CrispCalc catalog | CrispSorter: `unlimited_ocr2` → 13 selector in [CrispSorter PR #19](https://github.com/CrispStrobe/CrispSorter/pull/19) (experimental, gated); CrispCalc: N/A (see below) | CrispSorter: `minicpm_ocr` → 23 in the same PR; CrispCalc: N/A | N/A (not OCR) | N/A |

## Fixes made

1. Named orchestrator stage constants for ids 13 and 23 in C, Rust (sys and safe
   crate), Python and Dart. Before this, only TeleOCR, MinerU2.5-Pro and OXR
   had them, so callers had to hard-code the integers. No ABI change: the
   integers already existed in `map_engine`.
2. CLI and server `--ocr-engine` now accept `unlimited-ocr`, `unlimited-ocr2` and
   `unlimited_ocr2` in addition to `unlimited_ocr`. Both surfaces printed the
   hyphenated name in logs but rejected it as input, so it fell back to engine 0.
3. The `unlimited-ocr2` alias picks the `unlimited-ocr2` registry default when no
   explicit model is given. The plain `unlimited_ocr` name keeps the v1 default.
   The engine still refuses a variant-2 GGUF unless `CRISPEMBED_UNLIMITED_OCR2=1`.

## Follow-up (2 November 2026, same day) — gaps closed

- **CLI vs server default for MiniCPM-OCR: fixed.** The CLI now resolves the
  published `minicpm-ocr` alias when `--ocr-engine minicpm-ocr` has no
  `--ocr-rec`, matching the server. `CRISPEMBED_MINICPM_OCR=1` and
  `CRISPEMBED_QWEN35_OCR=1` are still required, and the orchestrator and engine
  enforce them too. `tests/test_cli_engine_names.py` fails if the two surfaces
  diverge again.
- **CrispSorter: PR open.** [CrispSorter PR #19](https://github.com/CrispStrobe/CrispSorter/pull/19)
  adds the `unlimited_ocr2` (→ 13, alias `unlimited-ocr2`) and `minicpm_ocr`
  (→ 23, alias `minicpm-ocr`) stage selectors and exposes the existing `unlimited_ocr`.
  Neither is a default. CrispEmbed refuses to initialise them without their gates.
  CrispSorter's CI must pass before merge.
- **CrispCalc: N/A, with a TODO.** CrispCalc has no orchestrator engine-id selector. Its
  `lib/engine/ocr_model_catalog.dart` is a download catalog of formula, layout,
  detection and VLM GGUFs, and the app is pinned to the immutable CrispEmbed 0.17.12
  runtime. If a catalog entry is wanted after a qualified release, add
  `OcrModelVariant`s under a new experimental group with the pinned URLs and hashes from
  `examples/cli/model_hashes.h` (`unlimited-ocr2` at its immutable revision, and
  `minicpm-ocr-1.0-f16.gguf` at `6705b065…`). Mark them desktop-only and experimental.
  Do this only after the app pin moves past a runtime that has ids 13-v2 and 23.
- **131 raw presence-style env reads: fixed.** All were converted:
  `core_env::on` for default-off switches and `NO_*`/`*_SCALAR`/`*_LEGACY` opt-outs, and
  the new `core_env::value` (empty = unset) for path, int and string variables
  (`CRISPEMBED_IMATRIX_OUT`, `CRISPEMBED_C2PA_*`, `LAYOUT_INJECT_*`,
  `MATH_OCR_DIFF_REF`, `CRISPEMBED_QWEN2VL_NO_REPEAT_NGRAM`,
  `CRISPEMBED_UNLIMITED_OCR2_OUTPUT`). `core_env::on`/`explicitly_off` now also read
  `false`/`off`/`no` (exact, case-insensitive) as off. No default changed. The
  inventory shows 0 `presence` reads, and `tests/test_env_vars_documented.py` fails on
  a new one, with a detector self-test and an explicit (empty) allowlist. Set-if-unset
  forwards of third-party variables (`GGML_METAL_PREC_F32_MM`) keep raw `getenv` on
  purpose, so an operator value is never overwritten.
  Pin impact: almost every `src/*.cpp` is listed in historical source-hash manifests
  under `docs/*.json`, such as `gguf-standard-file-type-v1.json` and the moedern and
  josev protocol manifests. Those receipts are frozen and were not edited, so a re-run
  of such a protocol needs a new versioned source pin. `python/crispembed/_moedern_text.py`
  (`CRISPEMBED_MOEDERN_FAST` now ignores `false`/`off`/`no`) is in
  `docs/moedern-public-text.json` `owned_sha256`. That pin was already stale on main
  before this change, so `tests/test_moedern_public_text.py` already failed there and
  is not run in CI.
- **Unlimited-OCR v1 registry URL: fixed.** It is now pinned to Hub commit
  `79d6658f26712df9479a77d0f0a5aaf8f807b6b5`. The LFS `x-linked-etag` at that commit
  equals the existing pin `84a9d881…`, size 2,252,419,328. The cache filename is
  unchanged, so existing caches stay valid.
