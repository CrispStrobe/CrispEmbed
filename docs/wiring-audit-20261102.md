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
| CLI | **fixed**: `--ocr-engine` accepted only `unlimited_ocr` while help and logs print `unlimited-ocr`; now also `unlimited-ocr`, `unlimited-ocr2`, `unlimited_ocr2` (`examples/cli/main.cpp:1432`), and the `unlimited-ocr2` alias defaults to the v2 registry model instead of v1 (`examples/cli/main.cpp:1588`) | wired `examples/cli/main.cpp:1428` (id 23; explicit `--ocr-rec` required) | wired `examples/cli/main.cpp:10` | wired `examples/cli/main.cpp:115` |
| HTTP server | **fixed** the same aliases and v2 default (`examples/server/server.cpp:1578`, `:1690`) | wired `examples/server/server.cpp:1586` (id 23, default `minicpm-ocr`) | wired `examples/server/server.cpp:39` | wired `examples/server/server.cpp:6` (`POST /systemone`) |
| Model registry | wired `examples/cli/model_mgr.cpp:1896` | wired `examples/cli/model_mgr.cpp:1885` | wired `examples/cli/model_mgr.cpp:1641` | wired `examples/cli/model_mgr.cpp:1636` |
| SHA-256 pins | wired `examples/cli/model_hashes.h:579` (immutable revision) | wired `examples/cli/model_hashes.h:575` | wired `examples/cli/model_hashes.h:501` | wired `examples/cli/model_hashes.h:499` |
| Python | wired through `CrispOcrModel` / orchestrator stages; **fixed** `OCR_ENGINE_UNLIMITED_OCR` (`python/crispembed/_binding.py:4205`) | wired; **fixed** `OCR_ENGINE_MINICPM_OCR` (`python/crispembed/_binding.py:4206`) | wired `CrispMoEdern` (`python/crispembed/_moedern_text.py:16`) | wired `CrispSystemOne` (`python/crispembed/_systemone.py:7`) |
| Rust | wired (stage int); **fixed** `OCR_ENGINE_UNLIMITED_OCR` (`crispembed-sys/src/lib.rs:211`, `crispembed/src/lib.rs:2721`) | **fixed** `OCR_ENGINE_MINICPM_OCR` (`crispembed-sys/src/lib.rs:213`, `crispembed/src/lib.rs:2722`) | wired `MoEdern` (`crispembed/src/lib.rs:40`) | wired `SystemOne` (`crispembed/src/lib.rs:29`) |
| Dart / Flutter | **fixed** `crispOcrEngineUnlimitedOcr` (`flutter/crispembed/lib/src/crispembed.dart:4073`) | **fixed** `crispOcrEngineMinicpmOcr` (`flutter/crispembed/lib/src/crispembed.dart:4076`) | wired `flutter/crispembed/lib/crispembed.dart:9` | wired `flutter/crispembed/lib/crispembed.dart:17` |
| Env vars documented | wired (this audit) | wired (incl. PR #109 speed gates) | wired | wired |
| CrispSorter / CrispCalc catalog | external, not verified | external, not verified | N/A (not OCR) | N/A |

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

## Open gaps

The scoped TODOs are in [PLAN.md](../PLAN.md#wiring-and-environment-variable-audit--2-november-2026).

- **CLI vs server default for MiniCPM-OCR.** The server resolves `minicpm-ocr`
  by default for id 23. The CLI requires an explicit `--ocr-rec`. Pick one
  behaviour.
- **CrispSorter / CrispCalc.** Neither consumer repository was checked for
  ids 13 (v2) or 23, and neither engine is qualified for a consumer default.
- **131 raw presence-style env reads.** For these, `X=0` means *on*. They are
  marked `presence` in the inventory, for example `CRISPEMBED_SHARED_GPU_BACKEND`,
  `CRISPEMBED_PPOCRV6_FORCE_CPU` and several `*_DEBUG` switches. They should
  move to `core_env::on`, which changes behaviour only for an explicit `=0`.
- **Unlimited-OCR v1 registry URL** (`unlimited-ocr-q4_k-stacked.gguf`) is pinned
  by hash but resolves `main`, not an immutable revision. This predates the
  engines in this audit.
