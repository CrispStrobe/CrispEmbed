# Successor assets v4: v3 plus a measured macOS arm64 desktop tolerance

Protocol: [successor-assets-v4.json](successor-assets-v4.json). Contract and
controls: `tools/successor_assets_v4.py`, `tests/test_successor_assets_v4.py`.
Workflow: `.github/workflows/successor-assets-v4.yml`, consumers
`.github/workflows/successor-assets-consumers-v4.yml`.

**Owner decision (3 November 2026).** Accept the measured native desktop drift
that failed the strict v3 review as a cross-platform tolerance, **without
attribution**. v1, v2 and v3 files and receipts are read-only: v4 pins all of
v3's pins and every v3 file (60 in total). It keeps v3's producers, final base,
strict gates, WASM tolerance and surface expectations unchanged.

## Scope (narrower than requested, from the measured evidence)

The owner asked to scope the bound to arm64-vs-x86_64 comparisons. The stored
vectors of runs 40710490021 and 41471904133 show that the drift is specific to
the macOS arm64 build, not to the architecture:

| pair (final vector, same model and text) | max-abs | cosine | strict 0.002/0.99999 |
|---|---|---|---|
| Linux arm64 vs Linux x86_64 | 1.20e-4 | 0.9999997 | passes |
| Windows x86_64 vs Linux x86_64 | 2.2e-8 | 1.0 | passes |
| **macOS arm64 vs Linux x86_64** (reviewed) | **0.0025543533** | **0.9998886285** | fails |
| macOS arm64 vs Linux arm64 (not a review pair) | 0.00260 | 0.9998877 | fails |

The bound therefore applies **only** to the review's macOS arm64 comparison
against the Linux x86_64 reference. Linux arm64, Windows and every other pair
stay strict. Same-platform comparisons also stay strict, byte for byte from v1:
CLI, offline CLI, Python and compiled C against the C ABI (measured at most
5e-7), the 12 Rust/Dart binding comparisons, and repeats.

## Bound

| | measured (identical in 40710490021 and 41471904133) | v4 bound |
|---|---|---|
| max-abs | 0.0025543533265590668 | **≤ 0.00307** |
| cosine | 0.9998886284627958 | **≥ 0.999866** |

- **Rule:** the same as the WASM bound. The measured value plus a 20% margin,
  rounded outward to three significant digits. The review recomputes both
  metrics from the stored vectors (`compare_desktop_vectors`, unchanged v1).
- **Evidence limit:** both measurements use the same producer libraries, so
  they show execution determinism, not stability across builds.
- **Task level:** desktop consumers compare vectors of the same model and
  fixture text (enforced by the unchanged v1 review); embeddings have no
  token/text output.
- **Derivation:** the derived review differs from v1 only in the v3 browser
  gate and one desktop gate line. `desktop_gate` returns strict unless the
  platform is `macos-arm64` against `linux-x86_64` and the pair is within the
  bound. Tests accept the measured values and the exact bound, and reject
  0.00308 and a cosine of 0.999865. The same drift on any other platform is
  rejected, and every other gate stays strict.
- **Unattributed:** the likely mechanism, by analogy with the WASM attribution,
  is a different SIMD summation order in the macOS arm64 build amplified by
  activation quantization. That is unverified. Attribution with the same-input
  op-by-op method is an optional follow-up in PLAN.

Nothing is qualified, no release alias is created and the app stays on 0.17.12.

## Runs
