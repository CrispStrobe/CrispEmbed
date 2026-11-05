# CrispEmbed v0.17.15

Patch release, 5 November 2026. Changes since [v0.17.14](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.14),
reviewed through `0b0af65e5914b6c50320147982caf4c204a50622` on 5 November 2026.

The main user-facing fix restores real-background propagation on grainy video
when using CLI temporal inpainting. This release also adds opt-in LightOnOCR-3-1B
numerical corrections, repairs its converter, and records the first completed
qualification panels for the experimental OCR profiles introduced in 0.17.14.

## Temporal inpainting: film grain no longer blocks background copying

- With `--inpaint-temporal`, ordinary film grain could reject every clean donor
  frame on a still shot. The stream then used an engine-generated fill instead
  of copying the real background. [PR #162](https://github.com/CrispStrobe/CrispEmbed/pull/162)
  corrects the motion score by subtracting a six-level noise allowance from each
  cell's mean absolute RGB residual, while retaining the cell-mean motion test.
- Added coverage for past and future donors, exact copied pixels, unchanged
  surrounding pixels, grainy fill reuse, propagation/reuse/blending boundaries,
  and coherent brightness changes. The temporal suite now has 396 checks;
  the submitted fix had 270 before review additions.
- The author's synthetic Apple MI-GAN512 benchmark reports masked PSNR improving
  from 27.15 to 34.57 dB on the static clip, with the moving clip essentially
  unchanged. Those model-backed figures were not independently replayed during
  integration. See [PERFORMANCE](https://github.com/CrispStrobe/CrispEmbed/blob/v0.17.15/PERFORMANCE.md) for settings and limitations.
- Temporal processing remains opt-in. The allowance applies to a cell average,
  so low-contrast motion can still be accepted; it does not bound the error of
  every copied pixel. Representative real-video quality remains open.

## LightOnOCR-3-1B: converter repair and opt-in CPU corrections

- Correct checkpoint-name handling in the LightOn3-1B GGUF converter so the
  original checkpoint's legacy language-model namespace maps to the existing
  native loader. Repair the pinned reference dependencies and capture setup.
- Add `CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS=1` to use the original Pixtral vision
  RMSNorm epsilon (`1e-5` instead of `1e-6`) for the admitted 1B profile.
- Add `CRISPEMBED_LIGHTONOCR3_KV_F32=1` for an F32 persistent KV cache on the
  admitted LightOn3-1B CPU path. This avoids the old cache's F16 rounding and
  doubles the cache payload relative to F16. It is not a speed optimization.
- With both corrections enabled, F32 and F16 each match all six retained original
  free outputs in the transcription/grounding panel. Old-path and prefill byte
  controls pass. The epsilon-only panel remains negative at F16, and strict
  intermediate failures and capped outputs remain recorded.
- Both correction gates stay **off by default**. This does not change legacy
  LightOn profiles or qualify a GPU path. The existing
  `CRISPEMBED_LIGHTONOCR3=1` admission gate is still required. Gate details and
  grounding admission are in [the environment reference](https://github.com/CrispStrobe/CrispEmbed/blob/v0.17.15/docs/environment-variables.md).

## Experimental OCR: completed evidence, remaining limits

These profiles already existed in 0.17.14; this release adds execution evidence
and audit repairs rather than a new set of qualified models.

- **MinerU2.5-Pro-2605:** the complete eight-case, four-precision panel passes
  32 numerical arms. F32/F16 decoded outputs match all eight cases each; Q8/Q4
  match six each, with retained layout regressions. Add processor/helper checks,
  hosted CPU preparation fixes and authenticated remote archive preservation.
- **Xiaomi-OCR-0:** fix metadata validation for the original processor's runtime
  list/tuple representations while retaining exact parameter/source checks.
  All four bounded F32/F16 final-logit and decoded comparisons pass, but every
  output hits the 16-token cap and six strict intermediate comparisons fail.
  This is not EOS or document-quality qualification.
- **Nemotron Parse frontend:** record 11 geometry/packing cases and nine negative
  controls passing. Native resize and the neural backend remain separate work.
- **LightOnOCR-3-0.8B:** preserve the failed exact-original guard and declare a
  retained-capture continuation. No successful current-source native replay is
  claimed by these notes.

The [qualification continuation](https://github.com/CrispStrobe/CrispEmbed/blob/v0.17.15/docs/ocr-qualification-2026-11-04.md) links exact
sources, reports, raw failures and outstanding gates. LightOn4B draft work in
PRs #155 and #157 is not part of this release. No model alias, model publication,
quantized-quality promotion or OCR default flip is added.

## Validation and release engineering

- The inpainting workflow moves from the retired macOS 14 runner to macOS 15.
  Final PR and merged-source runs pass 396 temporal checks, 48 CLI checks and all
  five fake-engine video cases on Linux, macOS and Windows.
- Expand hosted OCR controls, retained-reference comparison and artifact
  preservation tooling. PR141's compilation receipt records 294 model-free
  controls and actual native/mobile/Apple/WASM/Rust builds on its exact source.
- Full Build, OCR/embedding WASM, Rust and inpainting checks are required on the
  frozen release source before tagging. Exact validation and publication records
  belong in [PLAN](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md) and
  [HISTORY](https://github.com/CrispStrobe/CrispEmbed/blob/main/HISTORY.md).
- Native/package source versions are 0.17.15. Downstream application pins remain
  0.17.12, and pub.dev publication remains separately unqualified. No application
  upgrade or pub.dev upload is included in this release.

[Full comparison since v0.17.14](https://github.com/CrispStrobe/CrispEmbed/compare/v0.17.14...v0.17.15).
