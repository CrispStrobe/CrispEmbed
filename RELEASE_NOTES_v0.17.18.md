# CrispEmbed v0.17.18

Patch release, 10 November 2026. Changes since [v0.17.17](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.17),
through `5420dee6` on 10 November 2026.

This release integrates the experimental LightOn4B (LightOnOCR-3 4B / Qwen3.5
OCR) work and the MoEdern consumer tests through the aggregate
[PR #199](https://github.com/CrispStrobe/CrispEmbed/pull/199). All new paths
are default-off and env-gated. Existing entry points, model defaults and OCR
routes are unchanged for existing callers.

## LightOn4B stack (experimental, default-off)

[PR #155](https://github.com/CrispStrobe/CrispEmbed/pull/155) and
[PR #157](https://github.com/CrispStrobe/CrispEmbed/pull/157), integrated in
[PR #199](https://github.com/CrispStrobe/CrispEmbed/pull/199):

- Qwen3.5 OCR source, vision, prefill and quantisation code for the LightOn4B
  profile (converter and profile scripts, pinned upstream source fixtures),
  including gated physical Q6 / GDN operator candidates and source-boundary
  prefill scheduling. Every path sits behind an environment gate and is off by
  default.
- **Not qualified.** The retained receipts record unresolved quality, numerical
  and reference-replay gaps. No new model alias is published and no default
  changes.
- **Release fence:** the public OCR entry point (`crispembed_ocr_model_init`)
  rejects raw `qwen35` and the experimental LightOn4B GGUF profiles from
  metadata alone, before any weights are loaded. Prototype gates do not
  override this.
- Bounded evidence and control workflows for this work run separately from the
  release build.

## MoEdern and experimental-alias consumers

- Consumer tests exercise the experimental MoEdern, UltraBERT, Josev and GOT OCR
  aliases through the CLI and the Python client, with per-precision loader
  gates. These are tests and receipts only; the aliases stay experimental.

## Reviewed PDF and PDF Craft

- The remaining reviewed-PDF conformance pieces and the latest PDF Craft
  document-quality commits from their branches are included. Behaviour matches
  v0.17.17: capabilities still report `47`, without the BIDI bit.

## Packaging and CI

- Android builds no longer run the x86 CPU-feature probe; LightOn4B / Qwen3.5
  validation executables are not built for iOS.
- Test and CI fixes: a link fix for `test-lighton4b-text-owner`, CMake target
  de-duplication, Python test-isolation fixes, and hash-pinned tests that skip
  in shallow clones only when their pinned commit is absent.

No default model, default OCR route or pub.dev publication changes.
