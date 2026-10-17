# PosFormer normalization and ceil-pooling correctness correction

## Scope

Start from the exact CrispMath production bridge source `11e6d598521976f38081934106b55095b46b40e3`. Apply only the verified post-position learned LayerNorm and default encoder ceil-pooling correction. Preserve the public API, scalar encoder, model files, token IDs/ARM behavior and other backends. No diagnostic probes, intermediate capture or bounded-decoding switches belong in the production library.

The decoder previously omitted exported `dec.input_norm` parameters. Default GGML pooling used floor-sized outputs for odd feature-map dimensions, whereas the trained architecture and existing scalar encoder use ceil mode. Duplicate only missing edge samples before ordinary 2x2 pooling; this preserves both maximum and valid-sample average arithmetic, including negative corner values. Require learned normalization and positional tensors with compatible dimensions when loading a PosFormer model.

## Existing measurements

CrispMath hosted run [37215287926](https://github.com/CrispStrobe/CrispMath/actions/runs/37215287926) independently compared the actual native stages against pinned official PosFormer code reconstructed from all exported FP32 tensors. The corrected default and scalar encoders and bounded decoder stages matched preset tolerances for five frozen samples. Sixteen native ceil-pool controls matched PyTorch exactly; D256/384 normalization controls matched independent LayerNorm math within 3.18e-7. This establishes bounded exported-weight runtime agreement, not original training-checkpoint parity or full-length recognition correctness.

The unchanged production Q8 model remained 7/50, preserving the same seven IDs; there was no observed recognition accuracy improvement. The public FP32 reference fixture remained 0/50 and is excluded from distribution. Its licensing and original checkpoint provenance are not resolved by this runtime correction.

## Required exact-source hosted validation

- Build the probe-free production library and baseline separately; fingerprint actual commits, C++ source and libraries, and reject production probe symbols.
- Run the pinned CrispMath public-API benchmark on unchanged original Q8 weights and the frozen 50 manifest on Linux and Apple/macOS. Preserve all seven previously correct IDs, not only the total count. Reject altered weights, references, scoring, corpus, image identities and partial reports.
- Re-run printed OCR and calculator handoff against the corrected production library.
- Build a separate test-only copy for synthetic negative controls and bounded independent exported-FP32 reference comparison. Record both production and instrumented source/library identities. Never include the reference model or probes in release artifacts.
- Run standard upstream native/Apple/Android build checks, then the existing release workflow in non-tag dry-run mode. Inspect native/WASM artifacts, source provenance and SHA-256 checksums before any publishing or app pin change.

## Promotion

Version `0.17.12` was unused at the initial inspection; no version/tag/release is created by this source change. Follow `scripts/bump-version.sh` only after the exact-source regression and dry-run report is reviewed. CrispMath must update the bridge source, plugin/runtime version and all artifact checksum pins together; source-only pin changes would still download old `0.17.11` prebuilts. Publishing, app integration and new model weights remain separate coordinated steps.
