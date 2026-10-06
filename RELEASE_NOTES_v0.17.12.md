# v0.17.12 — PosFormer runtime correctness hotfix

This stable hotfix starts from the deployed `11e6d598521976f38081934106b55095b46b40e3` source. It does not include the subsequent unrelated changes on main.

PosFormer now applies the trained decoder LayerNorm after positional encoding. The default GGML encoder now uses the architecture's ceil-mode pooling for odd feature-map dimensions, preserving valid-sample averages and maxima at the edges. The public API, scalar encoder, existing model files and token/ARM conventions are preserved. No diagnostic probes, bounded-decoding switches or new model weights are distributed.

Hosted Linux and macOS measurements preserved the original model's same seven correct cases on an unchanged 50-sample corpus. Recognition remained 7/50; this is a runtime correctness correction, not an accuracy improvement claim. Separately built native controls and bounded independent exported-weight reference comparisons verified the corrected math. Printed OCR/calculator handoff and native/Apple/WASM build regressions were checked. Physical iPhone/iPad testing remains separate.

## Asset coverage

This release contains CPU native artifacts for Linux x86_64/arm64, macOS arm64, Windows x86_64, Android arm64-v8a/armeabi-v7a and iOS arm64, plus OCR and embedding WASM bundles. CUDA and Vulkan variants are intentionally not built or published for this hotfix. No previous-version binary is repackaged as v0.17.12.

`runtime-artifact-manifest.json` and `wasm-release-artifact-manifest.json` record the actual tagged source, GGML pin, version and SHA-256 hashes of the published artifacts. Consumers must use the published hashes; tarball timestamps can make dry-run archive hashes differ from the final tag build.
