# Pinned projector attribution: setup evidence

The first projector-only worker at source `0f92f3b6a1e6f46a81df15ea5287e2b377a3bb53` stopped during executable linking after250.86seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v1-link-negative/projector-audit.json) SHA256 `5f6845f8d836e4dea130e1c0a52a42947645c677c5216f4e52222d7009fb8c34`; [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-projector-attribution/v1-link-negative/independent-review.json).

The standalone C++ object and pinned libmtmd compiled, but the reference target requested obsolete library `common`, producing `cannot find -lcommon`. Compile/load-before-transfer admission worked: no parameter transfer, canonical comparison or projector forward occurred. There are no valid stage metrics from this attempt.

The scoped proposed correction `c01008696` removes the unneeded dependency and links only mtmd, which publicly links ggml and llama. Source arithmetic, parameters, CPU profiles, observer controls and resource bounds are unchanged. A retry remains a new result; this failure is retained. No product/runtime code, default gate or public loader changes are included.
