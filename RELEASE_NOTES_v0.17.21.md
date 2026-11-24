# CrispEmbed v0.17.21

- Preserve horizontal recognition steps in the PP-OCRv6 tiny model.
- Restore Vulkan recognition for PP-OCRv6 with ordered graph execution and direct batched depthwise convolution.
- Identify detector confidence separately from recognition confidence in PP-OCRv6 output.

Tiny, small, and medium matched the generated line transcriptions on RTX A1000 and Iris Xe in single-line and batched Windows tests. See [the benchmark](PERFORMANCE.md#pp-ocrv6-vulkan-recognition-24-november-2026).
