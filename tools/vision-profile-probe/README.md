# Source-compatible physical vision controls (unqualified)

The production draft is separately gated by `CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE`
and `CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION`. Existing defaults and Torch/BF16
arithmetic remain unchanged. Supported execution requires verified physical
ownership, the closed Linux AVX2/FMA/F16C four-thread private-ABI profile, and
P16/T2/M2 geometry of at most1024 patches; unsupported requested execution fails.

`controls.py` is a CPU-only model-free hosted worker: compile actual supported
and main-stub targets, source-fed adapter, and a pinned upstream22-bit resize
reference before any weight transfer. Eight predeclared synthetic resize shapes
cover identity, up/down and single-axis/skinny cases. Actual form RGB resize and
normalized packed patches must exactly match authenticated live projectorV3
outputs. It never acquires model weights or executes vision/text inference.
The reference bridge invokes actual upstream `img_tool::resize` at
llama.cpp `6c01da6278ab6c69ed67d9124806b50d2b299ba7`, preserving its arithmetic.

`native.cpp` is a separate, excluded reference-only executable for the next
component phase. Its readonly input is only the complete1,334,056,960-byte native
vision suffix plus an externally authenticated297-row schema. It never loads
text weights or a full F32 model. Schema rows are name, four GGML dimensions,
relative suffix offset and bytes, sorted by offset. The controller must prove
all297 per-tensor hashes and metadata against the immutable canonical manifest
before invocation. Legacy output must reproduce the retained composed-form
stages; the new profile must then compare against actual live source stages with
raw hashes, FP64 RMS/max and source magnitudes. A mismatch is diagnostic, not a
qualification. No component/decoded claims are established by the draft.

After component proof, complete independent decode on previously observed cases
is required. Historical BF16 text is a reference, not human ground truth;
held-out qualification remains a separate, provisional protocol.
