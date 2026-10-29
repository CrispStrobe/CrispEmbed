# MiniCPM missing seven arms completed

[Run40855153707](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40855153707)
at `2fde0c64bd6e637441beeb3d96cf47b0bc3a2cdc` completes the seven missing
ordinary forwards and two old-profile projection probes. No original forward
was repeated. [Independent review](../tests/results/minicpm-ocr/seven-arm-v3-review.json)
authenticates all405 small artifact members and checks actual Ref against
independent metrics. Raw arrays/models remain in remote canonical artifact
12892798527; binaries remain in12893237130.

All30 carried/new cases retain exact input/generated IDs, text, EOS and zero
CER/WER, including297 new generated IDs. This is decoded parity on the frozen
five-case development set, not representative OCR quality qualification.

The combined180 vision comparisons have130 passing and50 failing. F32 and
F16 each pass50/60; Q4_K passes25/30; Q8_0 passes5/30. The new tiny F32 arm
first exceeds the strict absolute budget at vision layer26: max0.0137939453,
RMS0.0000369527, cosine0.999999999982, original magnitude1629.5715.
The Q8 tiny arm already differs at vision layer0. Preserve all numerical
flags; high cosine cannot erase an absolute rejection. Higher precision alone
cannot explain or fix failures shared by F32/F16/Q4.

The recipes protect different groups: Q4_K changes186 decoder projection
matrices and preserves593 tensors from the F16-source model, including vision.
Q8 quantizes vision linears while retaining the exact F16 patch embedding.
Its tiny first vision block has max0.0706520/RMS0.00215511/cosine0.999984931;
patch protection alone is insufficient. A new decoder-only Q8 ablation should
retain all593 exclusions byte-exact before inference; this is not evidence
that four-bit storage is inherently more accurate than eight-bit storage.

The original merger hook captured LIST element0 only. Overview correspondence
is explicit;96 later crop parts across all30 arms remain unmapped to original
tensors. Next, declare a one-case source-derived vision operator attribution
with exact pixels and weights, starting at the earliest F32 boundary. Separate
its baseline rounding from Q8 error using same-input/same-weight replay before
proposing critical-tensor protection. Do not recapture the whole original set
or change budgets. Runtime integration, representative quality, all clients
and quantized publication remain open.
