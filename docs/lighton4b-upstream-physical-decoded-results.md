# Same physical text weights and source features: decoded result

The native baseline does not meet the requested llama.cpp output floor. With
identical physical publisher text weights and exact captured upstream image
features, form CER is worse despite better WER, table reaches the 256-token cap,
and Arabic matches exactly. No shipping, default change or runtime-parity claim
follows from this experiment.

The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/independent-review.json)
and [raw result](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/physical-decode.json)
preserve Kaggle CPU version2 at source
`ac34f72ca89823a177c538ad598dfa00fc69f124`. The first submitted dependency/setup
failure remains alongside it. The generated candidate is exactly the same
4,041,905,760-byte file as the first persisted transport, SHA-256
`843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.
No original BF16 forward, full F32 model, requantization, new calibration policy
or heldout evaluation ran.

Edits below are measured against the frozen genuine BF16 output, a reference
rather than human OCR gold. Counts use raw Unicode codepoints and
whitespace-delimited words, with no text normalization.

| Case | Native IDs / stop | Native character edits | Native word edits | llama.cpp Q4_K_M character edits | llama.cpp Q4_K_M word edits |
|---|---|---|---|---|---|
| Form | 174 / EOS | 202 / 656 | 38 / 90 | 189 / 656 | 46 / 90 |
| Table | 256 / cap, no EOS | 238 / 607 | 71 / 106 | 315 / 607 | 88 / 106 |
| Arabic | 6 / EOS | 0 / 12 | 0 / 2 | 0 / 12 | 0 / 2 |

Form fails a floor requiring both edit metrics to be no worse than llama.cpp:
202 character edits exceed 189. Table's smaller distances do not make its
capped output complete. Arabic is a short, previously observed two-word case.
The raw [form](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/simple_form/upstream_features/generated.txt),
[table](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/simple_table/upstream_features/generated.txt)
and [Arabic](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/arabic_printed_line/upstream_features/generated.txt)
outputs and adjacent ID files are retained. Exact ID prefixes versus the retained
upstream full outputs are 31, 37 and 6 respectively; only Arabic is fully exact.

The source controls close all three original prompt-ID byte hashes and actual
image/prompt geometry. Fresh observer-OFF and observer-ON upstream runs exactly
match the first two IDs and probability records of each retained full llama.cpp
baseline. Native Q6 embedding rows, the actual feature splice and complete
text/image MRoPE axes match independently decoded source data byte for byte.
The duplicated first-16-token layer0 block also matches the actual main hidden
prefix by `memcmp`; its operator files are therefore tied to the actual input.
This isolates text runtime differences without inserting reference arrays into
production. The production vision path is not evaluated in this source-feature
arm; protected original vision weights differ from the source F16 projector.

| Case | First-score RMS vs source | Max absolute error | Source max absolute value | Native process seconds | Observed peak RSS bytes |
|---|---|---|---|---|---|
| Form | 0.1023485054 | 0.5765159130 | 22.73378181 | 1677.989 | 3,204,321,280 |
| Table | 0.1552013612 | 1.0623531342 | 20.54155350 | 2438.190 | 3,151,659,008 |
| Arabic | 0.0947613313 | 0.5162243843 | 18.17414284 | 99.461 | 3,146,735,616 |

These times include model loading, reference-driver stage IO and duplicate
first-block observation; they are not product performance claims. This frozen
baseline uses the actual dequantized-F32 matmul/scalar-RMS implementation, not
retained quantized source matmul. A legacy `DIRECT_QUANT` environment entry does
not prove kernel selection; the current code has no reader for that flag.
The new, independently default-off physical arithmetic gate is evaluated in a
separate two-token smoke. Local primitive equality is not whole-model parity.

All 81 downloaded small outputs (1,677,754 bytes) are retained with an
[artifact index](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-physical-decode/v2/artifact-index.json).
Independent review rechecks 54 case-file hashes, every raw ID/text/stop result,
all six codepoint/word edit comparisons using a full two-dimensional edit table,
original and upstream capture hashes, and submitted bootstrap SHA. Large arrays
remain remote, with sizes and SHA-256 in the result. The separately scoped
read-only actual-main primitive closure rehashes those remote arrays and
compares the retained first-layer operators; its receipt is recorded separately.
