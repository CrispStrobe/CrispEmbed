# Genuine original BF16 mixed weights: bounded full free generation

This observed-form mechanism check asks whether the corrected native mixed
156-token EOS output differs from the genuine original BF16 model with the same
fixed attention/embedding Q8 + FFN Q4 weights. It creates no F32 model, replacement
gold, held-out calibration or model publication. The official BF16 original and
all previous numerical/decoded negatives remain unchanged.

The [immutable protocol](lighton4b-bf16-mixed-freegen-v1.json) retains the validated
streaming oracle's original checkpoint/driver/package/input pins, complete
723-tensor mapping, 153 Q8 + 96 Q4 selected matrices, 474 protected originals,
parameter identity/storage and tied embedding/output head. Dequantization chunks
are at most 65,536 F32 elements, converted to BF16 before in-place parameter copy.
Only one candidate file is mapped at once; clean-cache advice and existing
4 GiB streaming / 8 GiB effective plus 2 GiB actual cgroup hard-margin prefill
guards remain. Both complete public GGUF files pass full immutable hashes.

Before model loading, authenticate the previous one-position BF16 mixed report
and complete corrected-native report through immutable public source paths and
SHA-256. After streaming, all 249 effective BF16 mixed parameter hashes and all
723 original BF16 source hashes must equal the previous independent oracle.
The original one-token baseline must reproduce official first-score SHA
`a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd`.

Then execute exactly one independent original mixed-weight greedy generation,
cap 256, cache enabled, sampling disabled and original effective EOS 248046 /
source text-pad 248044. There is no teacher-forced replay. Count one original
anchor forward plus one root/text forward per generated token; vision must run
exactly once per arm. Accept EOS at any allowed position or an honest 256-token
non-EOS cap, with no assumption that the result matches 156 or 177 tokens.
The mixed first score must retain frozen SHA
`bea1064566634615c955582808c75b0003dc8f24983cbd12cbb661a1b5c33ae9`.
Protected parameters, buffers, dtype, storage and tied head are rechecked after
the generation. The unchanged original driver's audit receives only the minimal
baseline result and records all actual model calls, not a misleading two-call
count for multi-step generation.

Retain each complete finite vocabulary row (248,320 scores / 993,280 F32 storage
bytes), score SHA-256, raw generated IDs/text and content hashes. At most 256
score rows / 254,279,680 bytes are written; no intermediate stages or caches are
captured. Partial score manifests are persisted every 16 rows. Report independent
IDs/text, prefix divergence, actual EOS/cap and raw CER/WER agreement against both
the frozen official BF16 original and corrected native full report. CER preserves
Unicode case, punctuation, accents and newlines; WER uses Unicode whitespace
splitting with case/punctuation retained. These measure model-output agreement,
not human-ground-truth OCR accuracy.

The native report is public and hash-pinned; no GitHub artifact credential or
access token is passed to the model worker. Numerical native-row comparison is
deferred to a separate read-only consumer if needed, using only rows sharing
identical causal prefixes. Raw native artifact identity is retained for that
purpose. The effective-precision limit remains explicit: native retains F32
dequantized weights, whereas this genuine BF16 original rounds parameters to
BF16. A residual cannot automatically be called a logic bug or subtracted from
quantization loss.

Use exactly one CPU-only worker with a 5,400-second model-worker limit, initial
22 GiB effective RAM, 27 GiB scratch and 2 GiB output admission. The previous
mixed worker peaked at 29.65 GB of a 32.21 GB cgroup; reject capacity violations
and preserve partial failure instead of retrying blindly. The previous original
177-token BF16 generation took 568 seconds. The extra cache/score memory is
bounded, and all large payloads stay remote. Seventeen model-free controls pass
before dispatch; no local framework/model inference or build occurs.

The worker may finish red for the already known original-score numerical
rejection while still completing full mixed generation. Inspect capture status,
forward counters, hashes, actual stop and decoded agreement independently. No
runtime/default/release/app-pin change or new PR follows from preparation.
