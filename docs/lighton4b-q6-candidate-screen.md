# Actual selective Q6 candidate: calibration first-score screen

The Q4_K_M-style candidate was built and executed successfully, but its first-score
improvement is mixed. Arabic improves; the table's RMS barely changes and its
original-token probability worsens. Both emit the same single token as plain Q4.
This is a calibration screen, not complete decoded-quality or EOS proof.

## Exact candidate and lineage

[Run42212846887](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42212846887)
uses immutable source[c3d6b5f88](https://github.com/CrispStrobe/CrispEmbed/tree/c3d6b5f886cdef07216d37c863bc70e6e06e0b43).
Candidate`lighton4b-q4km-q6-q8head-v1.gguf` contains4,195,863,232 bytes, SHA
`76911f052870f4e800fdcea5d5a6205096883be3407867ddadb81e52b7f1fb09`.
Its custom marker and UINT32 file type15 accompany actual32 Q6_K matrices,
216 Q4_K matrices and one Q8_0 embedding shared by the output head. This is an
explicit Q8-head exception to the upstream-style policy, not an identical stock
llama.cpp export or a new public model alias.

The[conversion receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-candidate/conversion.json)
records691 payloads copied byte-for-byte from frozen plain Q4`f873eb3d…`, including
the216 Q4 matrices, Q8 head and474 protected F32 tensors. The[32 selected tensors](../tests/results/qwen35-ocr/lightonocr3-4b-q6-candidate/selected32.json)
are generated directly from original BF16 checkpoint`6b0dab74…`, revision
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`. No quantized value is used as a Q6
input. Source BF16 hashes match the retained723-parameter oracle; all691 copied
raw/BF16 hashes independently match it. All32 new Q6 tensors pass independent
native-versus-gguf decoded-BF16 hash checks, and all249 effective BF16 weight hashes
are retained. Actual row chunks contain64,000 or64,512 elements, below65,536.
No F32 model or original framework forward runs.

The[worker receipt](../tests/results/qwen35-ocr/lightonocr3-4b-q6-candidate/worker.json)
SHA is`c5459a3870b06648df125dc97df4e1f3b70eba19f21b3dfef8709f9948b4e966`;
conversion receipt SHA is
`54677f2c0a05815799b0d577735da8e05f46146c02788e7756416ff965843602`.
GGML is pinned to`3a27b6c665b191cb230be367db5c52d81dd3e2ac`. The predeclared
[input protocol](../tests/results/qwen35-ocr/lightonocr3-4b-q6-candidate/protocol.json)
is retained unchanged, including its historical prepared status; actual completion
is recorded in the receipts.

## Same-worker numerical screen

Only`simple_table` and`arabic_printed_line` calibration captures are read. No
held-out capture, labels or original batch JSON is downloaded. Both arms use the
same binary and four threads, corrected vision, BF16 head, parallel dequantization,
and disabled BF16-weight/decoder-boundary alternatives. Both plain-Q4 first-score
hashes exactly reproduce retained source26e6 controls before Q6 is measured.
Actual native input arrays are admitted against the original prepared inputs.

| Calibration case | Metric against retained original BF16 | Plain Q4 | Selective Q6 |
| --- | --- | ---: | ---: |
| Table | RMS | 0.70626033 | 0.70573124 |
| Table | Maximum absolute error | 4.2265625 | 4.359375 |
| Table | Cosine | 0.93886537 | 0.94277708 |
| Table | Original-token probability | 0.40557788 | 0.27142059 |
| Arabic | RMS | 0.36463124 | 0.31669830 |
| Arabic | Maximum absolute error | 2.953125 | 2.40625 |
| Arabic | Cosine | 0.98499781 | 0.98876507 |
| Arabic | Original-token probability | 0.87176702 | 0.90418077 |

Table emits token20353, prefix`<table`, in both arms. Arabic emits67321, prefix`أ`,
in both arms. All four runs stop at the declared one-token cap without EOS.
No CER/WER or complete-output result is inferred from these prefixes. The actual
portable Ref binary is`d6898491…`; equal/changed controls pass and all four raw
comparisons fail retained max0.002/mincos0.99999 budgets. Workflow success means
conversion and measurement completed, not that numerical parity passed.

## Earliest measured change and tested support

All shared pixels, embedding, prefill inputs/positions and24 vision-layer outputs
are byte-exact between arms. The first changed native decoder layer is0, where
selected weights intentionally differ. Table Q6-minus-Q4 layer0 max error is
0.071660995/RMS0.002572619, versus baseline/candidate magnitudes11.50419/11.53368.
Arabic max is0.054203033/RMS0.002535207, magnitudes7.33496/7.34280. These are
same-input candidate-versus-baseline changes; no original decoder-stage parity
claim follows. Raw native prefill stages and first scores remain in the artifact.

Guarded owner/metadata/streamed/grouped-GDN support is integrated in PR155 at
[635be5d9b](https://github.com/CrispStrobe/CrispEmbed/tree/635be5d9b36bbe8cf365d9109c38e498960ecc19).
All713 native controls pass in42207472045 and repeat within the actual producer.
Initial42206305308 compiled but failed control703 at a narrower Q6 stream
whitelist; that negative is[preserved](../tests/results/qwen35-ocr/lightonocr3-4b-q6-initial-controls/review.json).
The grouped whitelist was also extended behind the same gate. The
[model-free converter proof42210897150](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42210897150)
passes independent quantizer, BF16 decoder, chunk-tail, neighboring-byte and
wrong-source-hash controls. Converter files are unchanged between that proof and
the actual run; runtime source is unchanged from the integrated PR155 snapshot.

Initial resources were15.76GB available RAM and114.89GB free disk. Guards repeat
before native cases, but high-water RSS was not measured. Candidate storage grows
164,905,280 bytes over plain Q4, including320 metadata bytes. No performance,
complete decoded-quality, held-out, default or release qualification follows.

Artifact13500128433 contains3,549,881,778 ZIP bytes, SHA
`21cb50e617a777073da609bf6947a558d1f42bb1804f217722ff7a3f51c2e561`,
retained through2027-02-04T11:47:35Z. The complete model stays remote; only small
receipts/logs and first-layer pairs were downloaded for review. No model-hub
upload, registry alias, default or release changes.
