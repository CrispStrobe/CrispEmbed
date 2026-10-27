# Prepared original-first Josev text backbone panel

This separately bounded CPU lane tests the explicit hidden-only implementation
at `75a5e7fca0daedfe98b695f7814d8bed202a27fb` with exact ggml
`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. Its source guard rejects any
change to that `src` tree, dependency, converter, original receipt/input freeze
or directly reused preservation/schema helpers. The new standalone probe,
comparator, driver and workflow are measured separately. No production observer,
root CMake, public API, runtime default or historical guard changes are included.

The only original numerical evidence is the reviewed four-case GPU F32 reference
at worker `b2edf8c83c9aca1f8a9210198d47798ac13828b8`, receipt
`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`.
The hosted preparation authenticates successful preservation
[40429513945](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40429513945)
at `5da7c64d1a95df0f9a829c62938fc8d7d64f77b7`, exact review artifact
`12702676604` and raw artifact `12702553413`, their service digests and every
manifest member, all 251 original NPY declarations and the unchanged public
receipt/input freeze. All four case identities, IDs and all-attended masks are
exported to a versioned binary packet with the original receipt/worker/freeze
identities. No new original model, tokenizer or GPU execution takes place.

Only after this authentication does the driver import pinned NumPy 2.2.6 and
GGUF 0.17.1 and verify every original array's finite/negative-infinity policy.
The exact original Hub revision
[`6f0a31c8ed683aba79097eb3f48df2c9ab560afd`](https://huggingface.co/mayflowergmbh/Josev-2B/tree/6f0a31c8ed683aba79097eb3f48df2c9ab560afd)
supplies the 4,548,221,488-byte checkpoint and original index under their frozen
SHA-256 identities, plus eight pinned source/license/config notices. The existing
converter promotes only the exact 320 text tensors, in bounded chunks to
file-backed staging, and reads back every F32 tensor byte. The text parameter
count is 1,881,825,088; its F32 payload is 7,527,300,352 bytes before GGUF overhead.
The unused vision/vocabulary/head paths never execute. Checkpoint identity and
promotion prove stored source bytes; the original receipt did not capture
individual loaded backbone weight hashes, and this lane does not invent them.

The [standalone probe](../tests/josev-backbone/probe.cpp) requires both explicit
default-off gates, the exact GGUF architecture/original metadata, all 320 F32
tensor shapes and the four frozen ID/mask records. It uses an explicit CPU
backend, retains weight ownership through synchronous computation, and frees
weights before the backend. It captures the unchanged embedding operation,
all 24 hidden layers and final normalization for every case: 26 stages each,
104 overall, each in the original `[1, tokens, 2048]` shape and occurrence zero.
Ordinary/repeated embeddings and every layer/final row must be bit-exact and
finite; vocabulary logits must remain empty. Per-case structural failures are
recorded while the remaining cases continue. Context length remains at most
512, unpadded batch one; images and cache/generation are excluded.

Every successful case is compared against its original arrays by actual
`crispembed_diff::Ref` and independent NumPy F64 reductions. The lossless F32
GGUF adapters preserve original shape and exact bytes; contiguous row axis zero
corresponds to 2048 hidden channels. Both fixed gates, maximum absolute error
`0.002` and minimum row cosine `0.99999`, must pass. Reports retain maximum,
mean and signed mean error, RMS, original/native L2 and min/max magnitudes,
and both the earliest byte-differing stage and earliest failing stage per case.
The comparator exits nonzero on a
numerical rejection; the collecting driver retains all four cases before its
final nonzero numerical result. Reporting agreement uses `2e-7` relative only
to cross-check metric calculations and cannot waive either fixed gate.
Compiled equal/changed-row controls precede native model access. The small
comparator and adapter derive from authored test-only source at
[`6fccd863`](https://github.com/CrispStrobe/CrispEmbed/tree/6fccd8635f8489d2eff2d47814a75cf77df6d584/tests/moedern-sdpa),
with a separate schema and 2048-dimension bound; they are not production code.

The reusable [workflow](../.github/workflows/josev2b-backbone.yml) is registered
as metadata phase `josev-backbone`. It requires at least 10 GiB available RAM
and 25 GiB free disk before downloads and again before conversion. All builds,
checkpoint/array access and inference run on hosted CPU CI, with two workers
and a 240-minute job bound. Logs, source/build identities, conversion receipt,
partial numerical results and rejection records are preserved. Large converted
weights, native observations, exact executables and comparator adapters remain
remote for 90 days. Preserve existing original artifacts before their service-reported expiry. No local build or model run is
part of source preparation, and preparing this lane does not dispatch it.

A passing backbone panel would qualify only these fixed text hidden states.
Head component and composed typed responses, public tokenizer/schema/API,
representative quality, larger context/images and runtime integration remain
separate. Existing Ovis/LightOn numerical and decoded preservation gates on the
modified decoder source are still required before integration. No default,
release, application upgrade or model-publication claim follows from this lane.
