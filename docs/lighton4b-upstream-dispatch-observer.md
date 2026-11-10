# Reference loader dispatch metadata

`tools/lighton4b_upstream_dispatch_log.py` patches only
[`llama-model.cpp` at 6c01da6278ab6c69ed67d9124806b50d2b299ba7](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/llama-model.cpp).
Its original SHA256 must be
`c41727dd626ef2e68e6c2af82bfb2600f3c77222b70f391e214b1c9368157290`.
A byte-removal check proves that only the declared metadata insertion and two
standard-library includes change. Apply once, alongside the existing reference
feature/logit observer, before the external server build.

Set `CRISPEMBED_TEST_UPSTREAM_DISPATCH_DIR` to an existing output directory to
write `source.tensor-dispatch.tsv`. With the variable absent, the observer does
nothing. Capture both reference processes independently and retain their exact
full IDs, probabilities, decoded output and EOS neutrality checks, plus the
original 256-token prefix/probability and source-feature/logit anchors.

The insertion runs after all tensor data loads, repacking and Hadamard setup,
before any model forward. Nine fixed rows describe input norm, QKV, Z, alpha,
beta, output projection, FFN gate, and the actual embedding/output-head tensor
roles. Fields include the exact tensor type/shape, backend buffer and buffer-type
names, host and extra-traits presence, and pointer/data alias **booleans**. It
never emits pointer addresses, weights, model paths or environment identities.
It reads no tensor payload values and changes no graph, sampler, tensor metadata,
traits, allocation policy or arithmetic. Missing or ambiguous selected tensors
fail the diagnostic rather than substitute a different role.

This observes loaded tensor metadata. It is not itself a per-operation kernel
execution trace or evidence of decoded quality. In particular,
[`repack.cpp`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-cpu/repack.cpp)
selects an AVX2 Q4_K repack variant only for eligible shapes and compatible
operations. The actual backend buffer and extra-traits fields must be measured;
a compiled `GGML_CPU_REPACK=ON` flag alone does not establish dispatch parity.
No production target imports this helper, and no new model job is implied.
