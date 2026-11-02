# One-sentence live norm operands v3: observer protocol

v2 ([protocol](wasm-live-norm-operands-v2.json)) repaired component admission.
Its run [41418312256](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41418312256)
passed every weight-free component/browser control, model authentication and the
observer build. It then aborted in the first native observed encode:
`GGML_ASSERT(ctx->live_norm.add_norm(input, result))` (SIGABRT). That rejection
is retained. The plain native anchors were recorded first.

v3 ([protocol](wasm-live-norm-operands-v3.json),
[workflow](../.github/workflows/wasm-live-norm-operands-v3.yml)) reuses v2
admission and every v1 phase unchanged. It swaps only the disposable observer
header ([v3 observer](../tests/wasm-live-norm-v3/observer.h)). The
`crispembed.cpp` overlay is byte-identical to v1 (tested).

## Attempts
