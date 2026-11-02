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
header ([v3 observer](../tests/wasm-live-norm-v3/observer.h)) and brackets the
three scheduler-reservation builds. Apart from those six lines, the
`crispembed.cpp` overlay is byte-identical to v1 (tested).

## Attempts

- [41421952964](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41421952964)
  at `6288db9c`: the diagnostic observer had identical acceptance rules and
  logged rejection metadata. v2 admission and every weight-free control passed
  again. The same native abort occurred and logged
  `live_norm_reject: geometry name=norm00_input op=ADD type=f32 contiguous=1 ne=[384,16,1,1]`
  ([log](../tests/wasm-live-norm-v3/actual-41421952964-native1-observed.log)).
  This rejection is kept unchanged.

## Diagnosis

The rejected tensor is the first norm input (ADD, F32, contiguous) with
**16 rows, not 12**. On its first call, `encode_tokens` builds an unevaluated
scheduler-reservation graph at `bucket_seq_len(12) = 16` before the evaluated
12-token graph (`src/crispembed.cpp`, `measure_gf = build_encoder_graph(ctx, T_bucket)`).
The overlay's hooks run inside every `build_encoder_graph`. The observer
therefore saw the reservation graph, and its unchanged `ne[1] == 12` rule
correctly refused it. The `add_norm` op and `src[0]` checks never ran.

## v3 repair

- The overlay brackets exactly three reservation builds
  (`measure_gf = build_encoder_graph(...)`) with
  `ctx->live_norm.measuring = true/false`. Any other count rejects.
- While `measuring` is set, `begin()` only counts the build: no reset, no
  callback change. `add`/`add_norm` register nothing. Reservation graphs are
  never evaluated, so they produce no observation.
- Every v1 acceptance rule (12 rows, 384/1536 width, F32, contiguous, NORM or
  MAP_CUSTOM1, `src[0]` identity, 13 norms, unique names, 49 entries) is
  unchanged for the evaluated graph. No runtime/numeric change.
- A new weight-free phase, `observer-controls`, builds
  [observer_controls.cpp](../tests/wasm-live-norm-v3/observer_controls.cpp)
  against a06 GGML and runs 11 controls before model access. The model phase
  requires them. The controls reproduce the bucket-16 rejection, the no-op
  bracket and the still-rejected 16/13-row, wrong-op, disconnected,
  duplicate and fourteenth-norm cases.
