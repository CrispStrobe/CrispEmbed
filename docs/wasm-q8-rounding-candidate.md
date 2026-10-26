# Conditional WASM Q8_0 rounding candidate

This unmerged, default-off candidate tests a concrete source discrepancy. It
has no executed numerical result and does not establish the cause of the
[initial packaged browser failure](../tests/results/packaging/successor-wasm-initial.json).
The [frozen source trace](successor-wasm-trace.md) and a separately reviewed,
source-bound candidate A/B must complete before merge, a default change or new
release assets. The app remains on its qualified immutable release.

The CrispEmbed base is973b200edd5d1d7429edecff1bd3ff42f0d80ca7. The ggml
candidate is [a06fc0bb46b73cc10011fc90ae4054aee8438aeb](https://github.com/CrispStrobe/ggml/commit/a06fc0bb46b73cc10011fc90ae4054aee8438aeb),
on feature branch`feat/wasm-q8-rounding-candidate-20261026`, directly based on
pinned890278a8342c620197c90e702e1188bcab94f510. Fork branch
`sync/upstream-v0.17` was inspected at5049ebb8472fdc965eb3fb72c1cb111260726186:
its WASM converters and CPU plan/header files are unchanged from890. Its
unrelated UE4M3/NVFP4 changes are not included. Exact per-file identities are in
[the candidate source declaration](wasm-q8-rounding-candidate.json).

At890, the WASM SIMD Q8_0 converter truncates scaled F32 inputs toward zero.
The x86 SIMD converter rounds nearest-even, while the scalar reference uses
`roundf`, including ties-away. The candidate adds a separate converter with
`wasm_f32x4_nearest` before the existing integer truncation. The legacy
`quantize_row_q8_0`, all of`quantize_row_q8_1`, and the global CPU type-trait
table remain byte-identical. Halfway cases must keep these policies distinct.

The new CPU setter configures a backend-local boolean. CPU initialization leaves
it false; direct execution and cached-plan creation copy it into`ggml_cplan`,
and graph execution copies it into the worker compute parameters. Cached plans
retain the value captured when created. Converter selection is local to ordinary
and indirect matrix multiplication, and applies only when the activation
conversion type is Q8_0 under WASM SIMD. Other architectures, Q8_K, Q8_1,
prequantized inputs and conversion calls outside these two operators retain
existing arithmetic. No global mode or function-pointer mutation is used.

The encoder loader configures only its own CPU backends, only under Emscripten,
using`core_env::on("CRISPEMBED_WASM_Q8_0_ROUND")`. Unset, empty and literal`0`
preserve the legacy path; `1` enables the conditional path. Each new context
reads the environment at initialization. Existing contexts retain their settings.
As with existing CPU setters, callers must configure a backend before executing
it; concurrent reconfiguration of the same backend is unsupported. Independent
backends may hold different settings.

The public ggml CPU plan structure gains a boolean field and a CPU configuration
API. This is an internal dependency change requiring all affected binaries to
be rebuilt; it adds no CrispEmbed C ABI, browser environment setter or exported
diagnostic wrapper. Weight files and production probes are unchanged. No
version, tag or application pin is advanced for this unmerged experiment.

The dedicated `ggml/tests/wasm-q8-rounding/CMakeLists.txt` entry point exposes
`test-wasm-q8-rounding`, explicitly excluded from ordinary builds. It requires
Emscripten and disables the generic test suite, whose host-ISA discovery would
otherwise inject native AVX flags into a cross-build. Its source includes:

- Direct legacy/new/reference block conversion on positive/negative fractions,
  halfway values and zero, preserving FP16 scales and distinguishing ties-even
  from ties-away.
- Actual backend graph execution of`mul_mat` and`mul_mat_id` for weight-free
  authored Q8_0 and IQ4_NL constant-one matrices: input `[127, 1.6 ×31]`
  produces exactly158 with truncation and189 with nearest rounding. An unused
  scale2 expert detects incorrect expert selection. Independent activation-byte
  and weight-dot controls check those exact outputs.
- Two independent CPU backends with default-off/enabled states, interleaved
  execution, explicit disable/re-enable, and a cached plan that retains its
  setting after the owning backend is reconfigured. No model file is loaded.

These C++ controls are prepared, **not compiled or executed**. Four stdlib
source controls pass: both old converter bodies are preserved; the new converter
changes only the rounding expression; the global type-trait table is preserved;
and the only CrispEmbed production source addition is the scoped encoder gate.
No local compiler, build, inference, framework or model batch has run.

The original890 trace guard remains unchanged and must reject this candidate.
A new versioned A/B consumer must allow only the reviewed source delta, rehash
all frozen fixture/model/array identities, preserve initial evidence, and run
unset/empty/0/1 paths in isolated modules. Legacy outputs must retain initial
anchors. Candidate ordinary/dump neutrality, complete stages and actual final
outputs must satisfy unchanged maximum absolute0.002/minimum cosine0.99999 gates.
A controlled dot or successful weight-free test alone cannot qualify the network
or identify its earliest failing operator.


The registered metadata phase `successor-wasm-rounding-controls` runs the separate
[weight-free hosted workflow](../.github/workflows/successor-wasm-rounding-controls.yml).
It verifies the exact declared fork/file identities before compiling only the
operator executable with Emscripten6.0.2, processor `wasm`, two build workers and
a25-minute ceiling. Actual Node/WASM execution must emit all five converter/
backend control records, including both storage types and both matmul paths.
Reports retain source identities and compiler/runtime logs even on failure;
no model file is downloaded. The generic FFN unit fixture uses the existing
mainline repair that distinguishes a valid frozen recipe from current-source
rejection. The historical runtime preflight itself remains unchanged.
