# Frozen successor WASM source attribution

The first packaged MiniLM browser consumer failed maximum absolute0.002 and
cosine0.99999. Its [initial receipt](../tests/results/packaging/successor-wasm-initial.json)
retains the actual failure. This lane traces that same immutable model, sentence,
native archive and browser archive. It changes no production source or archives.
A successful attribution workflow still leaves qualification false.

Run the reusable [successor-wasm-trace workflow](https://github.com/CrispStrobe/CrispEmbed/blob/c18f7453912d9793edde3f3076d636ce127a69c9/.github/workflows/successor-wasm-trace.yml)
from a reviewed source commit; it has no retargetable model/reference inputs.
The metadata caller can register phase`successor-wasm-trace` with
`uses: ./.github/workflows/successor-wasm-trace.yml`. The job runs on hosted Ubuntu,
has a60-minute ceiling, and limits the diagnostic build to two workers. No local
build, inference, framework or model download is needed for source review.

The driver requires successful native producer40339796226 and WASM
producer40339800364 at40a82158146faf0cd44de2a4ee7056e7b0c6a58c.
It uses the exact production/build/binding Git object closure and archive/member
checks from the successor consumer. Initial consumer40354251257 must retain
source49161f608af811154bfeb816ac38caea34224cae, its failed overall conclusion
and successful Linux desktop job. Fixture artifact12671621333 is bound by its
18,259,261-byte size and ZIP SHA256
`a67abb8439d2af274c175dad9cefb8ba9606bec9c4ba024cf568a7359f41c8bf`.
Before loading, the driver verifies its two members, native report SHA256
`6cdd426e0b71fcbe80000494645deeb75fa75f9d5028e06d29abcdb9ff8cf435`,
the packaged library hash and model SHA256
`77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2`.
Each GGUF tensor's actual name, type and dimensions are recorded. The model's
IQ4_XS filename does not imply that every matrix uses that storage type.

A disposable source worktree at the producer commit receives three exact,
recorded test overlays: an environment setter and bounded Q8_0 controls in the
embedding wrapper; writes of the actual pre-L2 and final pooling buffers;
and explicit diagnostic exports plus a two-worker build limit. The production
checkout and both packaged archives remain unchanged. The overlay diff and
before/after source hashes, compiler commands and diagnostic modules are retained.
The release uses`--no-simd`, but the pinned ggml CPU CMake target independently
adds`-msimd128`; the report preserves the actual compile commands.

The source-bound single sentence is`A small reproducible embedding consumer.`
The following controls must hold before stage comparisons are trusted:

- Packaged native1/2-thread ordinary and dump outputs are bit-exact to the
  initial native F32 vector. Contexts and environments use separate processes.
- Packaged browser ordinary, diagnostic browser ordinary and diagnostic browser
  dump outputs are bit-exact to the initial browser F32 vector.
- Actual final token IDs agree across native and diagnostic browser modes.
  Both public-entry and graph-upload token observations must occur exactly once
  and agree; missing, extra or differing log lines reject.
  The unpadded WordPiece path has the expected CLS/SEP boundary and an all-one
  attention mask. The mask is source-derived, not claimed as a runtime capture.
- All eight complete F32 stages exist with shape`[384,T]`: `emb_ln_out`,
  `attn_out_0`, `layer_0` through`layer_4`, and`encoder_out` (the final block's
  renamed output). Missing stages or nonfinite data reject.

The stage report records max/mean absolute error, RMS, both L2 magnitudes,
extrema, global and minimum token-row cosine, hashes, and the earliest unequal
and failing stage. A standalone test executable uses`crispembed_diff::Ref`;
independent stdlib parsing and metrics check its results. Maximum absolute0.002
and minimum row cosine0.99999 remain visible for every stage. Final vector gates
remain unchanged. Stage differences are attribution evidence, not waived parity.

WASM pre-L2 pooling is captured from the actual test-overlay buffer and checked
against sequential-F32 reconstruction. Native pre-L2 pooling is explicitly
**recomputed** from the complete packaged`encoder_out` and source-derived mask;
it is not a captured buffer. Its measured L2 reconstruction error against actual
output and the existing six-decimal raw-norm diagnostic are recorded. The native
library is not rebuilt for this first trace.

Test-only operator controls compare the linked WASM architecture Q8_0 converter
with the pinned`quantize_row_q8_0_ref` on zero, positive/negative fractional,
halfway and actual first-token`emb_ln_out` inputs. Full block bytes retain FP16
scale and signed quantized values. A same-weight first-output-channel dot is
included only if the actual first query weight is IQ4_NL`[384,384]`, whose CPU
trait uses Q8_0. This is one controlled dot, not a full operator or native-SIMD
cause proof. Reference`roundf` ties-away and native SIMD nearest-even must not be
conflated. No rounding alternative replaces production arithmetic in this lane.

Small review artifacts preserve reports and logs even on failure. Stage arrays,
actual pooling buffers and test diagnostic modules remain in a separate remote
artifact. No production weight files, registry aliases, app pins, tolerances or
release assets are replaced. A proven production fix requires a new producer
and fresh affected consumers.
