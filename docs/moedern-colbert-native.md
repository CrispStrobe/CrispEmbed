# Prepared MoEdern 1.5M native token-ID experiment

This branch prepares an unexecuted CPU/F32 prototype. It does not add a public
model alias, text tokenizer, C ABI, binding, root build target or release asset.
The original producer at[790f7b1d](https://github.com/CrispStrobe/CrispEmbed/commit/790f7b1d0372c12d1807dfb2fc7fc3af20a3c445)
remains unchanged. The separately versioned
[native contract](moedern-colbert-native-contract-v1.json) deliberately leaves
the reviewed original run/artifact/receipt/conversion pins empty. Its guard
rejects before installation, build, payload access or native output. An actual
original capture must be independently reviewed and its immutable identities
filled explicitly before executing this lane. No model or native result is
claimed by source preparation.

The [original source audit](moedern-colbert.md) pins all three publisher sizes,
their Apache-2.0 source/configuration/weight metadata and the author ggml/MIT
provenance. This implementation independently follows the inspected mathematical
contract. It does not copy the author's global weight storage or process-exit
error handling. Original source/license notices are retained with the producer
artifacts. The native prototype accepts only the original-first lossless F32
conversion's explicit1.5M profile:32 exact tensor names and shapes, hidden128,
factorization16, two blocks, two query heads/one KV head, head width64, eight
routed/top-two experts, one shared expert and the128-dimensional Dense head.
Wrong metadata types/constants, extra/missing tensors, non-F32 data, nonfinite
weights and oversized files reject. It is not a generic architecture detector.

## Token-ID API and ownership

The [model-owned header](../src/moedern_colbert.h) exposes `init`, `free` and
`encode_tokens(ids, supplied_mask, role)`. Both initialization and encoding
require value-parsed `CRISPEMBED_MOEDERN_COLBERT`; unset, empty and literal0 are
off. Every context owns its CPU backend and complete weight allocation. Graph
arenas are local to operations, outputs are copied before destruction, errors
belong to the context, and a failed encode clears its output. There is no global
model pointer or retained observer. Contexts serialize their own encodes; callers
must not free or reenter a context from its synchronous observer.

The declared subset is batch1,3–512 IDs, CLS2, an explicit query30000/document30001
marker at position1, and SEP3 as the last real token. Masks are binary and
right-padded. Queries have exactly32 positions; padded expansion positions carry
MASK4. Literal special tokens inside the remaining sequence are preserved.
Token IDs come from the authenticated original capture; no text tokenizer or
prefix normalization is implemented here.

Keep three masks separate:

| Mask | Meaning |
| --- | --- |
| Supplied | Exact original wrapper input, including query expansion zeros |
| Effective | Query rows attend all32 positions, matching the custom backbone; document rows retain supplied attention |
| Scoring | A separate helper includes query expansion and removes configured document punctuation IDs |

The encoder returns every `S×128` vector, including positions later excluded
from scoring. The independent scoring resolver reads the actual frozen tokenizer
vocabulary/added-token IDs and saved skiplist, matching exact token-to-ID lookup;
it does not infer a policy from output masks. `select_rows` and the native
`maxsim` helper run separately. MaxSim sums query-row maxima over document dot
products. An empty scoring side or nonfinite input rejects.

## Mathematical path and source-only limits

Factorized embedding uses actual ggml `get_rows`, projection and LayerNorm.
Each block applies pre-RMSNorm, bias-free GQA projections, NeoX half-rotation RoPE
at theta160000, noncausal attention and residual; then RMSNorm, softmax/top-two
routing, `1e-9` selected-weight renormalization, GELU-erf gated routed/shared
experts and residual. Final RMSNorm, the separate Dense projection and L2 with
epsilon`1e-12` produce all-position vectors. The two-layer1.5M profile has no
convolution or active sliding window under512 tokens.

CPU projection, norm and expert views reuse the pinned ggml implementation.
Attention, expert reduction and GELU-erf follow explicit F32 mathematics. Reduction
order, RoPE/transcendental functions and Torch's actual CPU SDPA backend still
require measured comparison; source agreement does not establish numerical
parity. Exact router ties currently reject with an explicit unqualified-scope
error, rather than claiming that ascending IDs reproduce Torch top-k ties.
The prototype provides no quantization,27M/115M/convolution, GPU, batching,
CoreML, representative retrieval-quality or speed qualification.

## Prepared executable validation

The [standalone test build](../tests/moedern-native/CMakeLists.txt) links only this
model, shared GGUF loading and the pinned CPU ggml, without changing root CMake.
The [probe](../tests/moedern-native/probe.cpp) snapshots embedding/projection/norm,
Q/K/V, rotated/repeated heads, actual native attention output, router logits,
top-two indices/weights, shared and residual outputs, final norm, Dense and L2.
Attention probabilities and routed-only expert sums are labeled native-only
diagnostics because the original capture does not expose those primitive arrays.
Snapshots are synchronous copies while their owners are alive.

Each actual case requires ordinary/dump/repeat vectors and masks to be bit-exact,
then checks recovery after observer rejection and literal0, and reuses the first
context after a second context is destroyed. The standalone control executable
also covers role/mask/scoring/MaxSim negatives. Hosted synthetic GGUF fixtures
exercise valid typing and missing/unknown/wrong-shape/nonfinite/non-F32/wrong-profile loader failures;
these are not reference model weights or quality evidence.

The [hosted replay](../tools/moedern-colbert/native.py) authenticates the reviewed
service run, both artifact ZIPs, original receipt, checkpoint, every original
stage member and the lossless conversion before building or forwarding native
weights. It preserves the frozen producer's ten inputs and three score pairs.
Both short full-capture cases compare matching stages in dependency order;
all ten compare actual IDs, all three masks, every final row and scoring slice.
Native MaxSim runs on the actual selected vectors and is checked independently.
Maximum absolute error`0.002`, minimum row cosine`0.99999` and MaxSim error`0.002`
remain unchanged. Numerical failures retain the complete completed case/stage
panel and first captured divergence with magnitudes; they do not become a pass.
Structural/nonfinite failures remain rejected and preserve partial outputs/logs.

The [reusable workflow](../.github/workflows/moedern-colbert-native.yml) is not
registered in the shared dispatcher and has no automatic push trigger. It remains
blocked by the pending original dependency contract. All builds, synthetic
execution and actual model replay belong on hosted CI. Source preparation ran
only Python standard-library controls, parsing and formatting locally. Fetch
only the small review artifact later; preserve array ZIP/member hashes and expiry
remotely. No app pin, generic default or release qualification follows from this
bounded token-ID experiment.
