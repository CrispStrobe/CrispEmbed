# LightOn4B32-layer composition and retained attribution

Draft PR150 stacks PR149 (`a657860f5`) and both tested block components. The
separate default-off `lighton4b_decoder_hidden_cpu` path enforces32 layers in the
original LLLF schedule:24 grouped-linear and8 full-attention layers. It carries
independently owned tagged cache vectors at each layer, rejects inconsistent
cache types/counts/token lengths, treats previous caches as immutable and applies
the final zero-centered norm. Callers retain borrowed weight lifetimes and the
correct cache/weight association. No embedding read, vocabulary projection,
GGUF/profile admission or recognition is exposed.

`CRISPEMBED_LIGHTON4B_DECODER_COMPOSE` and the four existing block/grouped/OCR
gates must all be enabled. Historical decoder/attention/linear sources and the
tested block headers remain unchanged. The exact streamed projection component
at `e42de7a3c` remains separate; incorporating it here would change frozen source
closures. This path deliberately returns hidden states and final norm only.

## Initial bounded composition is numerically negative

Hosted [41979168300](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41979168300)
at `bf6a7f930df72b79e80189356abe68f3cb29c2ec` runs the genuine original
Transformers5.5.4 TextModel forward with `inputs_embeds`. It constructs32 distinct
original layer modules/caches but assigns exactly two shared synthetic full-width
weight sets, checking storage sharing explicitly. The unused embedding stays on
the meta device. This avoids allocating checkpoint-like32-layer parameter storage;
it is not the actual checkpoint. All26 synthetic parameter hashes are retained.
The original runs first; native prefill/cached states never consume source caches.

The CPU/F32 batch1 panel covers prefill2, cached1 and full-prefix3 tokens. All347
native controls pass and290/291 unchanged numerical comparisons pass. The sole
strict failure is `prefill.cache.21.recurrent`, minimum row cosine0.999980919011239
against the unchanged0.99999 requirement. Its maximum absolute error is
1.0477378964e-7 against state magnitude0.0127160493. The worst row1326 has reference
maximum9.5542183942e-8, L2=2.2861582964e-7 and maximum error5.6026081552e-9.

Numerical drift is already present before this failure: prefill layer0 hidden max
error3.2186508179e-6 against1.789220691; layer20 error9.6082687378e-5 against
24.06994820; layer31 error1.8072128296e-4 against34.58509827. Final norm errors
are1.9729137421e-5 prefill,1.8835067749e-5 cached and2.2113323212e-5 full-prefix.
Passing hidden/norm comparisons do not erase the failed cache gate. The initial
301,841,623-byte artifact remains remote, with raw arrays and every comparison.

## Exact retained-input attribution

Hosted [41980636649](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41980636649)
at `fbab5cf9f` rehydrates that exact artifact remotely under read-only Actions
permissions. It verifies the frozen source records and regenerates all26 synthetic
parameter hashes exactly. It executes only original/native block21 on the
retained original or native layer20 output, plus a test-only source-normalized
GDN boundary. Neither32-layer decoder forward is repeated. Source cache values
are comparison targets, never native inference inputs.

All six frozen original/native hidden/conv/recurrent anchors are byte-exact.
All22 same-input block-stage comparisons and four source-normalized boundary
comparisons pass the unchanged budgets. Same-input recurrent maximum errors are
1.5832483768e-8 and1.3038516045e-8, with minimum row cosines0.9999998702210342 and
0.999999844517905. Source-normalized recurrent cosines exceed0.9999999849.

Changing only the retained input to the original block21 reproduces a strict
recurrent rejection: max1.0617077351e-7, cosine0.9999773186981167. Accumulated
upstream hidden-state drift is therefore sufficient to cause the tiny-row cache
failure; the controlled same-input tests do not identify a local grouped-GDN
implementation mismatch. This attribution does not locate every contributing
upstream operator or repair the complete decoder gate. No budget, precision,
algorithm, frozen reference or default changes.

The [public review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-decoder-compose-attribution.json)
verifies18 exact source digest records and preserves both complete reports,
291 comparison rows,26 weight records,347 native controls,22+4 replay comparisons
and six bit-exact anchors. Raw cache arrays remain in the hosted artifacts. The
review checks source/receipt/anchor coverage; it does not claim an independent
local reanalysis of all remote vectors. A transient PR control failure41980642106
came from prepending a selector to a literal-guarded workflow condition; the
repair appends the same exclusion after existing selectors, preserving the old
security control and runtime behavior.

The complete synthetic decoder remains unqualified by its original strict gate.
Checkpoint-backed stages, F16, long contexts, vocabulary logits, frontend and
actual decoded text/EOS/quality remain open. No actual full-model inference,
format5 recognition admission, alias, release or default promotion follows.
