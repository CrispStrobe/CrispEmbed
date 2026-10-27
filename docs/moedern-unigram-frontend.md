# MoEdern1.5M Unigram frontend preparation

This separately versioned source lane starts from reviewed native-v2
`cec6b7f0dd3ea78bfefb8587b3638c8ab23f462f` and keeps all old native kernels,
producers, arrays, guards, generic tokenizers, root CMake and bindings unchanged.
The new internal `text_frontend` snapshots the default-off
`CRISPEMBED_MOEDERN_TEXT_FRONTEND` gate at load. It accepts explicit query/document
roles and returns IDs, supplied attention, effective backbone attention, scoring
mask and query expansion positions. There is no public C API, registry or new
model loader; public text-to-vector composition remains a later step.

Sources are the pinned
[1.5M checkpoint metadata at9d360b8b](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/tree/9d360b8b88ae11bea5784bf9012ff509fdd5c2da),
the unchanged successful original40412164041/receipt `b50ca1369f45...`, and reviewed
[native-v2 40441341206](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40441341206).
The tokenizer SHA is `7357662aa3be57e8fd953365a37f892dab4ee702b817e1efcef9f193a388aff0`.
Its original environment explicitly used tokenizers0.22.2. The source and
[tokenizer payload](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/blob/9d360b8b88ae11bea5784bf9012ff509fdd5c2da/tokenizer.json)
are Apache-2.0; the hosted task preserves the original LICENSE. The independent
C++ implementation follows the exact Rust0.22.2 Unigram algorithm: F64 scores,
strict greater-than path updates, unknown penalty `min_score-10`, UTF8 scalar
fallback and adjacent unknown fusion. It does not use the generic tokenizer's
F32 scores, byte fallback or XLM-R normalization.

The raw normalizer replaces double backticks/apostrophes with a double quote,
then applies two NFKD/StripAccents/char-Lowercase/ASCII-space-collapse passes.
WhitespaceSplit and Metaspace(always, split) are separate. All seven added-token
flags are preserved, including MASK's Unicode whitespace lstrip and Q/D's
trailing spaces. The hosted blueprint executes the exact AST-extracted
[`_register_prefix_tokens`](https://github.com/huggingface/sentence-transformers/blob/b0bf39968ba461c34b29372144794eff1be06f6d/sentence_transformers/multi_vector_encoder/model.py)
method before capture, using a Rust tokenizer facade. It records raw and effective
flags and actual registration actions: the runtime can re-register a normalized
role marker as a raw special token. Matching respects raw extraction before
normalization, then normalized added tokens, with no generic fallback.

Each role prepends the literal original marker. Queries truncate to32 including
CLS/SEP, pad with MASK, preserve supplied zeros, and set effective/scoring masks
to ones. Documents truncate to512 and filter the exact original vocabulary
skiplist only from scoring; attention remains separate. Only batch one is
implemented. Invalid roles, malformed UTF8 and text over64KiB reject, clear all
outputs, and permit subsequent valid calls. Failed reload clears context; gate
changes do not alter an existing context.

The zero-tensor component contains complete vocabulary, exact F64 score bytes,
raw/effective descriptors, source/config identities, skip IDs and Rust-generated
normalizer data. Source labels alone cannot authenticate arbitrary components:
a later caller must verify the actual reviewed component SHA before loading or
merging its metadata. The hosted job samples the exact Rust normalizer on every
Unicode scalar, retaining only changed mappings. Pair replacements precede that
mapping and ASCII-space collapse follows it. Explicit combining reorder/context,
Hangul, ligature/fullwidth, dotted-I/Greek, quote ordering, Unicode whitespace and
literal-metaspace sequences plus1024 deterministic compositions must reproduce
full-string Rust normalization before native code runs. Any counterexample stops
execution; no replacement table, generic fallback or original gold adjustment is
allowed. The mapping is tied to this exact Rust0.22.2 wheel and is not a promise
about another Unicode/runtime version or every untested multi-scalar sequence.

Metadata phase `moedern-unigram` invokes a25-minute reusable tokenizer-only task.
Before payload access it verifies exact new source closure and immutable SMALL
primary/native-v2 artifact identities. Only authenticated tokenizer/config/source
files and the exact original Rust wheel are downloaded; no Torch, Transformers,
Sentence Transformers, model weights, activations or forwards are installed/run.
The first ten tokenizer streams must exactly reproduce the unchanged original
IDs and all three masks. Twenty-four additional owned Unicode/literal/role/
truncation cases and normalization diagnostics are then frozen before C++ builds.
Two explicit owned synthetic tokenizers test Unigram ties, unknown fusion and
F64 distinctions; they are separate unit artifacts, never replacement checkpoint
metadata or primary targets.

The two-worker standalone build links GGUF metadata parsing only from ggml-base,
with all compute backends disabled. Ordinary/test-enabled ID and mask outputs
must be byte-identical. Gate/reload, input and semantic metadata rejection and
valid-before/reject/valid-after recovery controls run on actual compiled code.
Diagnostic methods exist only in the test build, with compiled ordinary-symbol
exclusion. Native normalized strings, pretoken strings and within-segment UTF8
byte boundaries compare exactly. Rust original character offsets are retained
with a separate label; no native original-source-offset API or parity is claimed.
All failures and partial outputs stay preserved, including original-only failures.

This remains source preparation until actual hosted evidence is reviewed. It
does not qualify public text-to-vector APIs,27M/115M, quantization, representative
retrieval, runtime defaults, release or application adoption. No new golden model
vectors or tolerated numerical drift are introduced.
