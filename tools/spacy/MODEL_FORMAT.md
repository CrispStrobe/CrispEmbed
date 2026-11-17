# German spaCy bundle contract, version 1

The only admitted source is the exact official `de_core_news_sm` 3.8.0 wheel
identified in [README.md](README.md). This contract describes exported data,
not a claim that every exported capability has a native implementation.

## Identity and admission

`manifest.json` binds the source wheel's size/SHA-256, source revisions,
reference versions, converter hash, component order, numerical semantics,
39 semantic tensor records, source-file inventory, and each output file's
size/SHA-256. The manifest is deterministic: no timestamps, host paths, or
random identifiers are embedded. Applications must trust an externally pinned
manifest/model hash; a checksum read from an untrusted file is not its own
trust anchor. Converted bundles cannot silently substitute a newer model.

The GGUF architecture is `spacy_de_core_news_sm`, with `spacy.format_version=1`.
Required typed metadata binds model/version, source/resources SHA-256,
component/feature order, F32 precision, width 96, depth/padding 4, and
layer-normalization epsilon 1e-8. Native admission must reject wrong/missing
types or values, tensor count/name/shape/dtype mismatches, invalid file
extents, non-finite weights, and changed authenticated bytes before returning
a context. There is no permissive unknown-component or partial-success mode.

## Layout and inference semantics

All tensors are little-endian IEEE F32 in original C row-major order. The
GGUF dimension list reverses the source shape; payload bytes are not
transposed. Every tensor record includes both shape lists and its raw-byte
digest. Semantic names do not depend on Thinc's serialized node numbering;
the source node index/name/parameter is retained for audit.

| Semantic name | Source shape | Meaning |
| --- | --- | --- |
| `embed.{0..5}.E` | `[rows,96]` | Rows `[5000,1000,2500,2500,50,50]`; seeds 8–13 |
| `embed.mix.W`, `.b` | `[96,3,576]`, `[96,3]` | Concatenated embeddings → maxout |
| `embed.norm.G`, `.b` | `[96]` | Learned projection normalization |
| `encoder.{0..3}.maxout.W`, `.b` | `[96,3,288]`, `[96,3]` | Three-token window maxout |
| `encoder.{0..3}.norm.G`, `.b` | `[96]` | Learned normalization before residual add |
| `tagger.W`, `.b` | `[55,96]`, `[55]` | Fine POS scores |
| `morphologizer.W`, `.b` | `[671,96]`, `[671]` | Contextual morph/POS scores |
| `lemmatizer.W`, `.b` | `[1311,96]`, `[1311]` | Scores for class-to-tree mapping |
| `parser.reduce.W`, `.b` | `[64,96]`, `[64]` | Shared encoder → parser input |
| `parser.lower.W`, `.b`, `.pad` | `[8,64,2,64]`, `[64,2]`, `[1,8,64,2]` | State features with learned missing-token padding |
| `parser.upper.W`, `.b` | `[137,64]`, `[137]` | Ordered transition-action scores |

The six input features are `NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE` in that
order. The first four are exact uint64 values; the last two are Boolean IDs.
Thinc hashes each feature into four ordered uint32 buckets modulo its table
size and **sums all four rows**, including repeated bucket IDs. String hashes
and embedding hashes are distinct operations. JSON uint64 IDs are decimal
strings, never floating point numbers.

The encoder pads four zero rows on each side **once**, then those rows evolve
through all four residual layers. Windows use previous/current/next order.
Maxout selects the first maximum across three pieces. Normalization uses
per-row population variance plus 1e-8, then learned scale and bias. Classifier
scores are unnormalized affine outputs. Shared encoding runs once per document.

Parser feature order is `B0,B1,S0,S1,S2,L(B0,1),L(S0,1),R(S0,1)`.
The numeric scorer does not implement transition legality or state updates.
Those, sentence boundaries, and pseudo-projective reconstruction must be
qualified separately against step traces and final annotations.

Numeric comparison budgets are fixed before port execution: absolute
tolerance 2e-4 plus relative tolerance 2e-5, with exact class/action decisions.
Reports must preserve per-stage magnitudes/errors and every failing comparison.
These are development numerical budgets, not grammar-quality criteria.

## Native resources and decoding

`resources.json` preserves enabled order
`tok2vec,tagger,morphologizer,parser,lemmatizer,attribute_ruler`. NER is omitted;
the separately trained sentence recognizer is disabled. Parser boundaries
supply sentences only after parser qualification.

Tokenizer regex strings and exceptions come from the authenticated model.
Exception attribute names are explicit. NORM lookup tables preserve outermost
first precedence, including hashed lexical overrides; the fallback is the
pinned CPython Unicode `str.lower`. Frozen single-scalar lowercase tables
and character-class ranges provide native data. The lexical port must also
implement contextual final sigma and the source tokenizer/shape algorithms;
the existence of these tables does not establish lexical parity.

Ordered morph labels have separate authoritative morph/POS mappings and
overwrite/extend flags. Lemma classes map to 6,092 stored edit trees. Tree
substitutions preserve exact string IDs and resolved text; class index is not
tree index. Top-k is one, an inapplicable winning tree falls back to the
original token, and existing lemmas are not overwritten. The final two
attribute-ruler rules fix whitespace annotations and must not be omitted.

## Ownership, limits, and staged capabilities

The internal numeric interface borrows immutable admitted weights for the
context lifetime. Results own their arrays; trace callback views last only
for the callback. Consumers must serialize calls on a context and must not
free a context or result while it is being used. A callback may request
cancellation; cancellation publishes no successful partial result.

Version 1's engineering bounds are 32 MiB per bundle/model admission and
2,048 prepared tokens per document. They are not measured mobile budgets.
No hidden truncation, download, daemon, Python runtime, quantization, or GPU
fallback is permitted. Empty input produces explicitly empty numeric outputs
without pretending that nonempty inference ran. The original numeric API still
accepts prepared features. The additive text ABI now performs tokenization and
greedy decoding from counted UTF-8, with qualified development spans in original
bytes, scalars and UTF-16 units. It requires the separately authenticated
`resources.json` whose hash is already bound by this GGUF profile.

## Text extension without a bundle change

The original five bundle files remain byte-identical. The native source includes
a reproducible lexical supplement generated from the pinned CPython 3.12.14 /
Unicode 15.0 and spaCy 3.8.18 environment: reserved string symbols, contextual
lowercase properties and Python regex character classes. PCRE2 executes the
translated patterns with bounded work; its Unicode shorthand tables do not
substitute for these frozen properties.

Text inputs are limited to 1 MiB and 2,048 tokens, with no truncation. The parser
allows at most `8*n+32` transitions; edit trees have depth 64, 65,536 visits per
token and a 1 MiB output bound. Cooperative cancellation covers lexical loops,
numerical stages, parsing and dependency reconstruction. A failed call returns
no successful partial document. Owned immutable documents may outlive contexts,
and borrowed counted-string views expire when the document is freed.

The text ABI version is 1. Legacy numeric capability queries retain mask 3;
resource-aware context queries additionally advertise tokenizer/analyzer bits,
for mask 15. NER, other locales/profiles, app integration and mobile deployment
are outside this profile. See the source-bound
[development analysis summary](../../docs/validation/native-spacy-20261117/text-summary.json)
and the separate frozen acceptance protocol before broader qualification claims.
