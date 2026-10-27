# SciFact retained tokenizer-only diagnosis

This source preparation adds a zero-input reusable workflow,
[scifact-tokenizer-diagnostic.yml](../.github/workflows/scifact-tokenizer-diagnostic.yml).
It does not authorize new embeddings, a runtime correction, or quality qualification.
The caller must be reviewed and registered separately.

The immutable panel [40375626336](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336)
at `e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902` completed all eight original shards
and froze all 5,483 original rows. Native jobs
[121891185945](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336/job/121891185945)
and [121891185906](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336/job/121891185906)
failed the actual text-ID guard on `d:21676556` (236 native versus235 original IDs;
first difference234) and `d:104143831` (192 versus190; first difference93).
The six later native jobs were cancelled while queued and executed zero steps.
Those cancellations are unexecuted work, distinct from the two executed negatives.
The parent run and exact completed job states are checked before retained input access.

The two small negative reports contain original/native IDs but omit literal text and
pieces. The new hosted job reads only selected metadata from the exact preparation
[40372423343](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40372423343),
checks the original freeze and negative ZIP service identities and bytes, and retains
the preparation's five attribution notices. It fetches only the pinned original
tokenizer JSON, never model weights or original vector shards. The prepared ZIP
also contains historical preflight arrays; this diagnostic does not read those members.
The source/license attribution remains the
[frozen SciFact notices](https://github.com/CrispStrobe/CrispEmbed/blob/e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902/docs/ultrabert-scifact-notices.md).

The original backend is `tokenizers==0.23.3`, confirmed by the executed original
shard's hashed environment report. Only that SHA-pinned binary wheel is installed,
with no dependencies; `Tokenizer.from_file` uses the local Rust backend and does
not invoke Hub or framework code. Every frozen original clipped ID/mask, full
length and full-ID digest must replay exactly before the native tokenizer runs.
The query prefix, literal bytes, order, BOS0 and right truncation to512 remain unchanged.

A standalone executable compiles only the exact historical BPE source/header
closure, with one test-only const accessor exposing the unchanged pretoken splitter.
No production file, shared library, GGUF, C ABI or default changes. Its value-parsed
`CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC` gate rejects unset/empty/0 before input
access. Ordinary-versus-observed IDs must match for every row. It reports clipped
and full-ID parity for all300 queries and5,183 documents, then requires both
executed negative arrays to reproduce exactly.

For the two failing rows, reports retain literal text, character/codepoint/UTF-8
spans, original Rust pretoken pieces and offsets, captured native splitter strings,
original/native ID-to-published-piece mapping and derived byte spans. Derived spans
are explicitly distinguished from backend offsets. Python's Unicode data version
is recorded for its descriptive categories; those labels do not establish the
Rust regex's Unicode tables or the cause of a mismatch.

A completed diagnostic may contain ID failures. Its report always declares
`qualification:false`, `embeddings_authorized:false`, and no new embeddings.
Review the earliest actual pretoken boundary before proposing a gated correction.
All5,483 frozen IDs must pass before future embeddings; original vectors and strict
retrieval/numerical gates must remain unchanged. Compile/inference/tokenizer batches
run only on hosted CI, bounded to20 minutes and one tokenizer worker. Small source
and stdlib controls are the only local validation for this preparation.
