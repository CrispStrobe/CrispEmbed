# UltraBERT Unicode pretokenizer candidate

This default-off alternative addresses the actual tokenizer-only diagnostic
[40424357191](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40424357191)
at`c86fc007cb247708e99266f777cc2bc9cadb2266`. Its original replay passes all5,483
frozen ID/mask/full-hash rows, while the historical native path has24 clipped
and25 untruncated mismatches. The [baseline](ultrabert-unicode-pretoken-baseline.json)
pins the exact review/archive/report/native-row and literal input-packet identities.
The two initial executed failures and six unexecuted cancellations remain preserved;
no original embeddings, source gold, numerical budgets or frozen producer branches
are modified.

Actual codepoint/pretoken evidence identifies the earliest boundary difference.
For`d:104143831`, original pretoken85 is` (“` followed by`caustic`, while the
legacy native splitter produces` (` then`“caustic”`. For`d:21676556`, original
pretoken81 is` ‘` then`stem`, while legacy produces` ‘stem`. The old generic byte
classifier treats these non-ASCII quotes as letters. This establishes the token
boundary defect; it does not establish embedding or retrieval parity.

`CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN` is read with`core_env::on()` only when the
UltraBERT tokenizer context is configured. Unset/empty/0 keep the existing path.
The mode is stored per tokenizer, not read globally during encoding, and a generic
vocabulary reload clears the new opt-in. The profile and model-free probe use the
same [configuration helper](../src/ultrabert_tokenizer_mode.h). Generic GPT2 and
CLIP contexts retain their old behavior even when the environment flag is1.
The old GPT2 and CLIP function bodies are preserved, with a scoped dispatch added
before the old GPT2 body. No default or public tokenizer API behavior is promoted.

The new branch transcribes the same GPT2 alternation on codepoint boundaries:
case-sensitive ASCII contractions, optional single literal ASCII space before
letter/number/other runs, then greedy whitespace with one-codepoint backtracking
before a nonspace and the final whitespace fallback. It does not normalize text.
Existing`core_unicode` predicates supply L/N/White_Space from the unchanged UCD14
table. Selected exact property controls include curly quotes, combining marks,
Arabic/Roman numbers, nonbreaking/em spaces, format characters and an unassigned
codepoint. Broader equivalence across Unicode versions is unqualified; full
original tokenizer parity must be measured on the frozen corpus before embeddings.

The new zero-input reusable
[workflow](../.github/workflows/ultrabert-unicode-pretoken.yml) leaves the historical
diagnostic workflow and its guards untouched. It authenticates the completed
actual diagnostic service/source, review ZIP/report, and remote metadata ZIP.
The latter supplies every historical native ID array under its exact raw SHA.
It reuses the exact preparation, original freeze and tokenizer metadata through
the historical helper. No weights or original vector shards are read. Historical
preflight arrays that happen to be inside the retained preparation ZIP are not read.

The candidate builds only the standalone tokenizer on hosted CI, with private
observation accessors inserted into a test-owned copied header. Weight-free C++
controls cover14 [authored cases](ultrabert-unicode-pretoken-cases.json), UTF-8 byte
preservation, model-context A0/B1/A0 interleaving, generic/CLIP isolation and reload
isolation. The actual original Rust backend must independently agree with every
authored expected split; a failed authored case is retained as rejection, not
redefined to fit the candidate.

All5,483 original clipped IDs/masks, full lengths and ID hashes replay before the
A/B. The exact packet SHA freezes literal inputs, vocabulary and merges. Candidate
unset/empty/0 must reproduce every historical native ID/full-ID/pretoken record
and the identical24/25 mismatch sets. Gate1 must pass every frozen clipped ID/mask,
full length and full-ID digest. Both initial rows must additionally have exact
original pretoken strings. Ordinary/observed token IDs and masks must remain exact.
A failure retains raw rows plus small mismatch identities and partial receipts.

The new source contract pins candidate files, unchanged historical helpers/core
files and the historical-only attribution document. Original input joining,
query prefix, BOS0, attended mask semantics and right truncation at512 are unchanged.
The backend wheel remains the actual original`tokenizers==0.23.3` with its exact
SHA, installed without dependencies. No framework or embedding import is allowed.

Local validation consists only of source/stdlib controls and formatting. Actual
compilation/tokenizer execution stays hosted, with one worker and a25-minute job
bound; no dispatch is included. Even a passing token A/B keeps
`embeddings_authorized:false`: the parent must review its receipt before preparing
new embeddings, preserve the original vectors, and retain unchanged numerical and
retrieval acceptance gates. This candidate does not qualify a model or release.
