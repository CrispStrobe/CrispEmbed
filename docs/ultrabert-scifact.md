# Full SciFact original-first audit contract

The separately versioned [protocol](../tests/regression/ultrabert/scifact-protocol-v1.json)
freezes5,183 corpus documents and all300 BEIR test queries, prompt/CLS/L2/512-token
semantics, exact IDs, eight shards, and acceptance before outputs. The owned
32-row retrieval smoke remains unchanged. This is a bounded representative
SciFact task audit; publication, release, pairs and historical full-stage
numerical qualification remain separate.

Run the default `prepare` phase of
[the hosted workflow](../.github/workflows/ultrabert-scifact.yml) first. It checks
all author corpus IDs/title/abstract joins and all1,109 BEIR query IDs/texts,
compares both author evidence IDs and cited document IDs to canonical test
qrels, and retains any initial mapping difference. Canonical pinned BEIR qrels
remain the frozen evaluation target even when author-derived relevance differs.
Text/ID mismatch fails before model output. The author immutable S3 object,
actual SHA256s, every raw dataset/notice member, all prepared inputs, gold,
lengths and exact clipped IDs are frozen before loading weights or vectors.
[Applicable attribution/terms](ultrabert-scifact-notices.md) are retained with
remote copies, including the BEIR CC BY-SA declaration and upstream terms.

The original-only preflight uses the actual pinned SentenceTransformer6.0.1
wrapper, custom model source, CPU/F32/eager and raw CLS plus actual wrapper L2.
It tokenizes all rows and times a deterministic length panel, checks observer
neutrality, and estimates the worst original shard with a declared safety
factor. This phase does not authorize the full panel. Review its actual source,
dataset freeze, mapping differences, length distribution, timings and receipt
SHA256 first. The full `panel` phase requires that explicit preparation run and
reviewed receipt hash, rejects changed source/protocol/input bytes, and executes
all eight original shards before freezing their exact vector artifacts. Only
then may the eight native shards start. Jobs are hosted, CPU-thread-bounded,
limited to two concurrent shards and240 minutes each; no speed claim is made.

Native text execution keeps the existing experimental architecture gate and adds
`CRISPEMBED_ULTRABERT_TRUNCATE=1` for single-text right truncation at512 tokens,
including attended BOS0 and any explicit query prefix. Unset/literal0 retain
overflow rejection. Byte/reserved-surface exclusions and failure recovery stay
in force. The hosted metadata-only tokenizer control checks unset/0, recovery,
BOS/prefix/exact511 text-token retention, repetition and exclusions. All5,483
actual task rows must match original IDs/masks; none may be dropped. Public pair
inputs remain excluded.

The scorer preserves raw/normalized vector max0.002 and cosine0.99999, requires
NDCG@10, MRR@10 and Recall@100 degradation no greater than0.001, and at least99%
top1 agreement. It retains all misses, relevant ranks, top100 scores/margins,
all vector errors and full float64 cosine score matrices remotely. Every
numerical/quality failure returns nonzero and stays in its actual report;
positive CLS quality cannot waive183 F32/1,294 F16 stage failures or strict
reference-replay failures. No default change or model publication follows.

Only source and lightweight stdlib controls have run when this contract is
introduced. Actual dataset/preflight, all16 shards and independent artifact
review remain pending. Retain report and raw artifacts for90 days, then preserve
verified ZIP/member bytes remotely before the returned service expiry.

The author archive's canonical claim inputs are exactly`claims_train.jsonl`,
`claims_dev.jsonl` and`claims_test.jsonl`, as documented by its pinned README.
Five additional cross-validation train/dev pairs repeat these IDs; retain their
source identities separately and never concatenate them into the canonical
inventory. Duplicate canonical IDs remain an error. Initial40346793408 preserves
that adapter rejection before any model output in the retained result receipt.

Preparation40353858454/57aab2cf next rejects author document10068634 before
any tokenizer, weight or model output. Stored corpus columns differ; the
initial failure does not yet identify the changed field/codepoints. Preserve
[the initial corpus rejection](../tests/results/ultrabert/scifact-initial-corpus-rejection.json).
The strict v1 author/BEIR equality gate remains unchanged. Hosted diagnostics
now retain all mismatch IDs/counts and eight bounded first-difference examples,
without applying any candidate text transformation. They independently compare
pinned pre-migration BEIR JSONL with Parquet to locate where divergence entered.

The [historical HF loader](https://huggingface.co/datasets/BeIR/scifact/raw/984eed826375f18d27936c4a32bf0f8491e3f414/scifact.py)
(SHA f839e4910524fd6577117a4dabd4133168eb50603a13e333666646053adf902d)
and [BEIR loader](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/datasets/data_loader.py)
(SHA bcb39af11b73861eda44f3083b399ba78c46ef11c25d56d8e195572af0be336f)
preserve stored strings. The [retrieval input join](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/models/util.py)
(SHA90585957246ac5a6baf4e7e0a13b01821b4ed0a0544136221ce1a1c995caf4ca)
strips outer whitespace after joining title/text; this does not explain a
stored-column mismatch. No normalization recipe is assumed from the
[format-migration announcement](https://huggingface.co/datasets/BeIR/scifact/discussions/3).
A successful preparation also copies its data-freeze/length summary and notices
to the small reports, so independent review needs no local corpus payload.

## Canonical BEIR v2 — original-first protocol after actual provenance review

[Data attribution40364838418](../tests/results/ultrabert/scifact-data-provenance.json)
atbf2f9e48 verifies exact legacy-JSON/Parquet IDs and stored columns for all5,183
corpus rows and1,109 queries. The current author archive differs in1,055 abstract
fields, with titles unchanged. The first eight bounded contexts illustrate
internal whitespace differences; this is not a claim that a guessed cleanup
reconstructs every field. Strict v1 remains rejected, with no model outputs.

The distinct [v2 protocol](../tests/regression/ultrabert/scifact-protocol-v2.json)
uses the actual pinned canonical BEIR columns literally. It changes no rows,
qrels, model precision,512-token policy or vector/ranking budget. Author-column,
claim, evidence and citation differences are journaled rather than relabeled
identical. Its runner must retrieve the exact reviewed negative v1 report ZIP
and verify both selected member SHAs before accessing data/models. It then
repeats exact legacy/Parquet comparison on the current pinned payloads.

The actual published241-byte`sentence_bert_config.json` is now downloaded and
pinned separately from the nine historical converter source files. This retains
the real text-forward/output mapping without changing GGUF converter metadata.
Pinned SentenceTransformer6.0.1 preserves input whitespace and prepends prompts
literally; its512 override actively changes the tokenizer limit. No historical
wrapper stripping assumption is adopted. Before the first original forward,
all5,483 rows must match the frozen IDs/masks through both actual
`preprocess(raw,prompt)` and`tokenize(literal-prefix+text)` paths. All-row controls
include a last-row mismatch; runtime proof remains an actual hosted gate.
Preparation still requires independent receipt/timing review before the complete
eight-original→freeze→eight-native panel. No publication/default/app promotion.
