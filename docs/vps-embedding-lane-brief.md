# Historical embedding language-coverage brief — 8 August 2026

This technical brief is retained for background. Use [PLAN](../PLAN.md) and the
[executable handoff](next-agent-handoff.md) for current status, owners and
acceptance; later tokenizer/language fixes may supersede the tasks below.
Machine-specific setup and access notes are kept outside tracked documentation.

Work in an isolated worktree, claim one bounded lane, and run builds, suites,
model/browser batches on hosted CI or authorized remote GPU infrastructure.
Inspect shared-host load, memory, swap and free space before costly work; keep
large models/artifacts remote. Native/Python intermediate and final-output
comparisons remain mandatory before runtime/default promotion.

## Why this work exists

Issue #44 asked "which model is best for Japanese?". It was first answered for
the OCR lanes; the asker may have meant the **embedding** models. That exposed
an untested axis: `docs/LANGUAGES.md` was OCR-only, every embedder parity test
uses English-only text, and the registry's embedder language strings are
upstream model-card claims. Japanese is now verified for **8** embedders
(`99f39f64`); this package finishes the job.

**The finding that shapes everything below:** English-only models do not
degrade gracefully on Japanese — `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
return **bit-identical vectors for two different Japanese sentences**, so a
naive paraphrase check scores them 1.0000 while they emit a constant. Any
harness you write must be able to catch that, or it is measuring nothing.

## Tasks

### E1 — finish the Japanese matrix (7 untested aliases)

`multilingual-e5-small` / `-base` / `-large`, `paraphrase-multilingual-MiniLM-L12-v2`,
`granite-embedding-278m`, `granite-embedding-97m-r2`, `granite-embedding-311m-r2`
(+ the GTE-v1.5 multilingual entries). They are missing only because they were
not cached on the Mac.

```bash
python tests/embed_language_eval.py ./build/crispembed <models-dir> out.json
```

Add each to `MODELS` in that harness, run, extend the table in
`docs/LANGUAGES.md`. **Keep the English-only models in the list** — they are
the negative control; a language test everything passes is measuring nothing.

⚠ **The e5 family has a prefix contract** (`query: ` / `passage: `). The
harness currently runs prefix-free, which is fair for *relative* comparisons
but must be stated per row. `--prefix` exists; consider running e5 both ways
and reporting both.

### E2 — rerankers on Japanese (an entire untested lane)

`bge-reranker-v2-m3` and `jina-reranker-v2-base-multilingual` ship multilingual
claims and have **never** run on non-English text. Reranking is first-class for
Japanese RAG.

This needs its **own** harness — the embedding eval does not apply. Shape:
Japanese query + relevant Japanese doc + irrelevant Japanese doc; assert the
relevant doc outranks, and report the score gap (not just the ordering).
Include an English-only reranker as the negative control. Cross-check one case
against the HF model to make sure a "win" is not an artifact of score scaling.

### E3 — languages beyond Japanese

The matrix is JA-only. Extending is a five-line `TEXTS` edit plus fixture
authoring. **Arabic and Korean are the highest-signal next picks**: different
scripts, different tokenizer failure modes than kana, and Arabic adds
RTL/normalization questions kana does not. Aligning with the languages the OCR
lane already ships (deu fra spa ita por nld rus ara jpn kor chi) makes both
halves of the matrix comparable.

### E5 — WordPiece CJK path is not HF-faithful

On the 30k uncased WordPiece models, two different Japanese sentences collapse
to one token sequence; HF's reference tokenizer does not
(`[UNK],[UNK],上,て,[UNK],っ,##て,##い,##る,。` vs
`[UNK],[UNK],か,[UNK],て,##い,##ま,##す,。` — HF splits words at CJK ideographs
and strips dakuten under NFD, so kana runs still decompose into subwords).

Impact is confined to English-only vocabularies fed CJK — every multilingual
embedder we ship is SentencePiece/XLM-R and unaffected (verified, not assumed)
— so this is correctness hygiene, not a shipping fire. Work: diff our
WordPiece pretokenizer + subword loop against HF `BasicTokenizer` on a CJK
corpus. **A guard belongs in `tests/test_bert_pretokenize.cpp`**, which today
asserts the kana-run-stays-whole behaviour without checking what WordPiece then
does with it — the test encodes half the law.

Tokenizer-only comparison needs no torch forward, so it is safe here (local
torch is known-broken for BERT forwards; use ONNX Runtime if you ever need a
reference *forward*).

### E6 — make the silent failure loud (highest user value)

Today a user can point an English-only embedder at Japanese and get confident,
arbitrary retrieval with no signal. The runtime knows enough to warn: compute
the unknown-token ratio at tokenization and emit a **one-shot** stderr warning
past a threshold, e.g.

```
warning: 87% of input tokens are [UNK] — this model's vocabulary does not
cover this script; see docs/LANGUAGES.md for models that do
```

Once per input, no hot-path cost, env-gated so it can be silenced. This
converts a whole class of wrong-model-for-the-language bugs from silent to
obvious.

## Acceptance (same protocol as every other lane)

- Numbers, not verdicts: report the actual cosines/margins per model, and state
  the negative control's result in the same table.
- Any new gated path ships **opt-in** until it wins on both quality and speed;
  never delete a working path.
- `tools/format.sh --fix` on changed C/C++ before merge.
- Findings land in `docs/LANGUAGES.md` **and** `PLAN.md` (E-items), with the
  fixture and the command to reproduce.
- ff-merge to `main`, delete the branch, remove the worktree, clear your board
  row.

## Explicitly NOT on this box

Anything needing CUDA (the conv-ab kernel v3 dbnet arm, the CUDA-rec
0-results bug) or Metal (the PP-OCR rec width-batch profile — another agent
owns it). Those are in the OCR round's queues in `PLAN.md`; leave them alone.
