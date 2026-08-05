# mxbai ContextPooler GELU A/B + two riders (2026-08-05)

Reference throughout: ONNX Runtime 1.25.1 (CPU EP) on the OFFICIAL repos' own
`onnx/model.onnx` (`mixedbread-ai/mxbai-rerank-{xsmall,base}-v1`), tokenized via
the repo `tokenizer.json` through `tokenizers` only — no torch forward anywhere
(local miniconda torch is disqualified for BERT-class forwards; see the G7c
summary). Fixture: 2 queries x 6 docs (the g7c Berlin set + an ML set), files
`query_q{0,1}.txt` / `docs_q{0,1}.txt`, reference scores `onnx_ref.txt`.
All native arms `--gpu-backend cpu`, run serially. Fresh conversions are
ollama-mode (matches the shipped `blk.*` naming).

## 1. erf-vs-tanh GELU (the assigned A/B) — real, tiny, now the default

Both configs say `"pooler_hidden_act": "gelu"`; HF `ACT2FN["gelu"]` is
erf-exact. The runtime used the tanh approximation.

f16, max |delta| vs the ONNX reference (6 docs):

| model | query | tanh arm | erf arm | improvement |
|---|---|--:|--:|--:|
| xsmall | q0 | 0.000205 | 0.000003 | 68x |
| xsmall | q1 | 0.000192 | 0.000002 | 96x |
| base | q0 | 0.000123 | 0.000009 | 14x |
| base | q1 | 0.000097 | 0.000008 | 12x |

The erf arm collapses the residual to f32-rounding noise — the tanh
approximation was the entire remaining f16 discrepancy. Magnitude is ~1e-4 on
scores of 1-6: Kendall tau vs reference is 1.000 in both arms on all four
pairs, and at q8_0 quantization error dominates by 200-1000x (erf not even
consistently closer there). So: correctness-clean, quality-immaterial.

**Decision (coordinator): default FLIPPED to erf-exact.** HF-exact, one
`std::erf` per hidden unit (no measurable cost), and — decisive — NO shipped
artifact carries pooler tensors today (see finding 2), so the flip changes
zero shipped-artifact outputs. `CRISPEMBED_RERANK_POOLER_GELU_ERF=0`
(value-parsed) restores the tanh approximation for A/B.

Three-spelling proof (agent run + coordinator re-run, both on the final
binary): absent == `=1` (erf); `=0` reproduces the old tanh scores to every
printed digit; pre-flip the agent proved absent == `=0` with `=1` differing —
both orientations of the gate verified against the same score tables.

## 2. UNASSIGNED FINDING: shipped mxbai GGUFs have NO pooler stage at all

The pinned `cstr/mxbai-rerank-{xsmall,base}-v1-GGUF` q8_0 artifacts (SHA-verified
against `model_hashes.h`: 901aa5f9…, 12ad024e…) carry no `pooler.*` tensors and
no `bert.pooler_act` KV — coordinator re-read the tensor lists independently.
HF scores `classifier(gelu(pooler(CLS)))`; the shipped artifacts score raw
`dot(CLS, w) + b`. G7c one architecture over (DeBERTa instead of BERT), worse
in degree:

| model | query | max abs delta | tau | ONNX order | shipped order |
|---|---|--:|--:|---|---|
| xsmall | q0 | 3.905 | -0.200 | 0,2,5,1,4,3 | 5,4,3,0,1,2 |
| xsmall | q1 | 4.573 | -0.733 | 1,0,3,4,5,2 | 2,4,5,3,1,0 |
| base | q0 | 5.847 | 1.000 | 0,2,5,4,1,3 | 0,2,5,4,1,3 |
| base | q1 | 4.363 | 0.600 | 0,1,3,4,5,2 | 0,1,4,2,3,5 |

Calibration destroyed (±0.3 vs a ±6 reference span) and xsmall's ranking is
near-inverted with the wrong top-1 on both queries. A fresh conversion with the
CURRENT main converter emits the pooler (`pooler: ok (act=gelu)`) and matches
the reference to 1e-4 (tanh) / 1e-5 (erf) — stale artifacts, not a converter
bug. Fix = G7c-style re-conversion + upload under new suffixed names + re-pin
(old files kept for released binaries' pins).

## 3. Rider CONFIRMED + FIXED: `crispembed_rerank_batch` aborted on quantized heads

`crispembed_rerank_batch` (the server's `POST /rerank`) duplicated the
classifier-cache population with raw `ggml_backend_tensor_get(...,
H*H*sizeof(float))` reads of the 2-D `classifier.dense` / `pooler.weight`
tensors — the known quantized-2-D raw-read overrun class the single-document
path (`crispembed_apply_classifier`) already guards with `core_cpu::to_f32`.
The quantizer DOES quantize both (`pooler.weight - f32, quantizing to
q8_0...`), and one `/rerank` request against a pooler-bearing q8_0 artifact
aborts the whole server process (`ggml-backend.cpp:349 "tensor read out of
bounds"`, backtrace in `rider_crash.txt`). Live today for jina-reranker-v2 and
every quantized 2-layer-head reranker served through `/rerank`; would have
become live for mxbai the moment finding 2's artifacts ship. Fixed by deleting
the duplicate block — `crispembed_apply_classifier()` populates the same cache
lazily and dequant-safely. After: `/rerank` returns HF-scale scores, server
stays up. The CLI `--rerank` path (`crispembed_rerank`) was never affected.

## Repro

```
MXGELU_WORK=<workdir> HF_HOME=<hf cache> python onnx_ref.py xsmall base
MXGELU_WORK=<workdir> CRISPEMBED_BIN=<build>/crispembed bash run_arms.sh
MXGELU_WORK=<workdir> python tabulate.py
```

`<workdir>` holds `mxbai-{xsmall,base}-f16-new.gguf` (fresh ollama-mode
conversions), `mxbai-{xsmall,base}-q8_0-new.gguf`, and the two downloaded
shipped `*-q8_0.gguf` files. Raw per-arm outputs: `arms_raw.txt`,
`scores.md`; ONNX reference: `onnx_ref.txt`.
