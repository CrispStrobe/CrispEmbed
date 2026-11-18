# Native spaCy: ggml engine and precision audit — 18 November 2026

Development evidence for the opt-in ggml document engine and quantized
derivatives of the admitted `de_core_news_sm` 3.8.0 F32 GGUF
(`fae2e6ab…`). Everything here is **default off**; the scalar F32 path, the
accepted W6/W7 package and its consumed panels are unchanged and were not
rerun. Host: macOS arm64 (Apple Silicon), development build of
`tools/spacy` (portable, `GGML_NATIVE=OFF`). Timings are single-host
diagnostics, not release numbers.

## Inputs

- Authored 60-text German development corpus
  [`tests/fixtures/spacy-ggml-ab/corpus-de.json`](../../../tests/fixtures/spacy-ggml-ab/corpus-de.json)
  (835 tokens). Not the frozen acceptance panel; no general quality claim.
- Original oracle [`oracle.json`](oracle.json): spaCy 3.8.18 / thinc 8.3.14,
  wheel SHA `fec69fec…` (pinned), written by `tools/spacy/ggml_ab_oracle.py`.
- The converted bundle is byte-identical to the pinned
  [bundle manifest](../native-spacy-20261117/bundle-manifest.json).

## Why the quantized models "all failed"

1. **The runtime refused them.** Before this change the loader admitted only
   F32 (`Only F32 spaCy tensors are admitted`) and exactly 13 metadata keys, so
   any generic quantizer output was rejected before inference; there was no
   quantized execution path to fail or pass.
2. **256-block K-quants cannot represent this model.** Every matrix row width
   is 96, 288, 576 or 64 — multiples of 32 but never of 256. Q*_K types fall
   back or skip; only 32-block types (Q8_0/Q5_0/Q4_0) and F16/BF16 apply.
3. **After admission, sub-F16 precision genuinely changes decisions.** This
   96-wide model makes greedy argmax decisions with small margins. Q8_0's
   normal ~0.5% relative weight error (no outliers: median block max/RMS
   ≈2.3, F16 2.1e-4) changes 7 field decisions in 5/60 sentences.

Point 3 is **not a runtime defect**. Attribution, per the crispembed-diff method:

| Check | Result |
| --- | --- |
| A. Port: native scalar F32 / ggml F32 vs original modules | all 22 referenced stages × 60 cases cos_min 1.000000; max abs ≤5.3e-5 / ≤3.8e-5 |
| B. Native scalar Q8_0 / BF16 vs **original spaCy running the same weights** | all stages cos_min 1.000000, max abs ≤5.3e-5 |
| Original spaCy with Q8_0 weights vs F32 oracle | exactly the same 7 flips as native (7 shared, 0 native-only, 0 replay-only) |
| Original spaCy with F16 / BF16 weights | 0 / 2 flips |
| B. ggml Q8_0 vs original-with-Q8_0 | cos_min 0.99991: extra activation rounding inside `ggml_mul_mat` (Q8_0 dot type) |
| C. ggml Q8_0 vs original F32 | cos_min ≥0.99985 at every stage — passes a 0.999 cosine gate yet changes 5/60 decoded sentences |

Stage tables: [`diff/`](diff/); replays: [`replay/`](replay/).

## Per-group Q8_0 ablation (scalar engine, rest F16)

Each group alone flips something, so no single "keep this in high precision"
choice restores parity below F16:

| Q8_0 group | Size | Exact cases | Field disagreements |
| --- | ---: | ---: | ---: |
| Hash embeddings | 2.72 MB | 58/60 | 3 |
| Embed mix | 3.57 MB | 57/60 | 4 |
| Encoder (4 layers) | 3.41 MB | 59/60 | 1 |
| Tagger/morph/lemma heads | 3.54 MB | 58/60 | 2 |
| Parser reduce/lower | 3.66 MB | 59/60 | 2 |

## Engine × precision (decoded agreement with the original oracle)

Long document = all 60 texts joined (835 tokens); tokens/s includes the
Python ctypes harness. `t` = `CRISPEMBED_SPACY_THREADS`; SGEMM = build option
`CRISPEMBED_SPACY_SGEMM=ON`.

| Configuration | GGUF | Exact | Disagreements | Long-doc tok/s | Max abs drift vs scalar F32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scalar F32 (default) | 7.38 MB | 60/60 | 0 | 1,375 | 0 (bit-identical to pre-change) |
| ggml F32, t1 | 7.38 MB | 60/60 | 0 | 3,968 | 1.9e-5 |
| ggml F32, t4 | 7.38 MB | 60/60 | 0 | 6,971 | — |
| **ggml F32 + SGEMM, t1** | 7.38 MB | **60/60** | **0** | **8,900** | 2.3e-5 |
| ggml F32 + SGEMM, t4 | 7.38 MB | 60/60 | 0 | 6,293 | 2.3e-5 |
| scalar F16 | 3.72 MB | 60/60 | 0 | 1,361 | 9.6e-3 |
| ggml F16, t1 / t4 | 3.72 MB | 60/60 | 0 | 4,116 / 4,650 | 9.6e-3 |
| scalar BF16 | 3.72 MB | 58/60 | 2 | 1,356 | 7.8e-2 |
| ggml BF16 | 3.72 MB | 60/60* | 0* | 1,169 | 1.0e-1 |
| scalar / ggml Q8_0 | 2.01 MB | 55 / 53 | 7 / 9 | — / 8,381 | 0.22 / 0.32 |
| scalar / ggml Q5_0 | 1.32 MB | 46 / 44 | 18 / 19 | — / 5,141 | 1.9 |
| scalar / ggml Q4_0 | 1.09 MB | 40 / 38 | 29 / 31 | — / 6,844 | 4.6 |

\* ggml BF16 is **not** qualified: original spaCy with the same BF16 weights
changes 2 decisions; the native result lands back on the oracle only because
of a second rounding step. BF16 also has no speed benefit here (no native BF16
kernels in the portable build) and its 8-bit mantissa is strictly worse than
F16 for these weights, whose range never needs BF16's exponent.

Quantized GGUF without `CRISPEMBED_SPACY_QUANT=1` is rejected:
[`ab/q8-gate-off.json`](ab/q8-gate-off.json).

## Conclusions

- Port correctness: both engines match original thinc modules stage by stage.
- Qualified on this development corpus: F32 (both engines) and F16 (both).
  Rejected: BF16, Q8_0, Q5_0, Q4_0, and every Q8_0 single-group recipe.
- Best measured configuration: ggml F32 with SGEMM, one thread —
  6.5× the scalar default with zero decoded differences.
- Regression controls: 4/4 ctest contracts and 21/21 real-model C-ABI
  controls (cancellation, trace abort, concurrency) pass with the engine off
  and on (`CRISPEMBED_SPACY_GGML=1 CRISPEMBED_SPACY_THREADS=2`).

Nothing here promotes a default, a package, or a release: the frozen W6/W7
evidence is bound to its own sources, and any default flip needs the
acceptance and packaging gates rerun at the new source.

## Reproduce

```sh
cmake -S tools/spacy -B build-spacy && cmake --build build-spacy -j
python3 tools/spacy/ggml_ab_oracle.py --wheel W --corpus C --output oracle.json
python3 models/quantize-spacy-gguf.py --input bundle/model.gguf --output q8_0.gguf --type q8_0
python3 tests/spacy_ggml_ab.py --library build-spacy/libcrispembed-spacy.dylib --bundle bundle \
    --oracle oracle.json --label ggml-q8_0 --report r.json --model q8_0.gguf \
    --env CRISPEMBED_SPACY_QUANT=1 --env CRISPEMBED_SPACY_GGML=1
python3 tools/spacy/dump_spacy_reference.py --wheel W --corpus C --output ref.gguf [--gguf q8_0.gguf]
CRISPEMBED_SPACY=1 build-spacy/crispembed-spacy-diff --ref ref.gguf --corpus C --bundle bundle [--model q8_0.gguf]
python3 tools/spacy/quant_oracle_replay.py --wheel W --gguf q8_0.gguf --oracle oracle.json --output replay.json
```
