# `CRISPEMBED_*_BENCH` value-parse audit — acceptance summary (2026-08-05)

The codebase-wide follow-up the DS_ audit (`91ebb55d`,
`tests/results/ds-gates/SUMMARY.md`) recorded as out of scope: every
`*_BENCH` gate resolved with `getenv(X) != nullptr` (or bare
`if (getenv(X))`), so `CRISPEMBED_FOO_BENCH=0` — the spelling an operator
reaches for to turn an instrument OFF — turned it ON.

**One shared helper.** `src/core/env_gate.h` / `core_env::on(name)`, hoisted
the way `core/imatrix_alias.h` and `core/no_repeat_ngram.h` were: set,
non-empty, and not exactly `"0"` => on. These are byte-for-byte the semantics
deepseek's `ds_env_on()` established; `ds_env_on` itself is untouched (it still
serves the 11 `DS_*`/`DS2_*` gates), only its BENCH gate now calls the shared
helper.

**68 presence-based sites converted** across 60 files, plus one already-correct
site each in `core/init_bench.h` (`CRISPEMBED_INIT_BENCH`) and
`fireredpunc.cpp` (`FIREREDPUNC_BENCH`) folded into the same helper so there is
exactly one implementation. `grep -rn getenv src/ | grep BENCH` now returns
nothing. Full table below.

**All 69 are diagnostic-only.** Every site was read: each one either sets a
`ctx->bench` flag whose only consumers are `if (bench) fprintf(stderr, ...)`
blocks, or guards an `fprintf` directly. Nothing behind a BENCH gate selects a
backend, a graph, a decode path, a shape, or a tolerance. The five sites that
look load-bearing at a glance are not:

| site | why it is still print-only |
|------|----------------------------|
| `ocr_orchestrator.cpp:667` | `if (boxes > max_graph_regions && BENCH)` — the graph budget is applied by the `ppocrv6_ocr_set_graph_accept` call on the line ABOVE, unconditionally; BENCH only adds the `[ppocrv6-graph-budget]` line |
| `ocr_orchestrator.cpp:1019` | `if (BENCH \|\| ctx->bench)` — prints `[tesseract-load-bench]` |
| `ocr_orchestrator.cpp:1088` | `ctx->bench` is one of three OR'd reasons to print the seg-router coverage line |
| `ppocrv6_ocr.cpp:1844` | the scalar fallback is decided by `batch_backend_ok`; BENCH only explains it on stderr |
| `mixtex_ocr.cpp:759` | a `static bool printed` print-once latch inside the bench block |

A handful of bench blocks also accumulate their own counters
(`decode_total_ms`, `decode_steps` in the VLM decode loops) — locals that feed
only the summary line.

## Evidence (`run_gates.sh` + `compare.sh` in this directory; 24/24 checks pass, `compare.log`)

Serialized, one model process at a time, `--gpu-backend metal` explicit on
every run, worktree binary by absolute path, `-DGGML_METAL=ON` verified in
`build/CMakeCache.txt` (`GGML_METAL:BOOL=ON`).

### Three-spelling proof — `CRISPEMBED_CRISPEMBED_BENCH`, multilingual-e5-small-q8_0, Metal (MTL0 in all three arms)

| arm | `[*-bench]` lines on stderr | stdout |
|-----|-----------------------------|--------|
| absent (`post-absent`) | 0 | 4089 B |
| `=0` (`post-zero`) | **0** — the fix | byte-identical |
| `=1` (`post-one`) | 4 | byte-identical |

```
[crispembed-bench] encode_tokens graph compute (T=10): 19.3 ms
[crispembed-bench] encode_tokens pool+normalize: 0.00 ms
[crispembed-bench] encode_tokens total: 19.9 ms
[crispembed-bench] crispembed_encode total: 20.0 ms
```

The stdout read (not just diffed) is the embedding JSON:
`{"text": "the quick brown fox", "embedding": [0.044926, -0.004542, ...]}` —
the same 384-d vector in all three arms.

**Pre-fix control, same build tree at the parent commit** (`pre-*.txt/.err`):
`pre-zero.err` carries the same four `[crispembed-bench]` lines that
`pre-one.err` does — `=0` enabled the instrument. `pre-absent.err` has none.
And `pre-{absent,zero,one}.txt == post-{absent,zero,one}.txt`, so the commit is
a no-op on output.

### `=1`-engages spot-checks (each also run with all three spellings)

| gate | invocation | absent | `=0` | `=1` | stdout |
|------|-----------|--------|------|------|--------|
| `CRISPEMBED_CC_DETECT_BENCH` | `--cc-detect scan_strip.png` (model-free) | 0 lines | **0 lines** | 5 lines | identical, 296 B (`detected 12 text regions` + boxes) |
| `CRISPEMBED_PARSEQ_BENCH` | `-m parseq-q8_0 --ocr arabic_printed_line.png` | 0 | **0** | 5 | identical, `WeijeRnal` |
| `CRISPEMBED_SCAN_CLEANUP_BENCH` | `--cleanup-only scan_strip.png` (model-free) | 0 | **0** | 6 | identical, sha256 `a34e8ff6…52b25a` over the 203485 B cleaned PNG |

```
[cc-detect-bench] binarize: 0.232 ms / close: 0.289 / CC label: 0.335 / filter: 0.001 / total: 0.923 ms
[parseq-bench] preprocess: 0.0 ms / encoder graph: 18.6 / decoder CA K/V: 24.5 / decoder total: 50.1 / total: 68.8 ms
[scan_cleanup-bench] despeckle: 8.2 ms / blackfilter: 0.6 / deskew: 6.0 / crop: 0.2 / whiten: 4.5 / total: 19.7 ms
```

Caveat, recorded rather than hidden: parseq is a Latin scene-text recognizer
and the only cached line fixture is Arabic, so `WeijeRnal` is not a correct
transcription. It is stable and byte-identical across all three arms, which is
the property under test; parseq's accuracy on that fixture is out of scope
here. `cc-detect` and `scan_cleanup` are CPU-only engines, so those arms show
no MTL0 line — expected, not a backend regression.

## Hermetic guard

`tests/test_env_gate.cpp` (target `test-env-gate`, added to the CI model-free
tier in `.github/workflows/build.yml` — which also picked up the missing
`./build/test-no-repeat-ngram` invocation): 10 checks over absent / `"0"` /
`"1"` / `""` / `"2"` / `"yes"` / `"00"` / `"0 "` / null name / never-set name.

**Red-gate proof.** With `core_env::on` temporarily reverted to presence
semantics (`return e != nullptr;`) the test goes red on exactly the two cases
the sweep exists to fix, then green again after restoring:

```
FAIL "0" => off (THE FIX: presence semantics said on): got on want off
FAIL "" => off: got on want off
env-gate: 10 checks, 2 failure(s)
```

## Out of scope, recorded (NOT touched)

The same line-pattern sweep finds **267 presence-based sites over 156 distinct
non-BENCH variables**. Unlike the BENCH class these are not all diagnostic —
several select a backend or a compute path, i.e. `=0` changes OUTPUT, not just
stderr. The largest and most defect-shaped cluster is `src/unlimited_ocr.cpp`'s
`UOCR_*` set (`UOCR_MMAP`, `UOCR_MOE_CPU`, `UOCR_DBG`, `UOCR_SAM_CONV_CPU`,
`UOCR_LMHEAD_CPU`, `UOCR_NO_KV`, `UOCR_FA_F32`, ~40 sites) — the exact mirror
of the `DS_*` set already fixed in `deepseek_ocr2.cpp`, in the engine that
shares deepseek's lineage. Other output-affecting examples:
`CRISPEMBED_PPOCRV6_FORCE_CPU`, `EASYOCR_FORCE_CPU`, `NAFNET_CPU`,
`CRISPEMBED_TESSERACT_FORCE_CPU`, `SAFMN_SR_METAL`, `ESRGAN_SCALAR`,
`CRISPEMBED_PPOCRV6_NO_GRAPH`, `GLM_OCR_DECODE_CACHE`,
`CRISPEMBED_NO_KV_CACHE`, `LAYOUT_DETECT_FLASH`. Converting them needs the
per-gate A/B this diagnostic-only sweep did not do: each must be shown not to
change decoded output when its default resolution flips from "enabled" to
"disabled".

## Provenance

- `run_gates.sh` — the 12-run matrix (4 gates x 3 spellings), bash 3.2 safe.
- `compare.sh` / `compare.log` — the 24 acceptance checks.
- `pre-*.txt` / `pre-*.err` — pre-fix control arms (parent-commit binary).
- `post-*`, `cc-*`, `parseq-*`, `cleanup-*` — post-fix arms, stdout / stderr.
  (`cleanup-*.txt` holds the sha256 of the emitted PNG, not the 203 kB payload.)

Ambient note: the `pre-absent` arm paid a 10.3 s cold Metal library load that
the later arms did not; no timing claim is made from these runs beyond
"the bench lines are present / absent".

## Full inventory

Every `*_BENCH` gate in `src/`, after the sweep. "was presence-based" = the
`=0`-means-on defect; the two `no` rows were already value-parsed and were
folded into the shared helper for single-implementation hygiene.

| # | file:line | variable | was presence-based | converted | output-affecting |
|---|-----------|----------|--------------------|-----------|------------------|
| 1 | `src/adair.cpp:955` | `CRISPEMBED_ADAIR_BENCH` | YES | yes | no (stderr only) |
| 2 | `src/bert_ner.cpp:42` | `CRISPEMBED_BERT_NER_BENCH` | YES | yes | no (stderr only) |
| 3 | `src/bidirlm_audio.cpp:58` | `CRISPEMBED_BIDIRLM_AUDIO_BENCH` | YES | yes | no (stderr only) |
| 4 | `src/bidirlm_vision.cpp:503` | `CRISPEMBED_BIDIRLM_VISION_BENCH` | YES | yes | no (stderr only) |
| 5 | `src/bttr_ocr.cpp:345` | `CRISPEMBED_BTTR_BENCH` | YES | yes | no (stderr only) |
| 6 | `src/cc_detect.cpp:184` | `CRISPEMBED_CC_DETECT_BENCH` | YES | yes | no (stderr only) |
| 7 | `src/clip_text_embed.cpp:92` | `CRISPEMBED_CLIP_TEXT_BENCH` | YES | yes | no (stderr only) |
| 8 | `src/cnn_embed.cpp:83` | `CRISPEMBED_CNN_EMBED_BENCH` | YES | yes | no (stderr only) |
| 9 | `src/core/init_bench.h:35` | `CRISPEMBED_INIT_BENCH` | no | yes | no (stderr only) |
| 10 | `src/crispembed.cpp:2346` | `CRISPEMBED_CRISPEMBED_BENCH` | YES | yes | no (stderr only) |
| 11 | `src/dat_sr.cpp:1398` | `CRISPEMBED_DAT_SR_BENCH` | YES | yes | no (stderr only) |
| 12 | `src/decoder_embed.cpp:525` | `CRISPEMBED_DECODER_EMBED_BENCH` | YES | yes | no (stderr only) |
| 13 | `src/deepseek_ocr2.cpp:2910` | `CRISPEMBED_DEEPSEEK_OCR2_BENCH` | YES | yes | no (stderr only) |
| 14 | `src/dewarp.cpp:233` | `CRISPEMBED_DEWARP_BENCH` | YES | yes | no (stderr only) |
| 15 | `src/easyocr_ocr.cpp:423` | `EASYOCR_BENCH` | YES | yes | no (stderr only) |
| 16 | `src/easyocr_pipeline.cpp:123` | `CRISPEMBED_EASYOCR_STAGE_BENCH` | YES | yes | no (stderr only) |
| 17 | `src/easyocr_pipeline.cpp:221` | `CRISPEMBED_EASYOCR_STAGE_BENCH` | YES | yes | no (stderr only) |
| 18 | `src/esrgan_sr.cpp:156` | `CRISPEMBED_ESRGAN_BENCH` | YES | yes | no (stderr only) |
| 19 | `src/face_align.cpp:102` | `CRISPEMBED_FACE_ALIGN_BENCH` | YES | yes | no (stderr only) |
| 20 | `src/fireredpunc.cpp:38` | `FIREREDPUNC_BENCH` | no | yes | no (stderr only) |
| 21 | `src/gliner_ner.cpp:1048` | `CRISPEMBED_GLINER_BENCH` | YES | yes | no (stderr only) |
| 22 | `src/glm_ocr.cpp:471` | `CRISPEMBED_GLM_OCR_BENCH` | YES | yes | no (stderr only) |
| 23 | `src/got_ocr.cpp:401` | `CRISPEMBED_GOT_OCR_BENCH` | YES | yes | no (stderr only) |
| 24 | `src/granite_vision_ocr.cpp:366` | `CRISPEMBED_GRANITE_OCR_BENCH` | YES | yes | no (stderr only) |
| 25 | `src/hat_sr.cpp:543` | `CRISPEMBED_HAT_SR_BENCH` | YES | yes | no (stderr only) |
| 26 | `src/hmer_ocr.cpp:314` | `CRISPEMBED_HMER_BENCH` | YES | yes | no (stderr only) |
| 27 | `src/instructir.cpp:568` | `CRISPEMBED_INSTRUCTIR_BENCH` | YES | yes | no (stderr only) |
| 28 | `src/internvl2_ocr.cpp:1065` | `CRISPEMBED_INTERNVL2_BENCH` | YES | yes | no (stderr only) |
| 29 | `src/kie_pipeline.cpp:32` | `CRISPEMBED_KIE_PIPELINE_BENCH` | YES | yes | no (stderr only) |
| 30 | `src/layout_detect.cpp:195` | `CRISPEMBED_LAYOUT_DETECT_BENCH` | YES | yes | no (stderr only) |
| 31 | `src/lfm2_embed.cpp:166` | `CRISPEMBED_LFM2_EMBED_BENCH` | YES | yes | no (stderr only) |
| 32 | `src/lightonocr.cpp:305` | `CRISPEMBED_LIGHTONOCR_BENCH` | YES | yes | no (stderr only) |
| 33 | `src/lilt_kie.cpp:105` | `CRISPEMBED_LILT_KIE_BENCH` | YES | yes | no (stderr only) |
| 34 | `src/math_ocr.cpp:1500` | `CRISPEMBED_MATH_OCR_BENCH` | YES | yes | no (stderr only) |
| 35 | `src/mixtex_ocr.cpp:444` | `CRISPEMBED_MIXTEX_BENCH` | YES | yes | no (stderr only) |
| 36 | `src/nafnet_denoise.cpp:388` | `CRISPEMBED_NAFNET_BENCH` | YES | yes | no (stderr only) |
| 37 | `src/ocr_detect.cpp:245` | `CRISPEMBED_OCR_DETECT_BENCH` | YES | yes | no (stderr only) |
| 38 | `src/ocr_orchestrator.cpp:571` | `CRISPEMBED_EASYOCR_BENCH` | YES | yes | no (stderr only) |
| 39 | `src/ocr_orchestrator.cpp:613` | `CRISPEMBED_PPOCRV6_BENCH` | YES | yes | no (stderr only) |
| 40 | `src/ocr_orchestrator.cpp:667` | `CRISPEMBED_PPOCRV6_BENCH` | YES | yes | no (stderr only) |
| 41 | `src/ocr_orchestrator.cpp:1019` | `CRISPEMBED_TESSERACT_BENCH` | YES | yes | no (stderr only) |
| 42 | `src/ocr_orchestrator.cpp:1802` | `CRISPEMBED_OCR_ORCH_BENCH` | YES | yes | no (stderr only) |
| 43 | `src/ocr_pipeline.cpp:68` | `CRISPEMBED_OCR_PIPELINE_BENCH` | YES | yes | no (stderr only) |
| 44 | `src/pan_sr.cpp:167` | `CRISPEMBED_PAN_SR_BENCH` | YES | yes | no (stderr only) |
| 45 | `src/parseq_ocr.cpp:380` | `CRISPEMBED_PARSEQ_BENCH` | YES | yes | no (stderr only) |
| 46 | `src/pix2struct.cpp:254` | `CRISPEMBED_PIX2STRUCT_BENCH` | YES | yes | no (stderr only) |
| 47 | `src/posformer_ocr.cpp:346` | `CRISPEMBED_POSFORMER_BENCH` | YES | yes | no (stderr only) |
| 48 | `src/ppformulanet_l_ocr.cpp:1076` | `CRISPEMBED_PPFN_L_BENCH` | YES | yes | no (stderr only) |
| 49 | `src/ppformulanet_ocr.cpp:1111` | `CRISPEMBED_PPFN_BENCH` | YES | yes | no (stderr only) |
| 50 | `src/ppocrv6_det.cpp:1094` | `CRISPEMBED_PPOCRV6_DET_BENCH` | YES | yes | no (stderr only) |
| 51 | `src/ppocrv6_ocr.cpp:1101` | `CRISPEMBED_PPOCRV6_GRAPH_BENCH` | YES | yes | no (stderr only) |
| 52 | `src/ppocrv6_ocr.cpp:1218` | `CRISPEMBED_PPOCRV6_GRAPH_BENCH` | YES | yes | no (stderr only) |
| 53 | `src/ppocrv6_ocr.cpp:1844` | `CRISPEMBED_PPOCRV6_BENCH` | YES | yes | no (stderr only) |
| 54 | `src/ppocrv6_ocr.cpp:1884` | `CRISPEMBED_PPOCRV6_BENCH` | YES | yes | no (stderr only) |
| 55 | `src/qwen2vl_ocr.cpp:1066` | `CRISPEMBED_QWEN2VL_BENCH` | YES | yes | no (stderr only) |
| 56 | `src/restormer.cpp:472` | `CRISPEMBED_RESTORMER_BENCH` | YES | yes | no (stderr only) |
| 57 | `src/safmn_sr.cpp:272` | `CRISPEMBED_SAFMN_SR_BENCH` | YES | yes | no (stderr only) |
| 58 | `src/scan_cleanup.cpp:57` | `CRISPEMBED_SCAN_CLEANUP_BENCH` | YES | yes | no (stderr only) |
| 59 | `src/scunet_denoise.cpp:608` | `CRISPEMBED_SCUNET_BENCH` | YES | yes | no (stderr only) |
| 60 | `src/smoldocling_ocr.cpp:352` | `CRISPEMBED_SMOLDOCLING_BENCH` | YES | yes | no (stderr only) |
| 61 | `src/surya_det.cpp:194` | `CRISPEMBED_SURYA_DET_BENCH` | YES | yes | no (stderr only) |
| 62 | `src/swinir_sr.cpp:392` | `CRISPEMBED_SWINIR_SR_BENCH` | YES | yes | no (stderr only) |
| 63 | `src/table_parse.cpp:213` | `CRISPEMBED_TABLE_PARSE_BENCH` | YES | yes | no (stderr only) |
| 64 | `src/tbsrn_sr.cpp:432` | `CRISPEMBED_TBSRN_SR_BENCH` | YES | yes | no (stderr only) |
| 65 | `src/tesseract_lstm.cpp:1122` | `CRISPEMBED_TESSERACT_BENCH` | YES | yes | no (stderr only) |
| 66 | `src/text_sr.cpp:298` | `CRISPEMBED_TEXT_SR_BENCH` | YES | yes | no (stderr only) |
| 67 | `src/tps_locnet.cpp:207` | `CRISPEMBED_TPS_LOCNET_BENCH` | YES | yes | no (stderr only) |
| 68 | `src/unlimited_ocr.cpp:2830` | `CRISPEMBED_UNLIMITED_OCR_BENCH` | YES | yes | no (stderr only) |
| 69 | `src/vit_embed.cpp:96` | `CRISPEMBED_VIT_EMBED_BENCH` | YES | yes | no (stderr only) |
