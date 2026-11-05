# Environment variables

Single reference for every runtime environment variable CrispEmbed reads. The
curated sections below explain the variables that matter for normal use and for
the recently added engines; the **full inventory** at the bottom is generated
from the source by [`tools/list_env_vars.py`](../tools/list_env_vars.py) and is
checked in CI by [`tests/test_env_vars_documented.py`](../tests/test_env_vars_documented.py),
which fails when a `CRISPEMBED_*` variable read by code is missing from this file
or when the generated block is stale.

Adding a variable: read it through `core_env::on()` (default-off gate),
`core_env::explicitly_off()` (default-on gate with `=0` opt-out) or
`core_env::value()` (path/int/string; empty counts as unset) from
`src/core/env_gate.h`, add a sentence to the matching curated section if users
need it, then run `python tools/list_env_vars.py --write`. See the
[contribution checklist](contributing.md).

## Semantics

| Kind | Reader | Unset | `=` (empty) | `=0` | Any other non-empty value |
|---|---|---|---|---|---|
| `on` | `core_env::on` | off | off | off | on |
| `off` | `core_env::explicitly_off` | **on** (default) | on (default) | off | on |
| `value` | `core_env::value` / getenv + parse | built-in default | built-in default (`core_env::value`) | parsed | parsed |

Only an exact `0` is off: `false`/`off`/`no` currently count as ON. Treating them as
off needs the same change in CrispASR's code-identical `src/core/env_gate.h`; this is
tracked in PLAN. `core_env::value` lives in `src/core/env_value.h`. Raw presence truth tests (`getenv(X) != nullptr`,
`if (getenv(X))`, `!getenv(X)`), for which `X=0` meant *on*, were all converted
on 2 November 2026 (131 variables; defaults unchanged, only explicit
`=0`/empty changed meaning). `tests/test_env_vars_documented.py`
fails on any new one (`python tools/list_env_vars.py --presence` lists them);
deliberate exceptions go in its `PRESENCE_ALLOWLIST` with a reason.
Status values in the
inventory: *production default (opt-out)*, *default-off experimental / opt-in*,
*debug* (dumps, timing, bench, trace output), *tuning / config* (value knobs) and
*internal* (written by the CLI/orchestrator, not a user knob).

## Core, threads, backends, model registry

| Variable | Default | Meaning |
|---|---|---|
| `CRISPEMBED_FORCE_CPU` | unset | Force the CPU backend for the main embedding context (also honoured by GLiNER). |
| `CRISPEMBED_SHARED_GPU_BACKEND` | off | Share one GPU backend between contexts. |
| `CRISPEMBED_GPU_PREF_CPU_LEGACY` | unset | Restore the legacy CPU-preference rule for GPU backend selection. |
| `CRISPEMBED_METAL_RESIDENCY`, `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB` | unset | Metal residency-set and pipeline-cache tuning. |
| `CRISPEMBED_CONV2D_GEMM`, `CRISPEMBED_CONV2D_MK`, `CRISPEMBED_CONV2D_THREADS`, `CRISPEMBED_CONV1X1_FAST`, `CRISPEMBED_CONVDW_FAST` | unset | Shared CPU convolution kernels (`src/core/cpu_ops.h`): GEMM path, micro-kernel, thread count. |
| `CRISPEMBED_RAM_GUARD`, `CRISPEMBED_RAM_GUARD_AVAILABLE_MB` | unset | Pre-load memory guard mode and the available-memory override. |
| `CRISPEMBED_OCR_POOL_SIZE` | unset | Number of pooled OCR contexts. |
| `CRISPEMBED_CACHE_DIR` | platform cache | Model cache directory (also set by CLI `--cache-dir`). |
| `CRISPEMBED_OFFLINE` | off | Never download; resolve only cached models (also honours `HF_HUB_OFFLINE`). |
| `CRISPEMBED_ALLOW_UNPINNED_MODEL` | unset | One-off override that lets a registry URL without a SHA-256 pin download. |
| `CRISPEMBED_ACCEPT_LICENSE`, `CRISPEMBED_ACCEPT_*_LICENSE`, `CRISPEMBED_ACCEPT_BIOMETRIC` | unset | Explicit licence / biometric acknowledgements required by specific models. |
| `CRISPEMBED_STRICT_HPARAMS`, `CRISPEMBED_ARCH_HPARAMS`, `CRISPEMBED_GGUF_REPARSE` | unset | GGUF metadata strictness and re-parse controls. |
| `CRISPEMBED_SERVER_RERANK_RAW_SCORES` | off | `/v1/rerank` emits raw logits instead of `sigmoid(logit)`. |
| `CRISPEMBED_SERVER_USAGE_TOKENS` | on | `=0` restores the historical `0` / text-count token usage on `/v1/embeddings` and `/api/embed`; the default reports the observed tokenizer ids. |
| `CRISPEMBED_SERVER_METRICS` | off | `GET /metrics` (Prometheus): running/waiting/in-flight requests, admitted/rejected/completed, request seconds, reported tokens. |
| `CRISPEMBED_SERVER_MAX_INFLIGHT` | unset (uncapped) | POST requests beyond N in flight get `429` before touching a model; `0`, empty or junk = uncapped. |
| `CRISPEMBED_SERVER_RETRY_AFTER` | unset | Seconds to send as `Retry-After` on those 429s; unset = no header. |
| `CRISPEMBED_SERVER_OPENAI_CHAT` | off | `POST /v1/chat/completions` + `GET /v1/models` for the `--ocr` model (one base64 data-URL image per request). |
| `CRISPEMBED_SERVER_MAX_IMAGE_BYTES` | 64 MiB | Decoded image size limit for `/v1/chat/completions`. |
| `CRISPEMBED_SERVER_LEGACY_JSON` | unset | Legacy server JSON shape. |
| `CRISPEMBED_LIB` (Python/examples), `CRISPEMBED_SYS_LIB_DIR` (Rust build) | unset | Location of the native library. |

## Embeddings and tokenizers

| Variable | Default | Meaning |
|---|---|---|
| `CRISPEMBED_ENCODER_PACKED`, `CRISPEMBED_ENCODER_PACK_MAXTOK`, `CRISPEMBED_ENCODER_4D`, `CRISPEMBED_ENCODER_4D_GROUP`, `CRISPEMBED_ENCODER_NO_SWA` | unset | Encoder batching layout (packed sequences, 4-D attention) and sliding-window override. |
| `CRISPEMBED_DECODER_PREFIX_CACHE` | unset | Decoder-embedding prompt-prefix KV cache. |
| `CRISPEMBED_EMBED_RAW_NORM`, `CRISPEMBED_DECODER_EMBED_RAW_NORM` | unset | Print the pre-normalisation norm (debug). |
| `CRISPEMBED_SPARSE_KEEP_SPECIAL` | off | Keep special tokens in SPLADE/BGE-M3 sparse output. |
| `CRISPEMBED_RERANK_POOLER_GELU_ERF` | unset | Exact-erf GELU in the reranker pooler. |
| `CRISPEMBED_PAIR_ROBERTA_SEP` | on | Cross-encoder pairs for `<s>`/`</s>` (XLM-R/RoBERTa) vocabularies use HF's `<s> a </s></s> b </s>`; `=0` restores the single separator. |
| `CRISPEMBED_POS_EMBD_TRUNCATE` | on | Truncate encoder input to the learned position table (`rows - position_offset`, 512 for XLM-R) instead of aborting; `=0` restores the old limit. |
| `CRISPEMBED_WORDPIECE_HF_*`, `CRISPEMBED_SPM_*`, `CRISPEMBED_BPE_LEGACY_WHITESPACE`, `CRISPEMBED_WARN_UNK` (default on) | — | Tokenizer normalisation compatibility switches. |
| `CRISPEMBED_ULTRABERT`, `_QUANT`, `_TRUNCATE`, `_UNICODE_PRETOKEN` | off | Default-off UltraBERT CPU/F32 text API and its options. |
| `CRISPEMBED_LFM2_ENCODER`, `CRISPEMBED_LFM2_F32_DOT`, `CRISPEMBED_LFM2_F32_MATMUL` | off | LFM2 encoder gate and precision options. |
| `CRISPEMBED_GEMMA_DECISION` (+ `_DUMP`, `_DUMP_PATH`, `_F32_DOT`, `_LEGACY_GELU`) | off | Gemma decision head gate, dumps and precision options. |

### WASM rounding gates

`CRISPEMBED_WASM_Q8_0_ROUND` (default off, `core_env::on`) enables the
per-backend Q8_0 activation-rounding candidate in `src/crispembed.cpp`. The
model A/B rejected it (8/10 finals); keep it off.

### MoEdern (moedern-1.5m / 27m / 115m)

All default off (`core_env::on`).

| Variable | Meaning |
|---|---|
| `CRISPEMBED_MOEDERN_COLBERT` | Required gate for the MoEdernColBERT engine. |
| `CRISPEMBED_MOEDERN_TEXT_FRONTEND`, `CRISPEMBED_MOEDERN_PUBLIC_TEXT`, `CRISPEMBED_MOEDERN_TEXT_COMPOSED`, `CRISPEMBED_MOEDERN_PACKED` | Raw-text frontend, public text API (`crispembed_moedern_*`), composed text path, packed GGUF loader. |
| `CRISPEMBED_MOEDERN_FAMILY` | Allows the 27M and 115M family members (1.5M works without it). |
| `CRISPEMBED_MOEDERN_FAST` | Fast ggml graph path; lifts the one/two-thread bound (Python allows up to 64 threads only with this set). |
| `CRISPEMBED_MOEDERN_QUANT` with `CRISPEMBED_MOEDERN_Q8_0` or `CRISPEMBED_MOEDERN_Q4_K_PADDED` | Quantized weights; fast path only. |

### Josev SystemOne (josev2b-systemone)

`CRISPEMBED_JOSEV_BACKBONE`, `_HEAD`, `_COMPOSED`, `_PACKED`, `_TEXT_FRONTEND`,
`_ANSWER_SERIALIZER` gate the stages of the packed CPU/F32 model, and
`CRISPEMBED_JOSEV_QUANT` (with `CRISPEMBED_QWEN35_QUANT`) the quantized backbone.
The backbone also requires `CRISPEMBED_QWEN35_OCR=1` because it reuses the
qwen35 decoder kernels. All default off.

## OCR engines

### Unlimited-OCR-2 (`unlimited-ocr2`, `unlimited-ocr2-q8`; orchestrator engine 13)

| Variable | Default | Meaning |
|---|---|---|
| `CRISPEMBED_UNLIMITED_OCR2` | off | **Required.** A variant-2 GGUF refuses to decode without it (never falls back to v1 attention). |
| `CRISPEMBED_UNLIMITED_OCR2_FAST` | off | Opt-in ggml decoder; the exact CPU/F32 reference decoder stays the default. |
| `CRISPEMBED_UNLIMITED_OCR2_FORCE_IDS` | unset | Diagnostic teacher forcing for the fast decoder: a file of whitespace-separated token IDs fed instead of the argmax (docs/unlimited-ocr2-q4k-recipes.md). |
| `CRISPEMBED_UNLIMITED_OCR2_LOGITS_OUT` | unset | Diagnostic: write each fast-decoder step's raw lm_head logits (F32) to this file. |
| `CRISPEMBED_UNLIMITED_OCR2_TILES` | on | Follows the original >640 px tiling (images wider or taller than 640 px are tiled). `=0` forces the untiled path. |
| `CRISPEMBED_UNLIMITED_OCR2_TILES_LEGACY_PREP`, `..._TILES_LEGACY_UPSAMPLE` | off | Restore the earlier tile preprocessing / SAM tile-grid upsampling. |
| `CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE` | on | Speed-up: vision cache and row-parallel projection for variant 2. `=0` disables. |
| `CRISPEMBED_UNLIMITED_OCR2_VISION_F32` | on | F32 copies of the SAM/CLIP block matrices made once at init (variant 2 only); F32 x F32 vision products. Default since 2026-11-03: 4/4 trial pages exact, real scans 1.43x, 17/19 ids-identical to the F16 path, original-exact 12 -> 13/20. `=0` restores the F16 path. |
| `CRISPEMBED_UNLIMITED_OCR2_LOW_MEMORY` | auto | Skip those F32 copies (F16 vision path, the pre-2026-11-03 default) when available RAM < copies + 1 GiB, with one stderr notice. `=1` always, `=0` never checks. |
| `CRISPEMBED_UNLIMITED_OCR2_TILE_BATCH` | off | Gundam tiles run through SAM's layers and CLIP as one graph per layer (vision bytes identical to unbatched). About 2% faster only. |
| `CRISPEMBED_UNLIMITED_OCR2_TILE_BATCH_MAX` | 16 | Tiles per batched graph (1-64) when `..._TILE_BATCH=1`. |
| `CRISPEMBED_UNLIMITED_OCR2_OUTPUT` | raw | `clean` (alias `markdown`) strips layout markup; `legacy` reproduces the earlier post-processing; anything else is raw model text. |
| `CRISPEMBED_UNLIMITED_OCR_BENCH` | off | Stage timings on stderr. |
| `UOCR_MAX_NEW`, `UOCR2_DUMP_VISION`, `UOCR_DUMP_PIXELS` | unset | Token cap override and debug dumps. |

See [the inference spec](unlimited-ocr2-inference-spec.md).

### MiniCPM-OCR (`minicpm-ocr`, `minicpm-ocr-q8`; orchestrator engine 23)

`--ocr-engine minicpm-ocr` without a model resolves the published `minicpm-ocr`
alias in both the CLI and the HTTP server; both gates below are still required.

| Variable | Default | Meaning |
|---|---|---|
| `CRISPEMBED_MINICPM_OCR` | off | **Required** engine gate. |
| `CRISPEMBED_QWEN35_OCR` | off | **Required**: the decoder is the shared qwen35 implementation. |
| `CRISPEMBED_MINICPM_OCR_DUMP` | unset | Directory for intermediate dumps. |
| `CRISPEMBED_MINICPM_OCR_FORCE_IDS` | unset | Test-only teacher forcing: file of token IDs that replace the greedy choice step by step (quantization A/B, docs/minicpm-ocr-q4k.md). |
| `CRISPEMBED_MINICPM_OCR_LOGITS_OUT` | unset | Test-only: append each step's raw F32 vocabulary scores to this file. |
| `CRISPEMBED_MINICPM_OCR_TIMING` | off | Per-stage wall times on stderr. |
| `CRISPEMBED_MINICPM_OCR_VISION_FLASH` | on | Fused flash attention in the vision tower (performance only). `=0` opts out. |
| `CRISPEMBED_MINICPM_OCR_GRAPH` | off | Older stateless graph-decoder candidate (`src/qwen35_graph_decoder.cpp`); disabled under `CRISPEMBED_QWEN35_IMATRIX`; exclusive with `CRISPEMBED_QWEN35_GRAPH_DECODER` (setting it to 1 turns the default graph decoder off unless that is forced with `=1`). |
| `CRISPEMBED_QWEN35_GRAPH_DECODER` | on | qwen35 text decoder as one ggml graph with persistent KV and delta-net state (`src/qwen35_graph_kv_decoder.cpp`; MiniCPM is the only caller). `=0` restores the CPU reference decoder. |
| `CRISPEMBED_QWEN35_GRAPH_MATMUL` | `f32` | Matmul precision of that graph decoder: `f32`, `exact` or `native` (`native` saves the F32 copies' RAM; also 5/5 exact). |
| `CRISPEMBED_MINICPM_OCR_VISION_F32` | on | Copy vision and projector matrices to F32 once at init (part of the ~3.3 GB higher peak RSS). `=0` opts out. |
| `CRISPEMBED_MINICPM_OCR_LOW_MEMORY` | auto | Skip the default F32 copies (vision matrices and the `f32` graph-decoder matmul; F16 vision + `exact` matmul instead, token-exact on the measured panels) when available RAM < copies + 1 GiB, with one stderr notice. `=1` always, `=0` never checks. |
| `CRISPEMBED_MINICPM_OCR_VISION_BATCH` | on | Run equally sized slices in one graph as independent sequences. `=0` opts out. |

The speed set (graph decoder `f32` + vision F32 + batch + flash) is the default since 2026-11-02:
token-ID exact on all five Kaggle cases and IDs identical to the old default on all 20 real-scan pages,
6.2x faster, peak RSS 8.4 GB (was 5.1 GB). Setting all four gates to `0` restores the old path. See
[the inference spec](minicpm-ocr-inference-spec.md) and PERFORMANCE.md.

### qwen35 decoder (shared by LightOnOCR-3 `qwen35_ocr`, OvisOCR2/3, MiniCPM-OCR, Josev)

| Variable | Default | Meaning |
|---|---|---|
| `CRISPEMBED_QWEN35_OCR` | off | Master gate for every qwen35 kernel. |
| `CRISPEMBED_QWEN35_GDN_GROUPED_HEADS` | off | Separate experimental linear-attention APIs for compact unequal key/value head counts; also requires `CRISPEMBED_QWEN35_OCR=1`. Unset, empty or `0` disables them. Existing equal-head callers are unchanged; this component gate does not admit a 4B model. |
| `CRISPEMBED_QWEN35_QUANT` | off | Quantized linear weights. |
| `CRISPEMBED_QWEN35_PARALLEL_F16` | off | Thread-parallel F16 conversion of large matrices. |
| `CRISPEMBED_QWEN35_WEIGHT_CACHE` | off | Per-context converted-weight cache (measured 2.7% cold / 0.8% warm gain with more RSS; kept off). |
| `CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES` | 268435456 (256 MiB) | Cache budget in bytes; empty or invalid = 0. |
| `CRISPEMBED_QWEN35_IMATRIX` | off | Importance-matrix collection (observer); `CRISPEMBED_IMATRIX_OUT` names the output. |
| `CRISPEMBED_QWEN35_PRECISE_RMS`, `CRISPEMBED_QWEN35_VISION_PRECISE_ATTN`, `..._PRECISE_NORM`, `..._CASCADE_NORM`, `..._SOURCE_INTERPOLATION` | off | Precision experiments; kept default-off. |
| `CRISPEMBED_QWEN35_TOKENIZER_MARKS`, `CRISPEMBED_QWEN35_PROFILE` | off | Tokenizer mark handling; profiling output. |
| `CRISPEMBED_OVISOCR2_FRONTEND`, `_INFERENCE`, `_LARGE_IMAGES`, `CRISPEMBED_OVISOCR3_FRONTEND`, `_INFERENCE` | off | OvisOCR2/3 gates. |

### MinerU 2605 checkpoint

`CRISPEMBED_MINERU25_PRO_2605=1` is an additional **default-off** admission gate
for the exact 2605 checkpoint profile. `CRISPEMBED_MINERU25_PRO=1` is also required.
Unset, empty or `0` keeps the new profile disabled. The runtime also verifies the
exact checkpoint repository/revision metadata; neither gate selects weights or
qualifies the new document helpers. See [the 2605 protocol](mineru25-pro-2605.md).

### Benchmark candidate profiles

`CRISPEMBED_LIGHTONOCR3_KV_F32=1` opts the admitted LightOn3-1B CPU path
into F32 persistent key/value storage; unset/empty/0, legacy profiles and GPU
backends retain F16. This doubles the cache allocation for the same token
capacity, without changing GGUF weight bytes. The first-only retained-original
cache A/B holds the corrected Pixtral epsilon gate at1. It is an experiment,
not a new default or completed model/publication qualification.

`CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS=1` opts the admitted LightOn3-1B profile
into the pinned Pixtral vision RMSNorm epsilon (1e-5 at ln_pre and both block
norms). Unset, empty or `0` preserves historical 1e-6. The gate never changes
legacy LightOn profiles, the projector or text norms. This experiment remains
default-off until retained-reference A/B review; it does not qualify publication.

`CRISPEMBED_LIGHTONOCR3=1` admits the exact 1B LightOn3 profile; grounding also
requires `CRISPEMBED_LIGHTONOCR3_GROUNDING=1`. Xiaomi format 4 requires
`CRISPEMBED_QWEN35_OCR=1`, `CRISPEMBED_XIAOMI_OCR0_FRONTEND=1`, and
`CRISPEMBED_XIAOMI_OCR0_INFERENCE=1`. Each candidate gate is default-off.
`CRISPEMBED_NEMOTRON_PARSE_GEOMETRY=1` enables only the bounded processor
geometry/patch prototype; it is a default-off Boolean even though the inventory
scanner sees its constant declaration as a value. It does not select an OCR
engine. Unset, empty and `0` leave these candidate paths disabled.

### Other OCR engines

Per-engine gates follow the same pattern: `CRISPEMBED_<ENGINE>` enables an
experimental engine (`TELEOCR`, `MINERU25_PRO`, `OXR`, `MARKER`), `_BENCH` /
`_TIMING` / `_DEBUG` / `_DUMP*` are diagnostics, `_FORCE_CPU` / `_GPU` pick the
backend, and Tesseract-LSTM has its own recode/DAWG/page-segmentation gates
(see `CLAUDE.md`). Legacy non-prefixed `<ENGINE>_FORCE_CPU` and `*_DEBUG`
switches are listed at the end of the inventory.

## Inpainting

`CRISPEMBED_MIGAN`, `CRISPEMBED_MIGAN512`, `CRISPEMBED_MOEBIUS`,
`CRISPEMBED_PIXELHACKER` gate the experimental inpainting engines;
`CRISPEMBED_INPAINT_*` and `CRISPEMBED_MIGAN_*` select graph/GPU/quantized
kernels. See [inpainting-engines.md](inpainting-engines.md).

## Full inventory

Regenerate with `python tools/list_env_vars.py --write`.

<!-- BEGIN GENERATED: tools/list_env_vars.py -->

447 `CRISPEMBED_*` variables and 223 non-prefixed variables. Kind: `on` = `core_env::on` (default off), `off` = `core_env::explicitly_off` (default on, `=0` opts out), `presence` = raw `getenv(X)` truth test (forbidden: `tests/test_env_vars_documented.py` fails on one), `value` = value read via getenv/os.environ, `set` = written by the CLI/orchestrator. Hint = nearest source comment (may be empty).

### CLI / server / examples

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ACCEPT_BIOMETRIC` | value | unset | string | tuning / config | `examples/cli/model_mgr.cpp`, `examples/face_verify.py`, `src/crispembed.cpp` |  |
| `CRISPEMBED_ACCEPT_LICENSE` | value | unset | value | tuning / config | `examples/cli/model_mgr.cpp` |  |
| `CRISPEMBED_ALLOW_UNPINNED_MODEL` | value | unset | path | tuning / config | `examples/cli/model_mgr.cpp` |  |
| `CRISPEMBED_CACHE_DIR` | set/value | unset | path | tuning / config | `examples/cli/main.cpp`, `examples/cli/model_mgr.cpp` | else |
| `CRISPEMBED_INPAINT_STREAM` | on | off | bool | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp` |  |
| `CRISPEMBED_LAYOUT_REPEAT` | value | unset | int | tuning / config | `examples/cli/main.cpp` |  |
| `CRISPEMBED_LIB` | value | unset | path | tuning / config | `examples/demo/demo.py`, `examples/face_verify.py` |  |
| `CRISPEMBED_MODEL` | value | unset | path | tuning / config | `examples/demo/demo.py`, `examples/demo/demo.rs` |  |
| `CRISPEMBED_OFFLINE` | on | off | bool | default-off experimental / opt-in | `examples/cli/model_mgr.cpp` |  |
| `CRISPEMBED_ONESHOT_CPU` | value | unset | string | tuning / config | `examples/cli/main.cpp` | coordinator's — a default flip should come with a size/thread rule, not a blanket switch. |
| `CRISPEMBED_PPOCRV6_ONESHOT_GPU` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp` |  |
| `CRISPEMBED_SERVER_MAX_IMAGE_BYTES` | value | unset | int | tuning / config | `examples/server/server.cpp` |  |
| `CRISPEMBED_SERVER_MAX_INFLIGHT` | value | unset | int | tuning / config | `examples/server/load_control.h` |  |
| `CRISPEMBED_SERVER_METRICS` | on | off | bool | default-off experimental / opt-in | `examples/server/load_control.h` |  |
| `CRISPEMBED_SERVER_OPENAI_CHAT` | on | off | bool | default-off experimental / opt-in | `examples/server/server.cpp` | the same crispembed_ocr_model_recognize call /ocr/model makes, with the same per-request prompt / max_tokens save-and-restore. |
| `CRISPEMBED_SERVER_RERANK_RAW_SCORES` | on | off | bool | default-off experimental / opt-in | `examples/server/server.cpp` | Read once: a per-request getenv in a served path is pointless work. |
| `CRISPEMBED_SERVER_RETRY_AFTER` | value | unset | int | tuning / config | `examples/server/load_control.h` |  |
| `CRISPEMBED_SERVER_USAGE_TOKENS` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `examples/server/server.cpp` | API usage reports observed tokenizer counts; CRISPEMBED_SERVER_USAGE_TOKENS=0 restores the historical zeros. |

### Rust bindings

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LFM2_ENCODER_MODEL` | value | unset | path | tuning / config | `crispembed/tests/lfm2_encoder_live.rs` |  |
| `CRISPEMBED_SYS_LIB_DIR` | value | unset | path | tuning / config | `crispembed-sys/build.rs` |  |

### adair

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ADAIR_BENCH` | on | off | bool | debug | `src/adair.cpp` |  |
| `CRISPEMBED_ADAIR_TILE` | value | unset | int | tuning / config | `src/adair.cpp` |  |

### bert_ner

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_BERT_NER_BENCH` | on | off | bool | debug | `src/bert_ner.cpp` |  |

### bidirlm_audio

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_BIDIRLM_AUDIO_BENCH` | on | off | bool | debug | `src/bidirlm_audio.cpp` |  |

### bidirlm_vision

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_BIDIRLM_VISION_BENCH` | on | off | bool | debug | `src/bidirlm_vision.cpp` |  |

### bttr_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_BTTR_BENCH` | on | off | bool | debug | `src/bttr_ocr.cpp` |  |

### cc_detect

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_CC_DETECT_BENCH` | on | off | bool | debug | `src/cc_detect.cpp` |  |

### clip_text_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_CLIP_TEXT_BENCH` | on | off | bool | debug | `src/clip_text_embed.cpp` |  |
| `CRISPEMBED_CLIP_TEXT_KEEP_EOS` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/clip_text_embed.cpp` |  |
| `CRISPEMBED_IMATRIX_OUT` | value | unset | path | tuning / config | `src/clip_text_embed.cpp`, `src/gliner_ner.cpp`, `src/imatrix.cpp` | When calibrating an imatrix, build a scheduler so the collector can hook the per-node eval callback (the fast gallocr path below has no such hook). |

### cnn_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_CNN_EMBED_BENCH` | on | off | bool | debug | `src/cnn_embed.cpp` |  |

### core

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ARCH_HPARAMS` | value | unset | value | tuning / config | `src/core/hparam_keys.h` |  |
| `CRISPEMBED_BPE_LEGACY_WHITESPACE` | on | off | bool | default-off experimental / opt-in | `src/core/bpe.h`, `src/qwen2vl_ocr.cpp`, `src/tokenizer_bpe.cpp` | prompt, while newline-free text is unaffected (1.000000). CRISPEMBED_BPE_LEGACY_WHITESPACE=1 restores the old behavior. |
| `CRISPEMBED_C2PA_CERT` | value | unset | value | tuning / config | `src/core/image_out.cpp`, `src/core/image_out.h` |  |
| `CRISPEMBED_C2PA_KEY` | value | unset | value | tuning / config | `src/core/image_out.cpp`, `src/core/image_out.h` |  |
| `CRISPEMBED_CONV1X1_FAST` | on | off | bool | default-off experimental / opt-in | `src/core/cpu_ops.h` |  |
| `CRISPEMBED_CONV2D_GEMM` | value | unset | value | tuning / config | `src/core/cpu_ops.h` |  |
| `CRISPEMBED_CONV2D_MK` | value | unset | value | tuning / config | `src/core/cpu_ops.h` | consume (task 3 / R6 follow-on; NOT bitwise vs the reference — its acceptance is per-engine decoded output) and implies the GEMM path. |
| `CRISPEMBED_CONV2D_THREADS` | value | unset | value | tuning / config | `src/core/cpu_ops.h` |  |
| `CRISPEMBED_CONVDW_FAST` | on | off | bool | default-off experimental / opt-in | `src/core/cpu_ops.h` |  |
| `CRISPEMBED_CRISPEMBED_BENCH` | on | off | bool | debug | `src/crispembed.cpp` |  |
| `CRISPEMBED_DEBUG_ENCODE` | value | unset | string | debug | `src/crispembed.cpp` |  |
| `CRISPEMBED_DEBUG_TOKENS` | value | unset | string | debug | `src/crispembed.cpp` | Used by tests/parity_layers_bert.py to diff against an HF tokenizer without exposing a tokenize-only public API. |
| `CRISPEMBED_DECODER_PREFIX_CACHE` | value | unset | value | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_DOT_WIDE` | on | off | bool | default-off experimental / opt-in | `src/core/cpu_ops.h` | times per page. This is initialised once before main and reads as a plain load, so the branch is perfectly predicted. |
| `CRISPEMBED_EMBED_RAW_NORM` | on | off | bool | default-off experimental / opt-in | `src/crispembed.cpp` | uniform scale error is invisible to cosine parity. Print it on request — same lever as CRISPEMBED_DECODER_EMBED_RAW_NORM on the decoder path. |
| `CRISPEMBED_ENCODER_4D` | value | unset | string | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_ENCODER_4D_GROUP` | value | unset | int | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_ENCODER_NO_SWA` | value | unset | string | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_ENCODER_PACKED` | value | unset | string | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_ENCODER_PACK_MAXTOK` | value | unset | int | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_FOO_REF` | value | unset | path | tuning / config | `src/core/env_value.h` | if (const char * ref = core_env::value("CRISPEMBED_FOO_REF")) { ... } |
| `CRISPEMBED_GGUF_REPARSE` | value | unset | string | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_GPU_PREF_CPU_LEGACY` | value | unset | value | tuning / config | `src/core/gpu_backend_pref.h` |  |
| `CRISPEMBED_IMAGE_FORMAT` | value | unset | string | tuning / config | `src/core/image_out.h` |  |
| `CRISPEMBED_INIT_BENCH` | on | off | bool | debug | `src/core/init_bench.h` | Shared value-parsed semantics (set and not "0" => on); see core/env_gate.h. |
| `CRISPEMBED_MARK_GENERATED` | value | unset | string | tuning / config | `src/core/provenance.h` |  |
| `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB` | value | unset | value | tuning / config | `src/core/metal_pipeline_cache_policy.h` |  |
| `CRISPEMBED_METAL_RESIDENCY` | on | off | bool | default-off experimental / opt-in | `src/core/gguf_loader.cpp` |  |
| `CRISPEMBED_MOE_NO_REPEAT` | value | unset | value | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_OCR_POOL_SIZE` | value | unset | value | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_PAIR_ROBERTA_SEP` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/crispembed.cpp` | (RoBERTa family; DeBERTa-v3 [CLS]/[SEP] and BERT keep one separator). CRISPEMBED_PAIR_ROBERTA_SEP=0 restores the single separator. |
| `CRISPEMBED_POS_EMBD_TRUNCATE` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/crispembed.cpp` | max_position_embeddings - 2 and keeps the special tokens; so do we now. CRISPEMBED_POS_EMBD_TRUNCATE=0 restores the old (aborting) limit. |
| `CRISPEMBED_RAM_GUARD` | value | unset | value | tuning / config | `src/core/ram_guard.h` |  |
| `CRISPEMBED_RAM_GUARD_AVAILABLE_MB` | value | unset | int | tuning / config | `src/core/low_memory.h`, `src/core/ram_guard.h` |  |
| `CRISPEMBED_RERANK_POOLER_GELU_ERF` | value | unset | string | tuning / config | `src/crispembed.cpp` |  |
| `CRISPEMBED_SERVER_LEGACY_JSON` | value | unset | value | tuning / config | `src/core/json.h` |  |
| `CRISPEMBED_SHARED_GPU_BACKEND` | on | off | bool | default-off experimental / opt-in | `src/core/gpu_backend_pref.h` |  |
| `CRISPEMBED_SPARSE_KEEP_SPECIAL` | on | off | bool | default-off experimental / opt-in | `src/crispembed.cpp` | (below-bar panel: 0/77 identical term sets, min cosine 0.675 on "hello"). Default on; CRISPEMBED_SPARSE_KEEP_SPECIAL=1 restores the old output. |
| `CRISPEMBED_STRICT_HPARAMS` | value | unset | value | tuning / config | `src/core/hparam_keys.h` |  |
| `CRISPEMBED_VERBOSE_OCR` | value | unset | value | debug | `src/crispembed.cpp` | Enable verbose logging via environment variable |
| `CRISPEMBED_WARN_UNK` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/crispembed.cpp` |  |
| `CRISPEMBED_WASM_Q8_0_ROUND` | on | off | bool | default-off experimental / opt-in | `src/crispembed.cpp` | value-parsed opt-in is enabled. Settings belong to this backend/context; changing another context's environment cannot rewrite an existing plan. |

### crispembed_diff

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DIFF_COS_THRESHOLD` | value | unset | value | tuning / config | `src/crispembed_diff.h` |  |
| `CRISPEMBED_DIFF_DUMP_DIR` | value | unset | path | debug | `src/crispembed_diff.h` | float32 <dir>/<name>.f32, so a summary number that looks wrong can be taken apart row by row outside the binary (row order, token count). |

### dat_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DAT_SR_BENCH` | on | off | bool | debug | `src/dat_sr.cpp` |  |

### decoder_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DECODER_EMBED_BENCH` | on | off | bool | debug | `src/decoder_embed.cpp` |  |
| `CRISPEMBED_DECODER_EMBED_RAW_NORM` | on | off | bool | default-off experimental / opt-in | `src/decoder_embed.cpp` | caller can see (the returned vector is unit-length by contract), so a uniform scale error is invisible to cosine parity. Print it on request. |
| `CRISPEMBED_DECODER_PREFIX_DEBUG` | on | off | bool | debug | `src/decoder_embed.cpp` |  |
| `CRISPEMBED_DUMP_LAYERS` | on | off | bool | debug | `src/crispembed.cpp`, `src/decoder_embed.cpp` |  |
| `CRISPEMBED_DUMP_LAYERS_GGUF` | value | unset | value | debug | `src/crispembed.cpp`, `src/decoder_embed.cpp` | tools/dump_encoder_reference.py: emb_ln_out == HF hidden_states[0], layer_i == HF hidden_states[i+1]. |
| `CRISPEMBED_SKIP_DEEPSTACK` | on | off | bool | default-off experimental / opt-in | `src/decoder_embed.cpp` | splice + 3D MRoPE. If parity improves with DEEPSTACK_OFF=1, the bug is in the deepstack hook; otherwise look at splice / MRoPE. |

### deepseek_ocr2

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DEEPSEEK_OCR2_BENCH` | on | off | bool | debug | `src/deepseek_ocr2.cpp` | ppocrv6 convention, without ppocrv6's stage-entry correction: no stage here loads anything). |

### dewarp

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DEWARP_BENCH` | on | off | bool | debug | `src/dewarp.cpp` |  |

### easyocr_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_EASYOCR_CONTRAST_RETRY` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/easyocr_ocr.cpp` | strictly more confident. Default on; CRISPEMBED_EASYOCR_CONTRAST_RETRY=0 keeps the single pass. |
| `CRISPEMBED_EASYOCR_FLOOR_WIDTH` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/easyocr_ocr.cpp` | matched Python on only 1/2, 6/11 and 21/36 page lines (fox, scan_strip, scan_page_pd) against 2/2, 11/11 and 36/36 with truncation. |
| `CRISPEMBED_EASYOCR_LIBPNG_GREY` | on | off | bool | default-off experimental / opt-in | `src/easyocr_ocr.cpp` | Python's readtext(<png path>); it differs by -1 on ~half the pixels of a scan and is only meaningful when the pixels came from a PNG file. |
| `CRISPEMBED_EASYOCR_LSTM_STATE_RESET` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/easyocr_ocr.cpp` | Default on; CRISPEMBED_EASYOCR_LSTM_STATE_RESET=0 restores the legacy first-match reset for A/B only. |
| `CRISPEMBED_EASYOCR_POISON_COMPUTE` | on | off | bool | default-off experimental / opt-in | `src/easyocr_ocr.cpp` | allocation (macOS, Metal, mmap-backed glibc) and stale only when glibc reuses a freed heap block — which is how the x86 CPU line drop surfaced. |

### easyocr_pipeline

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_EASYOCR_DEBUG_REGIONS` | on | off | bool | debug | `src/easyocr_pipeline.cpp` | stderr line per recognized region. Added for the CUDA audit, where the x86 CPU arm dropped a line that CUDA, Metal and the M1 CPU all keep. |
| `CRISPEMBED_EASYOCR_STAGE_BENCH` | on | off | bool | debug | `src/easyocr_pipeline.cpp` | recognize call itself, and counts how many width rebuilds the page actually triggered -- the number EASYOCR_WIDTH_SORT exists to reduce. |

### esrgan_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ESRGAN_BENCH` | on | off | bool | debug | `src/esrgan_sr.cpp` |  |
| `CRISPEMBED_ESRGAN_GPU` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/esrgan_sr.cpp` |  |
| `CRISPEMBED_ESRGAN_TILE` | value | unset | int | tuning / config | `src/esrgan_sr.cpp` |  |

### face_align

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_FACE_ALIGN_BENCH` | on | off | bool | debug | `src/face_align.cpp` |  |

### fireredpunc

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_FIREREDPUNC_HF_TOK` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/fireredpunc.cpp` |  |
| `CRISPEMBED_FIREREDPUNC_SEP` | on | off | bool | default-off experimental / opt-in | `src/fireredpunc.cpp` |  CRISPEMBED_FIREREDPUNC_SEP=1 restores the old shape for bisection. |

### gemma_decision

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_GEMMA_DECISION` | on | off | bool | default-off experimental / opt-in | `src/gemma_decision.cpp` |  |
| `CRISPEMBED_GEMMA_DECISION_DUMP` | on | off | bool | debug | `src/gemma_decision.cpp` |  |
| `CRISPEMBED_GEMMA_DECISION_DUMP_PATH` | value | unset | path | debug | `src/gemma_decision.cpp` |  |
| `CRISPEMBED_GEMMA_DECISION_F32_DOT` | on | off | bool | default-off experimental / opt-in | `src/gemma_decision.cpp` |  |
| `CRISPEMBED_GEMMA_DECISION_LEGACY_GELU` | on | off | bool | default-off experimental / opt-in | `src/gemma_decision.cpp` |  |

### gliner_ner

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_FORCE_CPU` | value | unset | string | tuning / config | `src/crispembed.cpp`, `src/gliner_ner.cpp` | Init backend |
| `CRISPEMBED_GLINER_BENCH` | on | off | bool | debug | `src/gliner_ner.cpp` |  |

### glm_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_GLM_OCR_BENCH` | on | off | bool | debug | `src/glm_ocr.cpp` |  |
| `CRISPEMBED_GLM_OCR_SCALAR_MERGER` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` |  |

### got_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE` | on | off | bool | default-off experimental / opt-in | `src/got_ocr.cpp` | Qianwen RESEARCH LICENSE: non-commercial, and no use of the model or its outputs to improve any other large language model. |
| `CRISPEMBED_GOT_OCR_BENCH` | on | off | bool | debug | `src/got_ocr.cpp` |  |
| `CRISPEMBED_GOT_OCR_DEBUG` | on | off | bool | debug | `src/got_ocr.cpp` |  |
| `CRISPEMBED_GOT_OCR_DUMP_DIR` | value | unset | path | debug | `src/got_ocr.cpp` | diagnostic: pixels + ids |
| `CRISPEMBED_GOT_OCR_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/got_ocr.cpp` |  |
| `CRISPEMBED_GOT_OCR_ORIG_NGRAM` | on/off | per-gate default | bool | tuning / config | `src/got_ocr.cpp` | No-repeat-ngram rule (see the gate block above argmax_no_repeat_ngram). |
| `CRISPEMBED_GOT_OCR_ORIG_SYSTEM` | on/off | per-gate default | bool | tuning / config | `src/got_ocr.cpp` | second line keeps 8 spaces of source indentation; tiktoken splits that into 286 ("       ", 7 spaces) + 1446 (" You"). |
| `CRISPEMBED_GOT_OCR_PIL_BICUBIC` | on/off | per-gate default | bool | tuning / config | `src/got_ocr.cpp` |  |
| `CRISPEMBED_GOT_OCR_SCALAR_NECK` | on | off | bool | default-off experimental / opt-in | `src/got_ocr.cpp` |  |
| `CRISPEMBED_GOT_OCR_SCALAR_PATCH` | on | off | bool | default-off experimental / opt-in | `src/got_ocr.cpp` |  |

### granite_vision_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_GRANITE_CPU` | on | off | bool | default-off experimental / opt-in | `src/granite_vision_ocr.cpp` |  |
| `CRISPEMBED_GRANITE_LLM_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/granite_vision_ocr.cpp` | last token's per-layer hidden states — the gv_llm_decode_step scalar path does NOT exercise the graph, so this is the only way to diff it. |
| `CRISPEMBED_GRANITE_LLM_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/granite_vision_ocr.cpp` |  |
| `CRISPEMBED_GRANITE_OCR_BENCH` | on | off | bool | debug | `src/granite_vision_ocr.cpp` |  |
| `CRISPEMBED_GRANITE_VIS_DBG` | on | off | bool | debug | `src/granite_vision_ocr.cpp` |  |
| `CRISPEMBED_GRANITE_VIS_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/granite_vision_ocr.cpp` | F16-table gelu + Q8_0-quantized activations in the attn matmuls — is fixed inside gv_run_vit_graph via explicit F32 gelu + CPU F32 weight casts.) |

### hat_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_HAT_SR_BENCH` | on | off | bool | debug | `src/hat_sr.cpp` |  |

### hmer_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_HMER_BENCH` | on | off | bool | debug | `src/hmer_ocr.cpp` |  |

### inpaint

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INPAINT_PROFILE` | on | off | bool | debug | `src/inpaint.cpp`, `src/inpaint_graph.cpp`, `src/latent_inpaint.cpp` +1 | endif |
| `CRISPEMBED_INPAINT_SHA256_ACCEL` | on | off | bool | default-off experimental / opt-in | `src/inpaint.cpp`, `src/latent_inpaint.cpp` |  |
| `CRISPEMBED_MIGAN` | on/value | off | string | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp`, `src/inpaint.cpp`, `src/migan.cpp` +1 | Both init and run require core_env::on("CRISPEMBED_MIGAN"). The expected |
| `CRISPEMBED_MIGAN512` | on | off | bool | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp`, `src/inpaint.cpp`, `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_QUANT` | on | off | bool | default-off experimental / opt-in | `src/inpaint.cpp`, `src/migan.cpp`, `src/migan.h` | core_env::on("CRISPEMBED_MIGAN_QUANT") and the reviewed F32 GGUF SHA. |

### inpaint_graph

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INPAINT_BACKEND` | value | unset | value | tuning / config | `src/inpaint_graph.cpp` |  |
| `CRISPEMBED_INPAINT_CPU_ATTENTION_TILES` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph.cpp` |  |
| `CRISPEMBED_INPAINT_CPU_DEPTHWISE` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph.cpp` |  |
| `CRISPEMBED_INPAINT_FAST` | on | off | bool | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp`, `src/inpaint_graph.cpp`, `src/migan.cpp` |  |
| `CRISPEMBED_INPAINT_GEMM_LAYOUT` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph.cpp` |  |
| `CRISPEMBED_INPAINT_REFLECT_GATHER` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph.cpp` |  |

### inpaint_graph_spectral

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INPAINT_CPU_FFT` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph_spectral.cpp` |  |
| `CRISPEMBED_INPAINT_CPU_GLA` | on | off | bool | default-off experimental / opt-in | `src/inpaint_graph_spectral.cpp` | Default-off measured CPU candidate. CUDA/Metal keep the portable graph: their native GLA support/geometry differs and is not qualified here. |

### inpaint_weights

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INPAINT_QUANT` | on | off | bool | default-off experimental / opt-in | `src/inpaint_weights.cpp` |  |
| `CRISPEMBED_INPAINT_QUANT_MATMUL` | on | off | bool | default-off experimental / opt-in | `src/inpaint_weights.cpp` |  |

### instructir

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INSTRUCTIR_BENCH` | on | off | bool | debug | `src/instructir.cpp` |  |
| `CRISPEMBED_INSTRUCTIR_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/instructir.cpp` |  |
| `CRISPEMBED_INSTRUCTIR_TILE` | value | unset | int | tuning / config | `src/instructir.cpp` |  |

### internvl2_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INTERNVL2_ADD_BOS` | value | unset | int | tuning / config | `src/internvl2_ocr.cpp` |  |
| `CRISPEMBED_INTERNVL2_BENCH` | on | off | bool | debug | `src/internvl2_ocr.cpp` |  |
| `CRISPEMBED_INTERNVL2_PROMPT` | value | unset | value | tuning / config | `src/internvl2_ocr.cpp` | description, a verbatim transcription, or an immediate <\|end\|> depending only on the instruction, and that is only findable by trying. |
| `CRISPEMBED_MAX_PIXELS` | value | unset | int | tuning / config | `src/internvl2_ocr.cpp`, `src/lightonocr.cpp`, `src/qwen2vl_ocr.cpp` | Allow runtime override for CPU-friendly inference (fewer tiles = faster) |

### josev2b_answer_serializer

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_JOSEV_ANSWER_SERIALIZER` | on | off | bool | default-off experimental / opt-in | `src/josev2b_answer_serializer.cpp` |  |

### josev2b_backbone

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_JOSEV_BACKBONE` | on | off | bool | default-off experimental / opt-in | `src/josev2b_backbone.cpp`, `src/josev2b_composed.cpp`, `src/qwen35_decoder.cpp` |  |
| `CRISPEMBED_JOSEV_QUANT` | on | off | bool | default-off experimental / opt-in | `src/josev2b_backbone.cpp`, `src/josev2b_composed.cpp`, `src/josev2b_text_frontend.cpp` |  |
| `CRISPEMBED_QWEN35_OCR` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/josev2b_backbone.cpp` +21 | 'minicpm-ocr' registry alias (same default as the HTTP server). The decoder reuses the qwen35 kernels, so both gates are still required. |
| `CRISPEMBED_QWEN35_QUANT` | on | off | bool | default-off experimental / opt-in | `src/josev2b_backbone.cpp`, `src/lighton4b_native_context.h`, `src/qwen35_lighton4b_text.h` +1 |  |

### josev2b_composed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_JOSEV_COMPOSED` | on | off | bool | default-off experimental / opt-in | `src/josev2b_composed.cpp` |  |
| `CRISPEMBED_JOSEV_HEAD` | on | off | bool | default-off experimental / opt-in | `src/josev2b_composed.cpp`, `src/josev2b_head.cpp`, `src/josev2b_head.h` | No backend fallback. Both init/run require core_env::on("CRISPEMBED_JOSEV_HEAD"). |
| `CRISPEMBED_JOSEV_PACKED` | on | off | bool | default-off experimental / opt-in | `src/josev2b_composed.cpp`, `src/josev2b_head.cpp`, `src/josev2b_text_frontend.cpp` |  |

### josev2b_text_frontend

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_JOSEV_TEXT_FRONTEND` | on | off | bool | default-off experimental / opt-in | `src/josev2b_text_frontend.cpp` |  |

### kie_pipeline

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_KIE_PIPELINE_BENCH` | on | off | bool | debug | `src/kie_pipeline.cpp` |  |

### latent_inpaint

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_INPAINT_GRAPH` | on | off | bool | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp`, `src/latent_inpaint.cpp`, `src/migan.cpp` |  |
| `CRISPEMBED_MOEBIUS` | value | unset | string | tuning / config | `examples/cli/inpaint_cli.cpp`, `src/latent_inpaint.cpp`, `src/latent_inpaint_unet.cpp` |  |
| `CRISPEMBED_PIXELHACKER` | value | unset | string | tuning / config | `examples/cli/inpaint_cli.cpp`, `src/latent_inpaint.cpp`, `src/latent_inpaint_unet.cpp` |  |

### latent_inpaint_graph_unet

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM` | on | off | bool | default-off experimental / opt-in | `src/latent_inpaint_graph_unet.cpp` |  |
| `CRISPEMBED_MOEBIUS_LOCAL_REASSOC` | on | off | bool | default-off experimental / opt-in | `src/latent_inpaint_graph_unet.cpp` |  |
| `CRISPEMBED_MOEBIUS_RELATIVE_REASSOC` | on | off | bool | default-off experimental / opt-in | `src/latent_inpaint_graph_unet.cpp` |  |

### layout_detect

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ACCEPT_OXR_LICENSE` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp`, `src/oxr_document_api.cpp`, `src/oxr_layout_api.cpp` +1 | OXR License gate (Apache-2.0 plus additional terms, licenses/oxr-LICENSE), identical to src/oxr_ocr.cpp: every surface loading these weights. |
| `CRISPEMBED_LAYOUT_DEBUG` | on | off | bool | debug | `src/layout_detect.cpp` | Backbone Verify stem weight is the folded version |
| `CRISPEMBED_LAYOUT_DETECT_BENCH` | on | off | bool | debug | `src/layout_detect.cpp` |  |
| `CRISPEMBED_LAYOUT_GPU_HP_MM` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/layout_detect.cpp` |  |
| `CRISPEMBED_LAYOUT_GPU_PREC_F32` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/layout_detect.cpp` | heron/: regions equal CPU on 5/5 pages, the half-tile default added a German region); CRISPEMBED_LAYOUT_GPU_PREC_F32=0 / CRISPEMBED_LAYOUT_GPU_HP_MM=0 opt out. |
| `CRISPEMBED_LAYOUT_VALPROJ_GPU` | value | unset | value | tuning / config | `src/layout_detect.cpp` |  |
| `CRISPEMBED_OXR` | on/value | off | value | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/layout_detect.cpp` +4 | OXR License gate (Apache-2.0 plus additional terms, licenses/oxr-LICENSE), identical to src/oxr_ocr.cpp: every surface loading these weights. |
| `CRISPEMBED_OXR_LAYOUT_DW_DIRECT` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/layout_detect.cpp` | GPU forwards default to the direct kernel (region-exact on Metal, faster); CRISPEMBED_OXR_LAYOUT_DW_DIRECT=0 restores im2col there. CPU default unchanged. |
| `CRISPEMBED_OXR_LAYOUT_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/layout_detect.cpp` |  |
| `CRISPEMBED_OXR_LAYOUT_GPU_HP_MM` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/layout_detect.cpp` | process-wide; an explicit GGML_METAL_PREC_F32_MM wins). Opt out with CRISPEMBED_OXR_LAYOUT_GPU_HP_MM=0 (mat-vec PREC_F32 route, ~2.5x slower). |
| `CRISPEMBED_OXR_LAYOUT_GPU_PREC_F32` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/layout_detect.cpp` |  |
| `CRISPEMBED_OXR_LAYOUT_GPU_SPLIT_F16` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp` |  |
| `CRISPEMBED_OXR_LAYOUT_MM_SHAPES` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp` | diagnostic: GEMM shapes and FLOPs |
| `CRISPEMBED_OXR_LAYOUT_TRACE` | on | off | bool | debug | `src/layout_detect.cpp` |  |

### lfm2_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LFM2_EMBED_BENCH` | on | off | bool | debug | `src/lfm2_embed.cpp` |  |
| `CRISPEMBED_LFM2_ENCODER` | off/on/set | mixed (see code) | bool | default-off experimental / opt-in | `crispembed/tests/lfm2_encoder_live.rs`, `src/lfm2_embed.cpp` | Added special tokens bypass byte-level pre-tokenization, exactly as in HF tokenizer.json. Split ordinary spans before BPE, preserving spaces. |
| `CRISPEMBED_LFM2_F32_DOT` | on | off | bool | default-off experimental / opt-in | `src/lfm2_embed.cpp` |  |
| `CRISPEMBED_LFM2_F32_MATMUL` | on | off | bool | default-off experimental / opt-in | `src/lfm2_embed.cpp` |  |

### lfm2_vl_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ACCEPT_LFM_LICENSE` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | License gate |

### lighton4b_attention_block

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_attention_block.h`, `src/lighton4b_decoder_compose.h` |  |

### lighton4b_decoder_compose

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_DECODER_COMPOSE` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_decoder_compose.h` |  |
| `CRISPEMBED_LIGHTON4B_LINEAR_BLOCK` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_decoder_compose.h`, `src/lighton4b_linear_block.h` |  |
| `CRISPEMBED_QWEN35_GDN_GROUPED_HEADS` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_decoder_compose.h`, `src/lighton4b_linear_block.h`, `src/qwen35_delta.cpp` +1 |  |

### lighton4b_gguf_inspection

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_GGUF_INSPECTION` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_gguf_inspection.h` |  |
| `CRISPEMBED_LIGHTON4B_VISION_OWNER` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_gguf_inspection.h`, `src/lighton4b_image_compose.h`, `src/lighton4b_vision_owner.cpp` |  |
| `CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_gguf_inspection.h`, `src/lighton4b_model_owner.h` |  |
| `CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_gguf_inspection.h`, `src/qwen35_lighton4b_text.h` | Experimental, default off; borrowed tensor metadata only, never inference admission. |

### lighton4b_image_compose

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_image_compose.h` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_image_compose.h`, `src/lighton4b_vision_bf16.cpp` |  |

### lighton4b_model_owner

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_QUANT` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_model_owner.h`, `src/lighton4b_native_context.h`, `src/qwen35_lighton4b_text.h` +1 | Converter stores all norms/biases in F32 in both F16 recipes. Frozen Q8/Q4 candidates protect every vision tensor as F32. |

### lighton4b_native_context

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_native_context.h` |  |

### lighton4b_tokenizer

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_FRONTEND` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_tokenizer.cpp` |  |

### lighton4b_vision_bf16

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_CONV` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_NORM` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_PATCH_LINEAR` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |
| `CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2` | on | off | bool | default-off experimental / opt-in | `src/lighton4b_vision_bf16.cpp` |  |

### lightonocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTONOCR3` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR3_GROUNDING` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR_BENCH` | on | off | bool | debug | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR_DEBUG` | on | off | bool | debug | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` | --gpu-backend); CRISPEMBED_LIGHTONOCR_FORCE_CPU=1 overrides it, matching the *_FORCE_CPU convention of every sibling engine. |
| `CRISPEMBED_LIGHTONOCR_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR_NO_FLASH` | value | unset | value | tuning / config | `src/lightonocr.cpp` |  |
| `CRISPEMBED_LIGHTONOCR_NO_KV` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` | define LOCR_DEBUG env_flag("CRISPEMBED_LIGHTONOCR_DEBUG") |
| `CRISPEMBED_LIGHTONOCR_SCALAR_PATCH` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` | Scalar fallback gated by env var |
| `CRISPEMBED_LIGHTON_DUMP` | value | unset | value | debug | `src/lightonocr.cpp` |  |

### lightonocr3_profile

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LIGHTONOCR3_KV_F32` | on | off | bool | default-off experimental / opt-in | `src/lightonocr3_profile.h` |  |
| `CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS` | on | off | bool | default-off experimental / opt-in | `src/lightonocr3_profile.h` | Pinned Pixtral uses 1e-5 for ln_pre and both norms in every vision block. Preserve the historical path and every legacy profile unless opted in. |

### lilt_kie

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_LILT_KIE_BENCH` | on | off | bool | debug | `src/lilt_kie.cpp` |  |

### marker

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/marker.cpp` +2 | TeleOCR's vision tower is under the Qwen RESEARCH LICENSE (docs/teleocr-license.md); the PDFium text mode loads no model and needs no acknowledgement. |
| `CRISPEMBED_MARKER` | on | off | bool | default-off experimental / opt-in | `src/marker.cpp` |  |
| `CRISPEMBED_TELEOCR` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/marker.cpp` +3 |  |

### math_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MATH_OCR_BENCH` | on | off | bool | debug | `src/math_ocr.cpp` |  |

### migan

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MIGAN_BATCH` | value | unset | int | tuning / config | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_CPU_KERNELS` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_DEQUANT_WEIGHTS` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_FAST_MATMUL` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` | Default keeps the F32 precision hint (Metal: F32 mul_mv_ext/_hp kernels). |
| `CRISPEMBED_MIGAN_GPU` | on | off | bool | default-off experimental / opt-in | `examples/cli/inpaint_cli.cpp`, `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_GPU_PAD_K` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_GPU_PAD_REWRITE` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_GPU_ROLL_PAD` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_GPU_SWAP_POINTWISE` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |
| `CRISPEMBED_MIGAN_GPU_TRANSPOSE_POINTWISE` | on | off | bool | default-off experimental / opt-in | `src/migan.cpp` |  |

### mineru_model_profile

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MINERU25_PRO` | on/value | off | value | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/mineru_model_profile.h` +1 |  |
| `CRISPEMBED_MINERU25_PRO_2605` | value | unset | value | tuning / config | `src/mineru_model_profile.h` |  |

### mineru_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MINERU25_PRO_FULL_DOC` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/mineru_ocr.cpp`, `src/ocr_orchestrator.cpp` | else |

### minicpm_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MINICPM_OCR` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/minicpm_ocr.cpp` +1 | 'minicpm-ocr' registry alias (same default as the HTTP server). The decoder reuses the qwen35 kernels, so both gates are still required. |
| `CRISPEMBED_MINICPM_OCR_DUMP` | value | unset | string | debug | `src/minicpm_ocr.cpp` |  |
| `CRISPEMBED_MINICPM_OCR_FORCE_IDS` | value | unset | string | tuning / config | `src/minicpm_ocr.cpp` |  |
| `CRISPEMBED_MINICPM_OCR_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/minicpm_ocr.cpp`, `src/qwen35_graph_decoder.cpp` | Snapshot once for the entire request. Never silently fall back after a candidate failure, which would mix arithmetic within one decode. |
| `CRISPEMBED_MINICPM_OCR_LOGITS_OUT` | value | unset | path | tuning / config | `src/minicpm_ocr.cpp` |  |
| `CRISPEMBED_MINICPM_OCR_LOW_MEMORY` | value | unset | value | tuning / config | `src/minicpm_ocr.cpp` |  |
| `CRISPEMBED_MINICPM_OCR_TIMING` | on | off | bool | debug | `src/minicpm_ocr.cpp` |  |
| `CRISPEMBED_MINICPM_OCR_VISION_BATCH` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/minicpm_ocr.cpp`, `src/minicpm_ocr_vision.cpp` | CRISPEMBED_MINICPM_OCR_VISION_BATCH (default on since 2026-11-02, =0 opts out): equally sized consecutive parts (the slices) run as one graph of independent ... |
| `CRISPEMBED_MINICPM_OCR_VISION_F32` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/minicpm_ocr.cpp`, `src/minicpm_ocr_vision.cpp` | copies: F16 vision products and the "exact" decoder matmul, both token-identical on the measured panels. CRISPEMBED_MINICPM_OCR_LOW_MEMORY=1 forces this, =0 ... |
| `CRISPEMBED_QWEN35_GRAPH_DECODER` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/minicpm_ocr.cpp`, `src/qwen35_graph_kv_decoder.cpp` | Graph decoder default on for MiniCPM since 2026-11-02 (=0 opts out); it yields to the exclusive stateless CRISPEMBED_MINICPM_OCR_GRAPH candidate unless force... |
| `CRISPEMBED_QWEN35_WEIGHT_CACHE` | on | off | bool | default-off experimental / opt-in | `src/minicpm_ocr.cpp`, `src/qwen35_linear.cpp`, `src/qwen35_ocr.cpp` |  |

### minicpm_ocr_vision

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MINICPM_OCR_VISION_FLASH` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/minicpm_ocr_vision.cpp` | explicit softmax(QK^T)V. Default on since the 2026-11-02 A/B (5 parity cases + 20 real scans, IDs identical); CRISPEMBED_MINICPM_OCR_VISION_FLASH=0 opts out. |

### mixtex_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MIXTEX_BENCH` | on | off | bool | debug | `src/mixtex_ocr.cpp` |  |
| `CRISPEMBED_MIXTEX_DEBUG` | on | off | bool | debug | `src/mixtex_ocr.cpp` | Diff: after self-attention (step 0, layer 0) |
| `CRISPEMBED_MIXTEX_TRACE` | on | off | bool | debug | `src/mixtex_ocr.cpp` |  |

### moedern_colbert

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEDERN_COLBERT` | on | off | bool | default-off experimental / opt-in | `src/moedern_colbert.cpp`, `src/moedern_family_engine.cpp`, `src/moedern_text.cpp` +1 |  |

### moedern_family_engine

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEDERN_FAMILY` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text.cpp`, `src/moedern_text_frontend.cpp` | else |
| `CRISPEMBED_MOEDERN_FAST` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text.cpp`, `src/moedern_text_frontend.cpp` | CRISPEMBED_MOEDERN_FAST lifts the one/two-thread bound for the family fast path only. |
| `CRISPEMBED_MOEDERN_Q4_K_MIXED` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text_frontend.cpp` | Default-off mixed recipe variant of the padded Q4_K profile. |
| `CRISPEMBED_MOEDERN_Q4_K_PADDED` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text.cpp`, `src/moedern_text_frontend.cpp` |  |
| `CRISPEMBED_MOEDERN_Q8_0` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text.cpp`, `src/moedern_text_frontend.cpp` |  |
| `CRISPEMBED_MOEDERN_QUANT` | on | off | bool | default-off experimental / opt-in | `src/moedern_family_engine.cpp`, `src/moedern_text_frontend.cpp` | Quantized weights are opt-in and only for the fast graph path. |

### moedern_text

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEDERN_PACKED` | on | off | bool | default-off experimental / opt-in | `src/moedern_text.cpp`, `src/moedern_text_frontend.cpp` |  |
| `CRISPEMBED_MOEDERN_TEXT_COMPOSED` | on | off | bool | default-off experimental / opt-in | `src/moedern_text.cpp` |  |

### moedern_text_api

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEDERN_PUBLIC_TEXT` | on | off | bool | default-off experimental / opt-in | `src/moedern_text_api.cpp` |  |

### moedern_text_frontend

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_MOEDERN_TEXT_FRONTEND` | on | off | bool | default-off experimental / opt-in | `src/moedern_text_frontend.cpp` |  |

### nafnet_denoise

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_NAFNET_BENCH` | on | off | bool | debug | `src/nafnet_denoise.cpp` |  |
| `CRISPEMBED_NAFNET_TILE` | value | unset | int | tuning / config | `src/nafnet_denoise.cpp` |  |

### nemotron_parse_geometry

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_NEMOTRON_PARSE_GEOMETRY` | value | unset | value | tuning / config | `src/nemotron_parse_geometry.h` |  |

### ocr_detect

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DBNET_DECONV_MATMUL` | on | off | bool | default-off experimental / opt-in | `src/ocr_detect.cpp` | (issue #25); mul_mat is the GEMM it is good at. Same math, so the default path is untouched and both stay selectable for A/B. |
| `CRISPEMBED_OCR_DETECT_BENCH` | on | off | bool | debug | `src/ocr_detect.cpp` |  |
| `CRISPEMBED_OCR_DET_MAX_UPSCALE` | value | unset | float | tuning / config | `src/ocr_detect.cpp` |  |
| `CRISPEMBED_OCR_DET_UPSCALE_FLOOR` | value | unset | int | tuning / config | `src/ocr_detect.cpp` |  |

### ocr_orchestrator

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_EASYOCR_BENCH` | on | off | bool | debug | `src/ocr_orchestrator.cpp` | easyocr_pipeline already validates against the Python reference. 'model_a' is therefore the DBNet GGUF, not a CRAFT one. |
| `CRISPEMBED_EASYOCR_CLEANUP` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_OCR_ORCH_BENCH` | on | off | bool | debug | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_OCR_READING_ORDER_ROWS` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_OVISOCR2_FRONTEND` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/ocr_orchestrator.cpp` +3 |  |
| `CRISPEMBED_OVISOCR2_INFERENCE` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/ocr_orchestrator.cpp` +2 |  |
| `CRISPEMBED_OVISOCR3_FRONTEND` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/ocr_orchestrator.cpp` +2 |  |
| `CRISPEMBED_OVISOCR3_INFERENCE` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `examples/server/server.cpp`, `src/ocr_orchestrator.cpp` +1 |  |
| `CRISPEMBED_PPOCRV6_BENCH` | on | off | bool | debug | `src/ocr_orchestrator.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_CLEANUP` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_PPOCRV6_FIXED_WIDTH` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_FORCE_CPU` | on/set | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `src/ocr_orchestrator.cpp`, `src/ppocrv6_det.cpp` +1 | while user+sys stayed stable. An earlier cross-run wall comparison reported the opposite result.) |
| `CRISPEMBED_PPOCRV6_GRAPH_ACCEPT` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `src/ocr_orchestrator.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_GRAPH_MAX_REGIONS` | value | unset | int | tuning / config | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_PPOCRV6_ONESHOT` | on/set | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `src/ocr_orchestrator.cpp` | the environment wins. MSVC has no setenv, and _putenv_s always overwrites, so the presence check carries that. |
| `CRISPEMBED_PPOCRV6_ONESHOT_CPU_MAX_REGIONS` | value | unset | int | tuning / config | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_PPOCRV6_WIDTH_BUCKET` | value | unset | int | tuning / config | `src/ocr_orchestrator.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_TESSERACT_BENCH` | on | off | bool | debug | `src/ocr_orchestrator.cpp`, `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp`, `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_CROP_DUMP_DIR` | value | unset | path | debug | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_CROP_PAD` | value | unset | value | tuning / config | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_CROP_TRIM_INK` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_CROP_TRIM_THRESHOLD` | value | unset | int | tuning / config | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_MIN_REC_CONFIDENCE` | value | unset | float | tuning / config | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_CLEANUP` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_DEBUG` | on/value | off | value | debug | `src/ocr_orchestrator.cpp`, `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_PROJECTION` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_PARALLEL_LOAD` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_SEG_COVERAGE` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_SEG_GUTTER_BALANCE` | on | off | bool | default-off experimental / opt-in | `src/ocr_orchestrator.cpp` |  |
| `CRISPEMBED_TESSERACT_SEG_ROUTER` | value | unset | string | tuning / config | `src/ocr_orchestrator.cpp` | CRISPEMBED_TESSERACT_SEG_ROUTER=0 restores the dbnet-first default. Value-parsed (the UOCR =0 lesson). |
| `CRISPEMBED_TESSERACT_WORKERS` | value | unset | int | tuning / config | `src/ocr_orchestrator.cpp` |  |

### ocr_pipeline

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DEBUG_ALLOW_OCR_Q4` | value | unset | string | debug | `src/ocr_pipeline.cpp` |  |
| `CRISPEMBED_OCR_PIPELINE_BENCH` | on | off | bool | debug | `src/ocr_pipeline.cpp` |  |

### oxr_document_api

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_OXR_NATIVE_DOC` | on | off | bool | default-off experimental / opt-in | `examples/server/server.cpp`, `src/oxr_document_api.cpp` |  |

### oxr_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_OXR_DEQUANT_VISION` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_FLASH_ATTN` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` | the CPU. Unset = auto: Metal for the qualified F16 recipe (token-exact on the five reference cases, tests/results/oxr/metal-m1/), CPU otherwise. |
| `CRISPEMBED_OXR_GPU_F16_HALF_ACT` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_GPU_FLASH` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_GPU_FLASH_CPU` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_GPU_HP_MM` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/oxr_ocr.cpp` | contexts; CRISPEMBED_OXR_GPU_HP_MM=0 keeps the mat-vec route. Process-wide: the ggml knob is read once; an explicit GGML_METAL_PREC_F32_MM wins. |
| `CRISPEMBED_OXR_GPU_SPLIT_F16` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_GPU_VISION_FLASH` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` | The unmasked vision tower may choose separately: CRISPEMBED_OXR_GPU_VISION_FLASH=1 native flash, =0 dense F32. |
| `CRISPEMBED_OXR_REUSE_DECODE_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_TIMING` | on | off | bool | debug | `src/oxr_ocr.cpp` |  |
| `CRISPEMBED_OXR_VTE_Q8` | on | off | bool | default-off experimental / opt-in | `src/oxr_ocr.cpp` |  |

### pan_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PAN_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/pan_sr.cpp` | PSNR 102 dB vs the CPU graph; receipts under tests/results/gpu-audit/cuda-kaggle/), OFF on Metal (not measured there). |
| `CRISPEMBED_PAN_SR_BENCH` | on | off | bool | debug | `src/pan_sr.cpp` |  |

### parseq_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PARSEQ_BENCH` | on | off | bool | debug | `src/parseq_ocr.cpp` |  |

### pix2struct

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PIX2STRUCT_BENCH` | on | off | bool | debug | `src/pix2struct.cpp` |  |
| `CRISPEMBED_PIX2STRUCT_ENC_GPU` | value | unset | value | tuning / config | `src/pix2struct.cpp` |  |
| `CRISPEMBED_PIX2STRUCT_GGML_DECODE` | value | unset | value | tuning / config | `src/pix2struct.cpp` |  |

### posformer_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_POSFORMER_BENCH` | on | off | bool | debug | `src/posformer_ocr.cpp` |  |
| `CRISPEMBED_POSFORMER_TRAINED_FORWARD` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/posformer_ocr.cpp` |  |

### ppformulanet_l_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PPFN_L_BENCH` | on | off | bool | debug | `src/ppformulanet_l_ocr.cpp` |  |

### ppformulanet_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PPFN_BENCH` | on | off | bool | debug | `src/ppformulanet_ocr.cpp` |  |

### ppocrv6_det

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PPOCRV6_CONV_DIRECT` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_det.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_BENCH` | on | off | bool | debug | `src/ppocrv6_det.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_F32_WEIGHTS` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_det.cpp` | resident half weights. Keep CPU graph parity in F32 and provide an explicit F32 override for backend-diff debugging. |
| `CRISPEMBED_PPOCRV6_DET_GPU` | value | unset | value | tuning / config | `src/ppocrv6_det.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_GPU_LOAD` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_det.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_det.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_PROFILE` | on | off | bool | debug | `src/ppocrv6_det.cpp` |  |
| `CRISPEMBED_PPOCRV6_DET_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_det.cpp` | (the CPU-scalar medium detector was why the medium tier timed out in every benchmark). German CER graph 0.04856 vs scalar 0.04955. |

### ppocrv6_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_PPOCRV6_BATCH_CPU_DECODER` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` | (attention over [120,320]) take ~250 ms of GPU-serial kernel launches — that arithmetic is microseconds of NEON work per item. |
| `CRISPEMBED_PPOCRV6_BATCH_GRAPH` | on/value | off | string | default-off experimental / opt-in | `examples/cli/main.cpp`, `src/ppocrv6_ocr.cpp` | An explicit accept-override of 0 (diagnostic runs demanding the scalar reference) also disables the fused lane. |
| `CRISPEMBED_PPOCRV6_BATCH_GRAPH_CPU_ONLY` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_BATCH_MAX` | value | unset | int | tuning / config | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_GRAPH_BENCH` | on | off | bool | debug | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_GRAPH_DEBUG` | on | off | bool | debug | `src/ppocrv6_ocr.cpp` | recognizer pools [H,W] with a 3x2 kernel and 3x2 stride. Swap the arguments here so graph pooling matches the CPU sequence contract. |
| `CRISPEMBED_PPOCRV6_GRAPH_DEQUANT` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_GRAPH_F16` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_GRAPH_STOP` | value | unset | string | tuning / config | `src/ppocrv6_ocr.cpp` | graph after the backbone so [ppocrv6-graph-bench] attributes compute to backbone vs neck/decoder/head. CRISPEMBED_PPOCRV6_GRAPH_STOP=backbone. |
| `CRISPEMBED_PPOCRV6_NO_BATCH_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_NO_GRAPH` | on | off | bool | default-off experimental / opt-in | `examples/cli/main.cpp`, `src/ppocrv6_ocr.cpp` |  |
| `CRISPEMBED_PPOCRV6_RGB` | on | off | bool | default-off experimental / opt-in | `src/ppocrv6_ocr.cpp` | explicit diagnostic switch so the two source formats can be compared without rebuilding the runtime. |
| `CRISPEMBED_PPOCRV6_WIDTH_FLOOR` | value | unset | int | tuning / config | `src/ppocrv6_ocr.cpp` |  |

### qwen2vl_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_DUMP_MERGER` | value | unset | value | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_DUMP_PATCHES` | value | unset | value | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_MINERU25_PRO_EXACT_GELU` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_MINERU25_PRO_METAL_F32_MATMUL` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_NO_KV_CACHE` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_OLMOCR_LONGEST` | value | unset | int | tuning / config | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_OLMOCR_RAW` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` | document text is what OCR callers want, so strip the block by default; CRISPEMBED_OLMOCR_RAW=1 keeps the full model output. |
| `CRISPEMBED_PADDLEOCR_STD_PROMPT` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QARI_NO_SYSTEM` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QARI_STD_PROMPT` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` | preamble. Set CRISPEMBED_QARI_STD_PROMPT=1 for the generic "Describe this image." fallback instead. |
| `CRISPEMBED_QWEN2VL_BENCH` | on | off | bool | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_FORCE_FROM_CALL` | value | unset | int | tuning / config | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_FORCE_IDS` | value | unset | value | tuning / config | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_METAL_FA` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_METAL_SPLIT_K` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` | tiles keep it at 0.9996 (tests/results/mineru25/metal-m1). =0 restores ggml-metal's default kernels for either profile. |
| `CRISPEMBED_QWEN2VL_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_REF` | value | unset | path | tuning / config | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_STAGE_DUMP` | value | unset | value | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_STAGE_DUMP_CALL` | value | unset | int | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN2VL_STAGE_DUMP_STEPS` | value | unset | int | debug | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_QWEN3VL_SYSTEM` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` | Includes Uni-MuMER: its processor uses the same template (stage diff vs phxember/Uni-MuMER-Qwen3-VL-2B: 11 extra leading tokens). |
| `CRISPEMBED_TELEOCR_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_TELEOCR_METAL_F32_MATMUL` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/qwen2vl_ocr.cpp` |  |
| `CRISPEMBED_TELEOCR_REUSE_DECODE_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |

### qwen35_graph_decoder

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_IMATRIX` | on | off | bool | default-off experimental / opt-in | `src/qwen35_graph_decoder.cpp`, `src/qwen35_linear.cpp` | Preserve calibration identity: observe the original named matrix once, then suppress tile callbacks while retaining all old-path behavior. |

### qwen35_graph_kv_decoder

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_GRAPH_MATMUL` | value | unset | value | tuning / config | `src/qwen35_graph_kv_decoder.cpp` |  |

### qwen35_image_bounds

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_OVISOCR2_LARGE_IMAGES` | on | off | bool | default-off experimental / opt-in | `src/qwen35_image_bounds.h` |  |

### qwen35_linear

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_PARALLEL_F16` | on | off | bool | default-off experimental / opt-in | `src/qwen35_linear.cpp` |  |
| `CRISPEMBED_QWEN35_STREAM_PROJECTION` | on | off | bool | default-off experimental / opt-in | `src/qwen35_linear.cpp` |  |
| `CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES` | value | unset | value | tuning / config | `src/qwen35_linear.cpp` |  |

### qwen35_math

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_PRECISE_RMS` | on | off | bool | default-off experimental / opt-in | `src/qwen35_math.h` |  |

### qwen35_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_XIAOMI_OCR0_FRONTEND` | on | off | bool | default-off experimental / opt-in | `src/qwen35_ocr.cpp`, `src/qwen35_tokenizer.cpp` |  |
| `CRISPEMBED_XIAOMI_OCR0_INFERENCE` | on | off | bool | default-off experimental / opt-in | `src/qwen35_ocr.cpp` |  |

### qwen35_profile

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_PROFILE` | on | off | bool | debug | `src/qwen35_profile.cpp` |  |

### qwen35_tokenizer

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_TOKENIZER_MARKS` | on | off | bool | default-off experimental / opt-in | `src/qwen35_tokenizer.cpp` |  |

### qwen35_vision

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_QWEN35_VISION_CASCADE_NORM` | on | off | bool | default-off experimental / opt-in | `src/qwen35_vision.cpp` |  |
| `CRISPEMBED_QWEN35_VISION_PRECISE_ATTN` | on | off | bool | default-off experimental / opt-in | `src/qwen35_vision.cpp` |  |
| `CRISPEMBED_QWEN35_VISION_PRECISE_NORM` | on | off | bool | default-off experimental / opt-in | `src/qwen35_vision.cpp` |  |
| `CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION` | on | off | bool | default-off experimental / opt-in | `src/qwen35_vision.cpp` |  |

### restormer

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_RESTORMER_BENCH` | on | off | bool | debug | `src/restormer.cpp` |  |
| `CRISPEMBED_RESTORMER_DEBUG` | on | off | bool | debug | `src/restormer.cpp` | Manual check: output[0,0,0] should be sum of 3x3 weighted patches |
| `CRISPEMBED_RESTORMER_GPU` | off/on | mixed (see code) | bool | default-off experimental / opt-in | `src/restormer.cpp` |  |

### safmn_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SAFMN_SR_BENCH` | on | off | bool | debug | `src/safmn_sr.cpp` |  |
| `CRISPEMBED_SAFMN_TILE` | value | unset | int | tuning / config | `src/safmn_sr.cpp` |  |

### sam2

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SAM2_BENCH` | on | off | bool | debug | `src/sam2.cpp` |  |

### scan_cleanup

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SCAN_CLEANUP_BENCH` | on | off | bool | debug | `src/scan_cleanup.cpp` |  |

### scunet_denoise

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SCUNET_BENCH` | on | off | bool | debug | `src/scunet_denoise.cpp` |  |
| `CRISPEMBED_SCUNET_TILE` | value | unset | int | tuning / config | `src/scunet_denoise.cpp` |  |

### smoldocling_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SMOLDOCLING_BENCH` | on | off | bool | debug | `src/smoldocling_ocr.cpp` |  |
| `CRISPEMBED_SMOLDOCLING_SCALAR_PATCH` | on | off | bool | default-off experimental / opt-in | `src/smoldocling_ocr.cpp` |  |

### surya_det

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SURYA_DET_BENCH` | on | off | bool | debug | `src/surya_det.cpp` |  |

### swinir_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SWINIR_DEBUG` | on | off | bool | debug | `src/swinir_sr.cpp` |  |
| `CRISPEMBED_SWINIR_SR_BENCH` | on | off | bool | debug | `src/swinir_sr.cpp` |  |

### table_parse

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TABLE_DEBUG` | on | off | bool | debug | `src/table_parse.cpp` |  |
| `CRISPEMBED_TABLE_PARSE_BENCH` | on | off | bool | debug | `src/table_parse.cpp` |  |

### tbsrn_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TBSRN_SR_BENCH` | on | off | bool | debug | `src/tbsrn_sr.cpp` |  |

### tesseract_lstm

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TESSERACT_BEAM_WIDTH` | value | unset | int | tuning / config | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_DAWG_LOAD` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` | DAWG payloads are opt-in. Existing GGUFs do not contain them, and loading them must not alter the default recognition path. |
| `CRISPEMBED_TESSERACT_DAWG_PREFIX_SCORE` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_DAWG_SCORE` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_DISABLE_INT_CACHE` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_GPU_LOAD` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH` | value | unset | int | tuning / config | `src/tesseract_lstm.cpp` |  |
| `CRISPEMBED_TESSERACT_RECODE_COMPOSE` | value | unset | value | tuning / config | `src/tesseract_lstm.cpp` | opt-in, preserved for Latin A/Bs). The old presence test treated ...COMPOSE=0 as ON — value-parse instead. |
| `CRISPEMBED_TESSERACT_REUSE_SCRATCH` | on | off | bool | default-off experimental / opt-in | `src/tesseract_lstm.cpp` |  |

### tesseract_pageseg

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TESSERACT_COMPONENT_BASELINE` | on | off | bool | default-off experimental / opt-in | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_COMPONENT_DEBUG` | on | off | bool | debug | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_LEGACY_BAND_PROFILE` | on | off | bool | debug | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_LEGACY_BAND_ROWS` | value | unset | value | tuning / config | `src/tesseract_pageseg.cpp` | 24-image arms gate (synthetic 20 + CC0) at equal stage time and takes the Fraktur page CER 0.271 -> 0.218; =0 restores the greedy chain. |
| `CRISPEMBED_TESSERACT_LEGACY_ROW_DEBUG` | on | off | bool | debug | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_BASELINE_TOLERANCE` | value | unset | value | tuning / config | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD` | value | unset | value | tuning / config | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_KEEP_RULES` | on | off | bool | default-off experimental / opt-in | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_ROW_BLOB_BOUNDS` | on | off | bool | default-off experimental / opt-in | `src/tesseract_pageseg.cpp` |  |
| `CRISPEMBED_TESSERACT_PAGESEG_ROW_GAP` | value | unset | value | tuning / config | `src/tesseract_pageseg.cpp` |  |

### text_sr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TEXT_SR_BENCH` | on | off | bool | debug | `src/text_sr.cpp` |  |

### tokenizer

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_WORDPIECE_HF_NORM` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/tokenizer.cpp` | 'CRISPEMBED_WORDPIECE_HF_NORM=0' restores the historical per-byte lowercase for bit-exact comparison against pre-fix output. |
| `CRISPEMBED_WORDPIECE_HF_PRETOK` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/tokenizer.cpp` | 'CRISPEMBED_WORDPIECE_HF_PRETOK=0' restores the historical splitter. 'tokenizer.ggml.pre = "bert"' still forces it on regardless. |
| `CRISPEMBED_WORDPIECE_HF_UNK` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/tokenizer.cpp` |  |

### tokenizer_spm

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_SPM_COLLAPSE_SPACES` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/tokenizer_spm.cpp` | vectors whose direction quantization flips (BGE-M3 Q8_0 ColBERT 0.45-0.59). Default on; CRISPEMBED_SPM_COLLAPSE_SPACES=0 restores the old ids. |
| `CRISPEMBED_SPM_HF_NORM` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/tokenizer_spm.cpp` |  |

### tps_locnet

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_TPS_LOCNET_BENCH` | on | off | bool | debug | `src/tps_locnet.cpp` |  |

### ultrabert_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ULTRABERT` | on | off | bool | default-off experimental / opt-in | `src/ultrabert_embed.cpp` |  |
| `CRISPEMBED_ULTRABERT_QUANT` | on | off | bool | default-off experimental / opt-in | `src/ultrabert_embed.cpp` |  |

### ultrabert_profile

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ULTRABERT_TRUNCATE` | on | off | bool | default-off experimental / opt-in | `src/ultrabert_profile.h` | Explicit, UltraBERT-only single-text right truncation. Keep BOS attended and count it inside the 512-token bound, matching the original wrapper. |

### ultrabert_tokenizer_mode

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN` | on | off | bool | default-off experimental / opt-in | `src/ultrabert_tokenizer_mode.h` |  |

### unlimited_ocr

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_UNLIMITED_OCR2` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | Unlimited-OCR-2 must not silently run v1 attention: its own gate is required. |
| `CRISPEMBED_UNLIMITED_OCR2_FAST` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_FORCE_IDS` | value | unset | value | tuning / config | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_LOGITS_OUT` | value | unset | path | tuning / config | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_LOW_MEMORY` | value | unset | value | tuning / config | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_OUTPUT` | value | unset | value | tuning / config | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_TILES` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_TILES_LEGACY_PREP` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_TILES_LEGACY_UPSAMPLE` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_TILE_BATCH` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_TILE_BATCH_MAX` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` |  |
| `CRISPEMBED_UNLIMITED_OCR2_VISION_CACHE` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/unlimited_ocr.cpp` | Only the separately measured second-generation path opts into this cache and row-parallel projector. Keep the first-generation route intact. |
| `CRISPEMBED_UNLIMITED_OCR2_VISION_F32` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/unlimited_ocr.cpp` | room for the copies (core/low_memory.h; CRISPEMBED_UNLIMITED_OCR2_LOW_MEMORY=1 forces, =0 disables the check) the F16 path runs instead. |
| `CRISPEMBED_UNLIMITED_OCR_BENCH` | on | off | bool | debug | `src/unlimited_ocr.cpp` |  |

### vit_embed

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `CRISPEMBED_VIT_EMBED_BENCH` | on | off | bool | debug | `src/vit_embed.cpp` |  |

### non-prefixed (legacy / third-party)

| Variable | Kind | Default | Type | Status | Where | Hint |
|---|---|---|---|---|---|---|
| `ADAIR_FORCE_CPU` | value | unset | int | tuning / config | `src/adair.cpp` |  |
| `ADAIR_LEGACY_NE3_DIMS` | value | unset | int | tuning / config | `src/adair.cpp` |  |
| `ADAIR_SCALAR` | value | unset | int | tuning / config | `src/adair.cpp` | heavy), so the convs dominate the forward — default ON, opt out with ADAIR_SCALAR=1. The 2D FFT (AFLB) and attention softmax stay SIMD-scalar. |
| `BERT_NER_FORCE_CPU` | value | unset | int | tuning / config | `src/bert_ner.cpp` | purely to pull the GGUF -- the same waste CRISPEMBED_TESSERACT_GPU_LOAD was added to remove. BERT_NER_GPU_LOAD=1 restores it for bisection. |
| `BERT_NER_GPU_LOAD` | on | off | bool | default-off experimental / opt-in | `src/bert_ner.cpp` |  |
| `BTTR_OCR_SCALAR_ENCODER` | on | off | bool | default-off experimental / opt-in | `src/bttr_ocr.cpp` |  |
| `CARGO_CFG_TARGET_OS` | value | unset | value | tuning / config | `crispembed-sys/build.rs` |  |
| `CLIP_TEXT_FORCE_CPU` | value | unset | int | tuning / config | `src/clip_text_embed.cpp` | Load weights — prefer GPU backend when available |
| `CNN_DEBUG` | on | off | bool | debug | `src/cnn_embed.cpp` | Debug: compare output tensors with ONNX reference (CNN_DEBUG=1) |
| `CNN_EMBED_FORCE_CPU` | value | unset | int | tuning / config | `src/cnn_embed.cpp` | Load weights — prefer GPU backend when available |
| `CNN_REF_DIR` | value | unset | path | tuning / config | `src/cnn_embed.cpp` |  |
| `CUDA` | value | unset | value | tuning / config | `src/instructir.cpp`, `src/lightonocr.cpp`, `src/pan_sr.cpp` +1 |  |
| `DAT_DUMP_DIR` | value | unset | path | debug | `src/dat_sr.cpp` |  |
| `DAT_SR_GGML_CONV` | value | unset | int | tuning / config | `src/dat_sr.cpp` | and gate the ggml path behind DAT_SR_GGML_CONV=1 (opt-in for benchmarking / a future batched-graph rewrite). Contrast swinir/scunet (conv-heavy → wins). |
| `DS2_CROP_MODE` | value | unset | string | tuning / config | `src/deepseek_ocr2.cpp` |  |
| `DS2_KV_BUCKET` | value | unset | int | tuning / config | `src/deepseek_ocr2.cpp` |  |
| `DS2_KV_F16` | value | unset | string | tuning / config | `src/deepseek_ocr2.cpp` |  |
| `DS2_LEGACY_DECODE` | value | unset | string | tuning / config | `src/deepseek_ocr2.cpp` | Value-parsed (G2b, same audit class as the DS2_KV_F16 fix 73beea9f): DS2_LEGACY_DECODE=0 must NOT select the legacy path. |
| `DS2_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/deepseek_ocr2.cpp` |  |
| `DS_REF` | value | unset | path | tuning / config | `src/deepseek_ocr2.cpp` | compared against the reference and a cos_min/max_abs line is printed. See tools/dump_deepseek_ocr2_reference.py. |
| `DS_TEXT_TEST` | value | unset | value | tuning / config | `src/deepseek_ocr2.cpp` | Isolation test: DS_TEXT_TEST runs the LLM decoder as a pure language model (no vision) to verify the decoder/MoE/rope in isolation. |
| `EASYOCR_BENCH` | on | off | bool | debug | `src/easyocr_ocr.cpp` |  |
| `EASYOCR_CRAFT_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/easyocr_craft.cpp` |  |
| `EASYOCR_DIFF_DEBUG` | on | off | bool | debug | `src/easyocr_ocr.cpp` |  |
| `EASYOCR_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/easyocr_ocr.cpp` |  |
| `EASYOCR_WIDTH_SORT` | on | off | bool | default-off experimental / opt-in | `src/easyocr_pipeline.cpp` |  |
| `ESRGAN_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/esrgan_sr.cpp` | Use ggml graph path if available |
| `FIREREDPUNC_BACKEND` | value | unset | string | tuning / config | `src/fireredpunc.cpp` | local workaround, 'FIREREDPUNC_BACKEND=cpu' forces CPU and 'FIREREDPUNC_BACKEND=gpu' forces 'init_best()'. |
| `FIREREDPUNC_BENCH` | on | off | bool | debug | `src/fireredpunc.cpp` |  |
| `FIREREDPUNC_DEBUG` | on | off | bool | debug | `src/fireredpunc.cpp` | Dump embedding output for diff-testing |
| `FIREREDPUNC_DUMP_LOGITS` | value | unset | value | debug | `src/fireredpunc.cpp` | the pre-argmax distribution — where quantization/imatrix effects actually live — instead of the thresholded restored-string exact-match. |
| `GGML_METAL_NO_RESIDENCY` | set | set internally | value | internal | `src/core/gguf_loader.cpp` |  |
| `GGML_METAL_PIPELINE_CACHE` | value | unset | value | tuning / config | `src/core/metal_pipeline_cache_policy.h` |  |
| `GGML_METAL_PIPELINE_CACHE_DISABLE` | set/value | unset | value | tuning / config | `src/core/metal_pipeline_cache_policy.h` | Respect an explicit caller decision either way. |
| `GGML_METAL_PREC_F32_MM` | set | set internally | value | internal | `src/layout_detect.cpp`, `src/oxr_ocr.cpp` | else |
| `GLINER_DEBUG` | on | off | bool | debug | `src/gliner_ner.cpp` |  |
| `GLINER_DIFF_REF` | value | unset | path | tuning / config | `src/gliner_ner.cpp` |  |
| `GLM_OCR_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` | Cache DECODE steps only (n_past > 0). Prefill (n_past == 0) has a different graph shape (image splice, full-sequence mRoPE) and stays on the sched. |
| `GLM_OCR_DUMP_EMBEDS` | on | off | bool | debug | `src/glm_ocr.cpp` |  |
| `GLM_OCR_DUMP_LLM_LAYERS` | value | unset | value | debug | `src/glm_ocr.cpp` | Divergence tracing: dump every layer's hidden states (T x D each). |
| `GLM_OCR_DUMP_LOGITS` | value | unset | value | debug | `src/glm_ocr.cpp` | Divergence tracing: append each step's last-token logits row. |
| `GLM_OCR_DUMP_VIS_OUTPUT` | value | unset | value | debug | `src/glm_ocr.cpp` | output (honours GLM_OCR_VISION_MAX_LAYERS). The bisection axis: run the same N on CPU and on Metal and diff the two files. |
| `GLM_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/glm_ocr.cpp` | Prefer GPU backend when available |
| `GLM_OCR_INJECT_MERGER` | value | unset | value | debug | `src/glm_ocr.cpp` | Debug: inject Python reference merger output if GLM_OCR_INJECT_MERGER is set |
| `GLM_OCR_INJECT_VIS` | value | unset | value | debug | `src/glm_ocr.cpp` | Vision encoder output is now in layer_data (vis_D × n_patches) Debug: inject Python reference vision output if GLM_OCR_INJECT_VIS is set |
| `GLM_OCR_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/glm_ocr.cpp` |  |
| `GLM_OCR_NO_ROPE_PERMUTE` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` | artifact works unchanged. GLM_OCR_NO_ROPE_PERMUTE=1 restores the old (mis-paired) behavior for A/B. |
| `GLM_OCR_ROPE_DEBUG` | on | off | bool | debug | `src/glm_ocr.cpp` |  |
| `GLM_OCR_ROPE_MERGE_ORDER` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` |  |
| `GLM_OCR_VISION_BAKE_F32` | value | unset | value | tuning / config | `src/glm_ocr.cpp` | exact, q8_0 dequant is deterministic), the per-image cast nodes become no-ops, and the price is ~4x tower-weight RAM — hence opt-in. |
| `GLM_OCR_VISION_F16MM` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` | cosine collapses. Cast quantized/F16 weights to F32 so matmuls stay F32. GLM_OCR_VISION_F16MM=1 restores the old (lossy) behavior for A/B testing. |
| `GLM_OCR_VISION_FLASH` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` | Explicit F32 attention by default (flash-attn's F16 K/V loses precision on this ViT's massive activations); GLM_OCR_VISION_FLASH=1 to A/B the flash path. |
| `GLM_OCR_VISION_MAX_LAYERS` | value | unset | int | tuning / config | `src/glm_ocr.cpp` |  |
| `GLM_OCR_VISION_PIN_INPUT` | on | off | bool | default-off experimental / opt-in | `src/glm_ocr.cpp` | nothing measured, so the default stays as-is per the A/B rule. GLM_OCR_VISION_PIN_INPUT=1 enables. |
| `GLM_OCR_VISION_ROPE` | value | unset | int | tuning / config | `src/glm_ocr.cpp` | do NOT validate the rope path — regenerate refs with rope before using per-layer diff as a gate. |
| `GOT_OCR_ATTN_CONT` | on | off | bool | default-off experimental / opt-in | `src/got_ocr.cpp` | GOT_OCR_ATTN_CONT=1 restores the copy for regression bisection. (Same removal proven byte-identical for math_ocr, MATH_OCR_ATTN_CONT.) |
| `GOT_OCR_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/got_ocr.cpp` | grow with n_past), so once reserved for the longest graph every subsequent ggml_gallocr_alloc_graph takes the no-realloc fast path. Output-identical. |
| `GOT_OCR_DUMP_VIS` | value | unset | value | debug | `src/got_ocr.cpp` | Diagnostic: dump final vision features (n_vis_tokens x vis_D) for HF diff. |
| `GOT_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/got_ocr.cpp` | Useful to sidestep Metal op gaps and to A/B Metal-vs-CPU outputs while debugging vision-encoder correctness (issue #25). |
| `GOT_OCR_NC_MAX` | value | unset | int | tuning / config | `src/got_ocr.cpp` |  |
| `GOT_OCR_NO_KV_CACHE` | value | unset | int | tuning / config | `src/got_ocr.cpp` | the SAME graph the per-layer parity diff validates — isolates whether the KV-cache path (write/read views, cached RoPE/mask) is what corrupts output. |
| `GOT_OCR_STEP_PROFILE` | on | off | bool | debug | `src/got_ocr.cpp` |  |
| `HAT_SR_FORCE_CPU` | value | unset | int | tuning / config | `src/hat_sr.cpp` |  |
| `HAT_SR_SCALAR` | value | unset | int | tuning / config | `src/hat_sr.cpp` | upsample/conv_last convs run at the 4×-upsampled resolution, so unlike attention-bound DAT the convs are a meaningful fraction of the forward. |
| `HF_HUB_OFFLINE` | value | unset | value | tuning / config | `examples/cli/model_mgr.cpp` |  |
| `HMER_OCR_SCALAR_ENCODER` | on | off | bool | default-off experimental / opt-in | `src/hmer_ocr.cpp` | Value-parsed: HMER_OCR_SCALAR_ENCODER=0 must mean "no", not "yes". The presence test this replaces turned the documented off-switch on. |
| `HOME` | value | unset | value | tuning / config | `examples/cli/model_mgr.cpp`, `src/core/metal_pipeline_cache_policy.h` | else |
| `INSTRUCTIR_LEGACY` | on | off | bool | default-off experimental / opt-in | `src/instructir.cpp` |  |
| `INTERNVL2_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/internvl2_ocr.cpp` | Cache DECODE steps only (n_past > 0). Prefill (n_past == 0) has a different graph shape (image splice) and stays on the sched. |
| `INTERNVL2_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/internvl2_ocr.cpp` | Init backend — prefer GPU when available |
| `LAYOUT_CONV_DIRECT` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp` |  |
| `LAYOUT_CONV_F16` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp` |  |
| `LAYOUT_DEBUG` | on | off | bool | debug | `src/layout_detect.cpp` | Dump encoder intermediates for diff comparison (ggml raw byte order) |
| `LAYOUT_DETECT_FLASH` | on | off | bool | default-off experimental / opt-in | `src/layout_detect.cpp` |  |
| `LAYOUT_DETECT_FORCE_CPU` | value | unset | int | tuning / config | `src/layout_detect.cpp` | Load weights — prefer GPU backend when available |
| `LAYOUT_INJECT_MEMORY` | value | unset | value | debug | `src/layout_detect.cpp` | Debug: also inject HF reference memory (raw encoder features for cross-attn) |
| `LAYOUT_INJECT_REF` | value | unset | path | debug | `src/layout_detect.cpp` | Debug: inject HF reference enc_output for decoder parity testing |
| `LFM2_COLBERT_DIFF_REF` | value | unset | path | tuning / config | `src/lfm2_embed.cpp` | hidden_states PASSES here but colbert_output FAILs, the discrepancy is the ColBERT projection head, not the backbone. |
| `LFM2_VL_ATTN_PREC_F32` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_BILINEAR_RESIZE` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_CPU_PROJECTOR` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | Two spellings because two sessions found this independently and shipped different gate names; both are honoured so either set of notes works. |
| `LFM2_VL_DBG` | on | off | bool | debug | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_DECODE_PROFILE` | on | off | bool | debug | `src/lfm2_vl_ocr.cpp` | — because "decode is slow" is not a finding, and the four are fixed by completely different changes. |
| `LFM2_VL_DIFF_REF` | value | unset | path | tuning / config | `src/lfm2_vl_ocr.cpp` | Diff reference (parity loop) |
| `LFM2_VL_FLASH_ATTN` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | Init backend |
| `LFM2_VL_KV_CACHE` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/lfm2_vl_ocr.cpp` | prefill per token. Keep it: it is the independent cross-check that caught both decode bugs, and it needs no KV cache allocation. |
| `LFM2_VL_KV_VIEW` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_LEGACY_RESIZE` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | pinning min = max upscaled small images to the full budget. LFM2_VL_LEGACY_RESIZE=1 restores the old parameters. |
| `LFM2_VL_LEGACY_TILE_LABELS` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | non-square grid; >= 5.0 is geometric, and that is what the shipped model is prompted with. LFM2_VL_LEGACY_TILE_LABELS=1 reproduces the old form. |
| `LFM2_VL_MULTI_TILE` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/lfm2_vl_ocr.cpp` | splits (7-9 vision encodes and a ~2000-token prefill instead of one and 273), so LFM2_VL_MULTI_TILE=0 keeps the fast single-tile path. |
| `LFM2_VL_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/lfm2_vl_ocr.cpp` |  |
| `LFM2_VL_PER_LAYER_MASK` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | See the causal-mask comment in the attention branch below. |
| `LFM2_VL_PROJ_GELU_TANH` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | made the tanh approximation look right here. LFM2_VL_PROJ_GELU_TANH=1 restores the tanh form for A/B. |
| `LFM2_VL_PROJ_GGML` | off | ON (opt-out `=0`) | bool | production default (opt-out) | `src/lfm2_vl_ocr.cpp` | Two spellings because two sessions found this independently and shipped different gate names; both are honoured so either set of notes works. |
| `LFM2_VL_TV_RESIZE` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | by at most 1/255 on ~100 pixels of 750k elsewhere. LFM2_VL_TV_RESIZE=1 selects the torchvision resampler for A/B. |
| `LFM2_VL_ZERO_CONV_STATE` | on | off | bool | default-off experimental / opt-in | `src/lfm2_vl_ocr.cpp` | Optionally skip conv state extraction (for debugging decode issues) |
| `LILT_KIE_FORCE_CPU` | value | unset | int | tuning / config | `src/lilt_kie.cpp` | Init backend — prefer GPU when available |
| `LOCR_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/lightonocr.cpp` |  |
| `MATH_OCR_ATTN_CONT` | on | off | bool | default-off experimental / opt-in | `src/math_ocr.cpp` | Default is now cont-off; MATH_OCR_ATTN_CONT=1 restores the old copies for regression bisection. |
| `MATH_OCR_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/math_ocr.cpp` | with n_kv), so every per-step alloc takes the no-realloc fast path and the sched's split_graph is skipped. Output-identical. Same trick as got_ocr. |
| `MATH_OCR_DEC_CPU` | value | unset | int | tuning / config | `src/math_ocr.cpp` | Decoder-on-CPU split (opt-in, requires a GPU primary backend). |
| `MATH_OCR_DIFF_REF` | value | unset | path | tuning / config | `src/math_ocr.cpp` | back and compare it to the Python reference — the doc's "first layer < 0.999 = the bug" recipe. Off by default (no perturbation of the normal forward). |
| `MATH_OCR_DUMP_ENC` | value | unset | value | debug | `src/math_ocr.cpp` | DEBUG: dump encoder memory [T,H] |
| `MATH_OCR_FFN_GELU` | value | unset | int | tuning / config | `src/math_ocr.cpp` |  |
| `MATH_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/math_ocr.cpp` |  |
| `MATH_OCR_MEAN` | value | unset | float | tuning / config | `src/math_ocr.cpp` |  |
| `MATH_OCR_NODES` | on | off | bool | default-off experimental / opt-in | `src/math_ocr.cpp` |  |
| `MATH_OCR_PAD` | value | unset | int | tuning / config | `src/math_ocr.cpp` |  |
| `MATH_OCR_PV_BIN` | value | unset | value | tuning / config | `src/math_ocr.cpp` | DEBUG: inject a raw normalized pixel_values buffer (S*S f32, replicated to n_channels) to isolate the graph from native preprocessing (see LEARNINGS). |
| `MATH_OCR_STD` | value | unset | float | tuning / config | `src/math_ocr.cpp` |  |
| `MIXTEX_BILINEAR` | on | off | bool | default-off experimental / opt-in | `src/mixtex_ocr.cpp` | ViTImageProcessor uses bicubic (resample=3), so this diverges from HF at borderline greedy steps; the default below matches HF. |
| `MIXTEX_DEC_DEQUANT_PER_STEP` | on | off | bool | default-off experimental / opt-in | `src/mixtex_ocr.cpp` | Decoder layers |
| `MIXTEX_DEC_REF` | value | unset | path | tuning / config | `src/mixtex_ocr.cpp` | Decoder diff harness |
| `MIXTEX_DIFF_REF` | value | unset | path | tuning / config | `src/mixtex_ocr.cpp` | Diff harness: compare with Python reference |
| `MIXTEX_DUMP` | on | off | bool | debug | `src/mixtex_ocr.cpp` |  |
| `MIXTEX_ENC_STAGE_REF` | value | unset | path | tuning / config | `src/mixtex_ocr.cpp` | Diff: per-stage encoder output |
| `MIXTEX_WMSA_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/mixtex_ocr.cpp` |  |
| `NAFNET_CPU` | on | off | bool | default-off experimental / opt-in | `src/nafnet_denoise.cpp` | the Metal device (idempotent; usually already applied by the crispasr_init_gpu_backend() call above, but NAFNET_FORCE_CPU=1 skips it). |
| `NAFNET_FORCE_CPU` | value | unset | int | tuning / config | `src/nafnet_denoise.cpp` | Forward pass is scalar CPU but weights can reside on GPU (read via ggml_backend_tensor_get). Backend kept alive for tensor access. |
| `NAFNET_LEGACY` | on | off | bool | default-off experimental / opt-in | `src/nafnet_denoise.cpp` | Fused single-graph block (default). Replaces the per-conv mini-graphs + scalar glue. Legacy path via NAFNET_LEGACY=1. |
| `NAFNET_SCALAR` | value | unset | int | tuning / config | `src/nafnet_denoise.cpp` |  |
| `OCR_DETECT_CAPTURE_TAPS` | on | off | bool | default-off experimental / opt-in | `src/ocr_detect.cpp` |  |
| `OCR_DETECT_DEBUG` | on | off | bool | debug | `src/ocr_detect.cpp` |  |
| `OCR_DETECT_DIRECT_CONV` | on | off | bool | default-off experimental / opt-in | `src/ocr_detect.cpp` |  |
| `OCR_DETECT_FORCE_CPU` | value | unset | int | tuning / config | `src/ocr_detect.cpp` | OCR_DETECT_USE_GPU=0 forces CPU, =1 forces the GPU path (either kind); unset = the per-kind default. |
| `OCR_DETECT_SCALAR_SCORE` | on | off | bool | default-off experimental / opt-in | `src/ocr_detect.cpp` | box output is byte-identical. OCR_DETECT_SCALAR_SCORE=1 restores the old per-pixel path for bisection. |
| `OCR_DETECT_USE_GPU` | value | unset | int | tuning / config | `src/ocr_detect.cpp` |  |
| `OUT_DIR` | value | unset | path | tuning / config | `crispembed-sys/build.rs` |  |
| `PAN_SR_FORCE_CPU` | value | unset | int | tuning / config | `src/pan_sr.cpp` |  |
| `PAN_SR_SCALAR` | value | unset | int | tuning / config | `src/pan_sr.cpp` |  |
| `PARSEQ_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/parseq_ocr.cpp` |  |
| `PATH` | value | unset | value | tuning / config | `python/crispembed/_binding.py` |  |
| `PCS_DEBUG` | on | off | bool | debug | `src/pcs.cpp` | Diff-harness dump: PCS_DEBUG=1 prints tokenization + per-token head predictions so they can be compared against the ONNX reference (tools/dump_pcs_reference.... |
| `PCS_DUMP_HIDDEN` | value | unset | value | debug | `src/pcs.cpp` | Diff-harness: dump encoder hidden (row-major [N, d]) for cosine parity vs ONNX. With PCS_DUMP_LAYER set, dump that layer's full [seq_len, d] output instead. |
| `PCS_DUMP_LAYER` | value | unset | int | debug | `src/pcs.cpp` |  |
| `PCS_DUMP_LOGITS` | value | unset | value | debug | `src/pcs.cpp` | just the thresholded restored string. Plain fprintf — no imatrix dep, so it stays identical in the CrispASR copy. |
| `PCS_FLASH_ATTN` | on | off | bool | default-off experimental / opt-in | `src/pcs.cpp` |  |
| `PCS_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/pcs.cpp` | Load weights. PCS_FORCE_CPU pins the CPU backend (diff-harness parity: match the ONNX/onnxruntime CPU reference without GPU float-noise on borderline logits). |
| `PIX2STRUCT_DUMP_ENC` | value | unset | value | debug | `src/pix2struct.cpp` | Parity tracing: dump the encoder output cache ([n_enc, hidden]) |
| `PIX2STRUCT_DUMP_PATCHES` | value | unset | value | debug | `src/pix2struct.cpp` | Parity tracing: dump the packed patch tensor ([max_patches, 2+patch_flat]) |
| `PIX2STRUCT_RAW_IDS` | on | off | bool | default-off experimental / opt-in | `src/pix2struct.cpp` |  |
| `POSFORMER_DUMP` | value | unset | value | debug | `src/posformer_ocr.cpp` | Dump encoder output if requested |
| `POSFORMER_SCALAR_ENCODER` | on | off | bool | default-off experimental / opt-in | `src/posformer_ocr.cpp` |  |
| `PPFNL_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/ppformulanet_l_ocr.cpp` | Prefer GPU backend when available |
| `PPFN_DEBUG` | on | off | bool | debug | `src/ppformulanet_ocr.cpp` | Embedding LayerNorm |
| `PPFN_SCALAR_ENCODER` | on | off | bool | default-off experimental / opt-in | `src/ppformulanet_ocr.cpp` | Input is already preprocessed CHW [3, S, S] normalized data |
| `PPLCNET_ORIENTATION_GRAPH` | on | off | bool | default-off experimental / opt-in | `src/pplcnet_orientation.cpp` |  |
| `PPLCNET_ORIENTATION_GRAPH_ACCEPT` | on | off | bool | default-off experimental / opt-in | `src/pplcnet_orientation.cpp` |  |
| `PPLCNET_ORIENTATION_GRAPH_CPU` | on | off | bool | default-off experimental / opt-in | `src/pplcnet_orientation.cpp` |  |
| `PPLCNET_ORIENTATION_GRAPH_DEBUG` | on | off | bool | debug | `src/pplcnet_orientation.cpp` |  |
| `PPLCNET_ORIENTATION_GRAPH_PIPELINE` | on | off | bool | default-off experimental / opt-in | `src/pplcnet_orientation.cpp` |  |
| `PPOCRV6_REF` | value | unset | path | tuning / config | `src/ppocrv6_ocr.cpp` |  |
| `QWEN2VL_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/qwen2vl_ocr.cpp` | CRISPEMBED_TELEOCR_GPU=1; CRISPEMBED_TELEOCR_GPU=0 or QWEN2VL_OCR_FORCE_CPU=1 force the CPU path everywhere. |
| `QWEN2VL_OCR_NO_WINDOW` | on | off | bool | default-off experimental / opt-in | `src/qwen2vl_ocr.cpp` |  |
| `QWEN_DBG` | on | off | bool | debug | `src/qwen2vl_ocr.cpp` |  |
| `RESTORMER_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/restormer.cpp` |  |
| `SAFMN_SR_LEGACY` | on | off | bool | default-off experimental / opt-in | `src/safmn_sr.cpp` | Fused single-graph forward (default). The per-conv legacy path below is kept for A/B via SAFMN_SR_LEGACY=1. |
| `SAFMN_SR_METAL` | on | off | bool | default-off experimental / opt-in | `src/safmn_sr.cpp` | G4: this opt-in Metal path bypasses crispasr_init_gpu_backend(), so cap the MTLBinaryArchive open here too (idempotent, no-op on CPU default). |
| `SAM2_FORCE_CPU` | on | off | bool | default-off experimental / opt-in | `src/sam2.cpp` |  |
| `SCUNET_FORCE_CPU` | value | unset | int | tuning / config | `src/scunet_denoise.cpp` |  |
| `SCUNET_SCALAR` | value | unset | int | tuning / config | `src/scunet_denoise.cpp` | POINTERS held in the context — not by shape, which can't tell a 1×1 conv (ne=[1,1,IC,OC]) from a 2D linear. |
| `SCUNET_WMSA_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/scunet_denoise.cpp` |  |
| `SMOLDOCLING_DEBUG_PROMPT` | value | unset | value | debug | `src/smoldocling_ocr.cpp` | Prompt-contract debug: dump the prefill ids for parity checks against the reference processor (one id per line). |
| `SMOLDOCLING_FORCE_CPU` | value | unset | int | tuning / config | `src/smoldocling_ocr.cpp` |  |
| `SMOLDOCLING_LEGACY_PREPROC` | value | unset | value | tuning / config | `src/smoldocling_ocr.cpp` | 512x512 input (regression-bisection gate; known to hallucinate duplicated regions on non-square pages). |
| `SMT_OCR_DEBUG` | on/value | off | value | debug | `src/smt_ocr.cpp` |  |
| `SMT_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/smt_ocr.cpp` |  |
| `SMT_OCR_FULL_DECODE` | on | off | bool | default-off experimental / opt-in | `src/smt_ocr.cpp` | SMT_OCR_HOSTKV=1 = the host-round-trip KV path; SMT_OCR_FULL_DECODE=1 = the no-cache teacher-forced full graph. The three are output-identical. |
| `SMT_OCR_HOSTKV` | on | off | bool | default-off experimental / opt-in | `src/smt_ocr.cpp` |  |
| `SMT_OCR_NO_DECODE_CACHE` | on | off | bool | default-off experimental / opt-in | `src/smt_ocr.cpp` | is the win for the ~thousands-of-steps full-page decode. Gated so the sched path stays available (SMT_OCR_NO_DECODE_CACHE=1); output-identical. |
| `SMT_OCR_REDUCE_RATIO` | value | unset | float | tuning / config | `src/smt_ocr.cpp` | test-only scale override |
| `SURYA_DET_DUMP` | on | off | bool | debug | `src/surya_det.cpp` |  |
| `SURYA_DET_FORCE_CPU` | value | unset | string | tuning / config | `src/surya_det.cpp` | Silicon, CUDA elsewhere) so the stage 0-2 / decode-head graph runs on GPU; set SURYA_DET_FORCE_CPU=1 to pin to CPU for parity debugging. |
| `SURYA_DET_SCALAR` | on | off | bool | default-off experimental / opt-in | `src/surya_det.cpp` | Use graph-accelerated path unless SURYA_DET_SCALAR is set |
| `SWINIR_DUMP_DIR` | value | unset | path | debug | `src/swinir_sr.cpp` |  |
| `SWINIR_SR_FORCE_CPU` | value | unset | int | tuning / config | `src/swinir_sr.cpp` |  |
| `SWINIR_SR_SCALAR` | value | unset | int | tuning / config | `src/swinir_sr.cpp` | conv_after_body, upsample) with ggml_conv_2d on a dedicated CPU sched. The Swin window attention / norms / pixel-shuffle stay scalar. |
| `TBSRN_SR_FORCE_CPU` | value | unset | int | tuning / config | `src/tbsrn_sr.cpp` |  |
| `TBSRN_SR_SCALAR` | value | unset | int | tuning / config | `src/tbsrn_sr.cpp` | fused data where present). Kernels are custom-writer PyTorch [OC,IC,KH,KW]; reverse the 4 ne axes to ggml [KW,KH,IC,OC] over the same bytes. |
| `TEMP` | value | unset | value | tuning / config | `src/core/temp_file.h` |  |
| `TESSERACT_DIFF_DEBUG` | on | off | bool | debug | `src/tesseract_lstm.cpp` |  |
| `TEXT_SR_FORCE_CPU` | value | unset | int | tuning / config | `src/text_sr.cpp` | to remove (measured there at ~85% of a one-shot invocation). TEXT_SR_GPU_LOAD=1 restores the old behaviour for bisection. |
| `TEXT_SR_GPU_LOAD` | on | off | bool | default-off experimental / opt-in | `src/text_sr.cpp` |  |
| `TMPDIR` | value | unset | value | tuning / config | `src/core/temp_file.h` |  |
| `TPS_LOCNET_FORCE_CPU` | value | unset | int | tuning / config | `src/tps_locnet.cpp` | nothing -- the waste CRISPEMBED_TESSERACT_GPU_LOAD was introduced to remove. TPS_LOCNET_GPU_LOAD=1 restores the old behaviour for bisection. |
| `TPS_LOCNET_GPU_LOAD` | on | off | bool | default-off experimental / opt-in | `src/tps_locnet.cpp` |  |
| `TRANSCODA_OCR_DUMP_PX` | value | unset | value | debug | `src/transcoda_ocr.cpp` |  |
| `TRANSCODA_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/transcoda_ocr.cpp` |  |
| `TRANSCODA_OCR_FULL_DECODE` | on | off | bool | default-off experimental / opt-in | `src/transcoda_ocr.cpp` |  |
| `TRANSCODA_OCR_HOST_KV` | on | off | bool | default-off experimental / opt-in | `src/transcoda_ocr.cpp` |  |
| `TROMR_OCR_FORCE_CPU` | value | unset | int | tuning / config | `src/tromr_ocr.cpp` |  |
| `UOCR2_DUMP_VISION` | value | unset | value | debug | `src/unlimited_ocr.cpp` | view_separator; untiled: global + view_separator. Diagnostics only; the decode is unchanged. Tile pixels (if tiled) go to <path>.tiles.f32. |
| `UOCR2_FAST_CHUNK` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_CLIP_DBG` | on | off | bool | debug | `src/unlimited_ocr.cpp` | Pre-LN output check |
| `UOCR_DBG` | on | off | bool | debug | `src/unlimited_ocr.cpp` |  |
| `UOCR_DECODE_REBUILD` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_DECODE_TIMING` | on | off | bool | debug | `src/unlimited_ocr.cpp` |  |
| `UOCR_DUMP_PIXELS` | value | unset | value | debug | `src/unlimited_ocr.cpp` | (float32), so a reference can be run on exactly these pixels - separates preprocessing rounding from model differences. |
| `UOCR_FA_F32` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | debug |
| `UOCR_INJECT_REF` | on | off | bool | debug | `src/unlimited_ocr.cpp` | UOCR_INJECT_REF: replace C++ SAM output with Python reference SAM output to isolate CLIP encoder bugs from SAM quantization noise. |
| `UOCR_INJECT_VIS` | on | off | bool | debug | `src/unlimited_ocr.cpp` | (perfect, HF-equal vision input) AND avoids the vision Metal buffers so f16/q8_0 fit in memory. Used to prove decode bugs are not "quantization". |
| `UOCR_INSTR` | value | unset | value | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_LMHEAD_CPU` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_MAX_NEW` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_MMAP` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_MOE_CPU` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | otherwise deref a raw device pointer (Metal segfault); ggml_backend_tensor_get then reads via view_src->buffer. |
| `UOCR_NGRAM_WINDOW` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_NO_KV` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_NO_REPEAT_NGRAM` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_OPT_FUSED_DECODE` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_OPT_GRAPH_LN` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | UOCR_OPT_GRAPH_LN: do LN inside the ggml graph (no CPU LN + separate residual) |
| `UOCR_OPT_PD_F32` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | Use F32 for PD KV cache when UOCR_OPT_PD_F32=1 to avoid F16 precision issues |
| `UOCR_OPT_SAM_RES` | value | unset | int | tuning / config | `src/unlimited_ocr.cpp` | UOCR_OPT_SAM_RES=N: process SAM at reduced resolution for speedup Must be a multiple of patch_size (16). Examples: 512 (~5x), 768 (~2.5x) |
| `UOCR_PD` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_PD_DBG` | on | off | bool | debug | `src/unlimited_ocr.cpp` | Debug: mark per-layer output for comparison |
| `UOCR_PD_REPLAY` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` | winning mode (and its decode still diverges, see the gate comment above); it remains an opt-in research path. |
| `UOCR_REF` | value | unset | path | tuning / config | `src/unlimited_ocr.cpp` |  |
| `UOCR_SAM_CONV_CPU` | on | off | bool | default-off experimental / opt-in | `src/unlimited_ocr.cpp` |  |
| `UOCR_TEXT_TEST` | value | unset | value | tuning / config | `src/unlimited_ocr.cpp` | Isolation test: UOCR_TEXT_TEST runs the LLM decoder as a pure language model |
| `USERPROFILE` | value | unset | value | debug | `examples/cli/model_mgr.cpp` |  |
| `VIT_DEBUG` | on | off | bool | debug | `src/vit_embed.cpp` | Debug: mark intermediates for per-layer comparison (VIT_DEBUG=1) |
| `VIT_EMBED_FORCE_CPU` | value | unset | int | tuning / config | `src/vit_embed.cpp` | Load weights — prefer GPU backend when available |

<!-- END GENERATED: tools/list_env_vars.py -->

`CRISPEMBED_LIGHTON4B_VISION_BF16` (default off) selects the isolated4B vision
arithmetic candidate: BF16 input cast, linear/norm/nonlinearity/residual outputs,
source BF16 interpolation and attention tensor boundaries. Float weights remain
unchanged; the legacy F32 vision engine remains available. This is an unqualified
precision-semantics candidate, not a decoded-quality or speed promotion.
