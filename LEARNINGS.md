# CrispEmbed — Technical Learnings

## A cached CMake `option()` silently disabled Metal — and the binary still reported `GGML_METAL:BOOL=ON` (2026-08-05, T14)

Every measurement in the first half of the T14 round was CPU while being
labelled Metal. The tell was not in the build system at all: `sam`, a stage
neither A/B arm touches, varied 17.7 s vs 38.1 s between runs, and PLAN's warm
profile said the whole page should take ~12 s.

**The mechanism is a stale cache entry, and that is the durable part.** The
build dir was first configured *without* Metal:

```
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release          # GGML_METAL=OFF
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
```

ggml declares `option(GGML_METAL_EMBED_LIBRARY "..." ${GGML_METAL})`. On the
FIRST configure `GGML_METAL` was OFF, so `GGML_METAL_EMBED_LIBRARY` was cached
`OFF`. **`option()` never revisits a value that is already in the cache**, so
the second configure turned Metal on but left the library un-embedded. The
cache then reads `GGML_METAL:BOOL=ON` — which is what a reviewer checks — while
`GGML_METAL_EMBED_LIBRARY:BOOL=OFF` two lines away is what actually decides it.

Un-embedded, ggml writes `default.metallib` into `${CMAKE_RUNTIME_OUTPUT_
DIRECTORY}` = `build/bin/`, but `ggml_metal_library_init` looks for it beside
`argv[0]` (`ggml-metal-device.m`: `bin_dir = argv[0] stringByDeletingLast
PathComponent`), and CrispEmbed links `crispembed` into `build/`. It then falls
back to compiling `ggml-metal.metal` from source — which ggml's own CMake
`rm -f`s after building the metallib — so init fails and the engine silently
runs on CPU:

```
ggml_metal_library_init: default.metallib not found, loading from source
ggml_metal_library_init: error: ... "ggml-metal.metal" couldn't be opened ...
ggml_metal_device_init: error: failed to create library
```

Fixed for the measurement with `ln -sf bin/default.metallib
build/default.metallib` (ggml resolves the symlink). The real fixes are
`-DGGML_METAL_EMBED_LIBRARY=ON`, a clean build dir, or having CrispEmbed copy
the metallib beside its own executable target.

**Checks that do NOT catch this**, all of which were performed: `GGML_METAL:
BOOL=ON` in `CMakeCache.txt`; `default.metallib` existing in the build tree;
the binary linking Metal; a correct decoded transcript. The only reliable
positives are `ggml_metal_library_init: found '<path>'` +
`ggml_metal_init: allocating` on stderr, or `GGML_SCHED_DEBUG=2` showing splits
on a device that is not `CPU`. The recorded rule "check `CMakeCache` + MTL0
stderr before calling a measurement Metal" needs the second half enforced: the
cache line alone is not evidence. **A stage that neither arm of an A/B touches
is the cheapest tripwire for a mislabelled backend — if `sam` moves 2x between
runs of identical code, stop and find out why before reading any A/B column.**


## A low q4_k cosine is NOT a bug — prove it with a precision control (2026-07-16)

When a shipped q4_k GGUF scores a low cosine vs the HF model (nomic-embed-text-v1.5
hit **0.9515**), that number alone can NEVER distinguish "quant floor" from "our
encoder graph is wrong" — both look identical at the output. The ONLY thing that
separates them is re-running the **same code path at higher precision**: if the
f16/f32 GGUF of the same model matches the original Python model to ~1.0 at EVERY
per-stage layer, the graph is exact and the whole q4_k gap is quantization.

Measured this way, all three encoder paths are proven exact:
`bge-small f32` = 1.000000/stage, `nomic-v1.5 f16` = 1.000000/stage,
`nomic-v2-moe f16` ≥ 0.9998/stage — so nomic-v1.5's 0.9515 is a real *quality*
fact (that model's last block is unusually quant-sensitive; prefer f16/q8), NOT a
port bug. Automated as `tests/prove_quant_control.py` (`control_file` per matrix
entry). **Do this before ever calling a low cosine a bug.** Corollary: cross-
comparing two of OUR OWN conversions (q4_k vs cstr-iq4_xs) is not ground truth —
both can agree and both be wrong; only the original Python model is.

## "Loads + emits" proves nothing; a loud failure beats silent garbage (2026-07-16)

Two related discipline lessons from the community-GGUF work:

1. **A model that loads and returns a same-shape vector can still be garbage.**
   The community `modern-bert` loader fix made gte-modernbert-base load
   (22L/768d, right dims) and emit a 768-dim embedding — yet cos(related)=0.068 <
   cos(unrelated)=0.157 (negative margin). "It loads" and "the dim is right" are
   necessary, never sufficient (HARD RULE #3). The per-stage HF diff then showed
   the STRUCTURAL GATE itself failing (`emb_ln_out` cos 0.58 = divergence *before*
   block 0), which localized the bug to tokenization, not the graph — the
   opposite of where intuition (GeGLU/attention) pointed.

2. **Do not "fix" a loud failure into a silent-wrong success.** The modern-bert
   loader aliases turned a LOUD `missing required tensor` into a SILENT garbage
   embedding (loads, exit 0). That is strictly worse and was NOT shipped — the
   change was reverted, only the diagnosis kept. Same principle drives
   `CRISPEMBED_STRICT_HPARAMS`: a missing required hparam that silently defaults to
   384-dim/6-layer emits a plausible-but-wrong embedding with exit 0. A model that
   won't load tells the user something is wrong; one that loads-and-lies does not.

## Community GGUFs ≠ our own conversions — the ecosystem-compat gap (2026-07-16, #33)

We ship registry entries pointing at our `cstr/*` GGUFs and test THOSE, so a model
"works" while the *community* GGUF of the same model — a llama.cpp/Ollama export,
which is what users reach for first — fails to load. That is exactly what issue
#33 was (nomic-embed-text-v2-moe). The gaps are systematic:

- **Metadata keys.** Our converter writes `bert.*`; llama.cpp writes
  `<general.architecture>.*` (e.g. `nomic-bert-moe.embedding_length`). Fix once,
  generally: read `general.architecture` and derive the prefix (`core_hparams`,
  A2) — every future arch resolves with no new code. Appending one alias per model
  (what #33's upstream PR did — it added only the 2 `expert_count` keys and would
  still have loaded at 384-dim/6-layer) never finishes.
- **Tensor names.** Fused `attn_qkv`, stacked `ffn_up_exps`/`ffn_down_exps`,
  `attn_norm`/`ffn_norm`/`output_norm` (ModernBERT) — llama.cpp names differ from
  ours. `get_any({...})` alias lists.
- **Tokenizer selection (the deep one).** Our converter writes a numeric
  `tokenizer.ggml.type`; community GGUFs write the standard STRING
  `tokenizer.ggml.model` (`gpt2`/`bert`/`t5`) with merges in the
  `tokenizer.ggml.merges` KV ARRAY, not a tensor. crispembed reading only
  `tokenizer.ggml.type` (+ an `n>100000→SPM` vocab-size heuristic) silently picks
  the WRONG tokenizer for a modern-bert `gpt2` GGUF → WordPiece not BPE → garbage
  from token 0. (The heuristic only made e5/granite work by luck.)

Guard it: `tests/community_gguf_matrix.json` loads THIRD-PARTY GGUFs and gates on
load + shape + a garbage guard + HF parity. Adding 3 entries immediately surfaced
a 3rd arch string (`nomic-bert`), a latent RoPE default bug, and the modern-bert
tokenizer bug — the coverage IS the bug-finder.

## Fixing community modern-bert: the tokenizer, not the graph — and the gotchas that make each step lie (2026-07-16, `feat/modernbert-community-gguf`)

The RESOLUTION of the modern-bert entry above (`gte-modernbert-base`, arch
`modern-bert`). crispembed already had the ModernBERT compute graph; the fix was
5 tokenizer/loader steps, each gated on the per-stage HF diff. Durable learnings:

- **Make the tokenizer.ggml.model STRING authoritative, but only when the numeric
  type is absent — and keep the legacy `n>100000→SPM` heuristic as the last
  resort.** Changing the *dispatch* condition (not just the derivation) would flip
  an explicit-`type=0`-huge-vocab GGUF from SPM to WordPiece. Derive the type from
  the model string when `tokenizer.ggml.type` is missing, then leave the existing
  dispatch conditions byte-identical. Net effect: only the previously-broken gpt2
  GGUFs change behaviour; every bert model is unchanged (verified — full 5-model
  matrix still PASS).
- **Read the `tokenizer.ggml.merges` KV array BEFORE `gguf_free`, into a local
  that survives to the post-weight-load merges site.** Community gpt2 GGUFs store
  merges in that KV string array; our converter stores a `tokenizer.merges`
  TENSOR. Prefer the tensor, fall back to the KV array. (Use-after-free landmine:
  the vocab/merges must be pulled while the `gguf_context` is live.)
- **The structural gate (`emb_ln_out` cos) is the ONLY thing that proves
  tokenization — not the final embedding.** Wrong tokens → emb_ln_out cos 0.58;
  right tokens → 0.9999. A final-embedding cosine can look "okay-ish" under a
  token shift; the pre-block-0 gate cannot. Also assert the token IDS match HF
  (`[50281,25521,1533,50282]` for "hello world") — a shifted token mimics numeric
  drift.
- **The GPT-2 ByteLevel regex pre-tokenizer coincidentally equals the simple
  whitespace-split for `"hello world"` — do not let that fool you into skipping
  it.** They diverge on punctuation/digits/contractions/multi-space. The HF
  `ByteLevel(use_regex=true)` regex is
  `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
  the subtle part is whitespace: a run of ≥2 spaces before a word emits all-but-
  the-last as one token and the LAST space rides into the next word's ` ?`
  (a single non-0x20 whitespace like `\t` becomes its own token via the final
  `\s+`). Gate the regex path on arch so Qwen3's whitespace-split path is
  untouched.
- **ModernBERT's GeGLU differs from the validated GTE-v1.5 path ONLY in the gelu
  approximation.** ggml `ggml_geglu` (non-swapped) = `gelu_tanh(first_half) *
  second_half`; ModernBERT's `hidden_activation="gelu"` is exact-erf, so
  `ggml_geglu_erf`. Same layout, same split order — the fused `Wi` is stored
  `[input; gate]` verbatim, so non-swapped is right. Gate `geglu_erf` on
  `arch=="modern-bert"` so GTE-v1.5's tanh stays.
- **The RoPE theta naming is INVERTED vs crispembed's fields.** llama.cpp's
  `<arch>.rope.freq_base` is the GLOBAL theta (160000) and `rope.freq_base_swa`
  is the LOCAL/sliding theta (10000). The generic `ak("rope.freq_base")` read
  loads the global into `rope_theta` — override it in the `modern-bert` block.
- **The f16 control is what turns "q8_0 = 0.9996" from *maybe a bug* into *proven
  quant*.** Same code path at f16 gave cos=1.000000 at EVERY stage (and 0.999999
  final) vs the HF fp32 reference — so the tokenizer AND the graph are exact and
  the entire q8_0 gap is quantization. The control's f16 GGUF lived in a DIFFERENT
  repo than the shipped q8_0 (eranmazur ships only q8_0; the f16 is in
  `cstr/*-GGUF`), so `prove_quant_control.py` gained a `control_repo` override.

## Community `gemma-embedding`: SentencePiece **BPE** needs merge-from-scores, not Viterbi — and the crash/tokenizer/Dense split (2026-07-17, `feat/embeddinggemma-community-gguf`)

Same class as the modern-bert fix above: crispembed had the whole Gemma3 compute
graph; the official llama.cpp EmbeddingGemma export
(`ggml-org/embeddinggemma-300m-*-GGUF`, arch `gemma-embedding`) still failed, and
the handover's diagnosis (missing Dense, or a gemma-norm `1+w` bug) was wrong.

- **Loud crash → route to the decoder, but never ship routing alone.** The
  hyphenated arch missed the `is_dec` allow-list and fell into the generic MHA
  encoder graph, whose QKV reshape assumes `n_heads==n_kv_heads`; gemma-embedding
  is GQA (3 heads / 1 kv, `head_dim=256`), so K/V hold `1*256` per token not
  `3*256` → the reshape overruns → `GGML_ASSERT`. Routing it to `decoder_embed.cpp`
  makes it LOAD, but with the broken tokenizer below it emits a silently-weak
  embedding (margin 0.039) — the modern-bert anti-pattern. A loud crash is safer
  than that; fix the tokenizer in the same change.
- **The real bug: a SentencePiece export loaded as char-level BPE.** The GGUF has
  `tokenizer.ggml.model=llama`, a `scores` array, and **no** `merges`. The decoder
  loader hardcoded `use_bpe=true`; a BPE with 0 merges can't merge, so it falls to
  single characters ("hello world" → BOS + 11 char tokens + pad). Detect this
  (`merges.empty() && scores present` → SentencePiece) and route to
  `SentencePieceTokenizer`.
- **Gemma SentencePiece is BPE, and its `scores` are merge RANKS — Viterbi is the
  WRONG algorithm.** This is the exact complement of the XLM-R/Unigram rule below.
  crispembed's `SentencePieceTokenizer` was Viterbi-only (max-sum over scores,
  correct for Unigram log-probs). Gemma's scores are large-negative ranks
  (`▁world`=-1408 vs `▁w`+`or`+`ld` = -21-10-177 = -208), so max-sum picks the
  3-way split over the single vocab token `▁world` → over-segmentation that still
  looks plausible. Fix: add an `spm_bpe` mode implementing llama.cpp's SPM bigram
  greedy-merge (priority queue, merge the adjacent pair whose concatenation has the
  highest vocab score), selected when `tokenizer.ggml.model` is `llama`/`gemma`.
  Keep Viterbi the default so XLM-R is untouched. Per-stage lie to avoid: judge on
  the **exact token IDs** (`[2,23391,1902,1]` for "hello world"), not the final
  cosine — a wrong segmentation that stays in-vocab gives a plausible-but-wrong
  vector.
- **`add_space_prefix` is a real per-model flag.** The Viterbi path hardcoded the
  XLM-R dummy leading `▁`; Gemma sets `tokenizer.ggml.add_space_prefix=false`
  (its first word matches the bare vocab token `hello`=23391, not `▁hello`). Read
  the flag; wrong-prefix alone re-splits the first word.
- **The GGUF has NO Dense head — and without it the output is orthogonal to real
  EmbeddingGemma.** llama.cpp applies the SentenceTransformers Dense/Matryoshka
  head from an external `--sentence-transformers-dense-modules` file, so the export
  omits it. Right tokenizer + right backbone still gives cos **−0.02** vs HF full
  output (the 768→3072→768 Dense remaps the space entirely) — it discriminates
  in isolation (margin 0.39) but is not the published embedding. `decoder_embed.cpp`
  already applies `dense.N.weight` post-pool; `models/add-st-dense-to-gguf.py` bakes
  `2_Dense`/`3_Dense` (linear.weight `[out,in]`, F32, no transpose) into the GGUF →
  cos vs the full HF `SentenceTransformer` = **0.985**. Disentangle A(Dense) vs
  B(norm) with a **pre-Dense backbone control**: cos(crispembed, HF mean-pool
  BEFORE Dense) = 0.9835 proves the backbone/norms are right (no bug B); the 0.985
  ceiling is QAT-vs-vanilla checkpoint drift + the known EmbeddingGemma Dense-
  bottleneck discrepancy ([[embeddinggemma-parity-state]], and the "EmbeddingGemma:
  a non-orthogonal Dense bottleneck" section below) + q8_0, not a defect.
- **GGUF surgery must skip `GGUFReader`'s synthetic pseudo-keys.** `.fields`
  exposes `GGUF.version`/`GGUF.tensor_count`/`GGUF.kv_count` alongside real KV, but
  those come from the file HEADER — copying them writes literal `"GGUF.version"`
  metadata (readers then warn "Duplicate key", kv_count inflates 35→38). Skip
  `key.startswith("GGUF.")`. Also verified the *reassuring* half by a raw-header
  round-trip: all 35 real KV (tokens/scores/token_type arrays) copy with zero
  type/value drift, so `GGUFWriter`'s array-element-type inference is faithful.
  `models/gguf_merge_core.py` can NOT be used for this copy — its reader drops the
  array element type.
- **A "not installed" import can be the `USE_TF=0` gotcha.** `import
  sentence_transformers` raised `ImportError: cannot import name 'TFPreTrainedModel'`
  and I concluded it was absent, shipping the matrix entry without its HF-parity
  gate. It IS installed — the import only fails via the TensorFlow integration
  path; `USE_TF=0` (which `hf_parity_community.py` sets) fixes it. Test the real
  cause before declaring a capability missing ([[dont-assume-env-capabilities]]).

## Position offset can NOT be inferred from the tokenizer — a community XLM-R "bert" GGUF that omits `position_offset` is under-specified (2026-07-16, e5 vs granite)

Extending the community matrix to two XLM-RoBERTa-family SPM embedders exported as
arch `bert` split them:
- `granite-embedding-107m-multilingual` (community q4_k) parity-matches HF at the
  structural gate (cos 0.9999) — it uses absolute position **offset 0**.
- `multilingual-e5-small` (community `rodion-m` fp32) does NOT: gate cos **0.467**
  with matching norms (16.4 vs 16.5) = a pure position-embedding SHIFT. HF
  `intfloat/multilingual-e5-small` is XLM-RoBERTa (padding_idx=1 → position
  **offset 2**), but the community `bert`-arch GGUF drops `bert.position_offset`,
  so crispembed defaults to 0.

The trap: **both models share the identical RoBERTa SentencePiece tokenizer**
(bos=0/eos=2/pad=1, tokens `[0,33600,31,8999,2]` for "hello world"), yet need
DIFFERENT position offsets. So there is no tokenizer-side signal to key an
"offset 2" heuristic off — it would break granite. Position-embedding convention
lives in the model config (`padding_idx`), which a `bert`-arch GGUF export can
silently omit. Do NOT add a speculative auto-detect; it's an under-specified
export (fix belongs upstream / use a GGUF that declares the offset — our `cstr/*`
e5 does). The matrix is what surfaced it: granite ADDED + validated, e5
documented as a found gap, no false "passing" entry shipped. (A garbage-guard-only
check would have *passed* e5 on a self-consistent-but-wrong vector — only the HF
per-stage structural gate caught the shift.)

## Hand-rolled JSON drifts into N diverged copies — centralize in `core/` (2026-07-16)

The HTTP server and the CLI each had their own `json_escape`, and they had already
DIVERGED (server: `"`,`\`,`\n`; CLI added `\r`,`\t`) — both echoing OCR/KIE/NER
text, both missing `\b`/`\f`/`\u00XX`, so both emitted invalid JSON on a control
char (a tab in OCR'd table text breaks every strict client). Same class as the
`pcs.cpp` two-copies-drift lesson. Unified into `src/core/json.h` (`core_json`,
matching `core_util`/`core_gguf`). Two durable sub-lessons:

- **The escaper must be the exact inverse of the decoder**, and the way to prove
  it is a round-trip PROPERTY test — `decode(escape(x)) == x` over all 256 byte
  values — not example cases. That property test would have caught the missing
  `\t`/`\r` immediately; the example-based tests didn't.
- **Locate JSON keys structurally (brace-depth 1), never `body.find("\"key\"")`.**
  A string VALUE equal to the key name (legal unescaped, e.g. `["input"]` in a
  labels array) makes `find` match the decoy and skip to the next colon — returning
  another key's values, or values for a key that isn't present. Reachable via
  user-controlled `/ner` labels.


## Converter-emitted stacked MoE experts halve the resident expert memory — and measure the win by PEAK FOOTPRINT, not max RSS (2026-07-13, deepseek-ocr2 #4)

The MoE decoders (deepseek_ocr2, unlimited_ocr) shipped per-expert 2D weight
tensors (`l.blk.{i}.exp.{e}.ffn_{gate,up,down}.weight`) and `stack_moe_experts()`
rebuilt them at load into 3D `[in,out,n_exp]` tensors for `ggml_mul_mat_id` — so
BOTH the per-expert copies (in `model_buf`) AND the stacked copy (in `moe_buf`)
sat resident: ~1.3 GB duplicated on a 2 GB q4_k model.

**Fix: emit the stacked tensors from the CONVERTER, load them directly.** The key
layout identity: `np.stack([expert_e for e], axis=0)` → numpy `[n_exp,out,in]`
which gguf reverses to ggml `ne=[in,out,n_exp]` — **byte-identical** to what
`stack_moe_experts` builds (expert `e` at slice offset `e*nb[2]`). So the loader
just points `gate_exps` at the loaded tensor; no copy, no stacking pass. Verified
the identity locally on a synthetic gguf before spending Kaggle compute, and
byte-validated the real slices vs the source safetensors on Kaggle.

Gotchas that mattered:
- **`down_proj` has the opposite shape.** gate/up are hidden→moe_inter
  (`ne=[1280,896,64]`); down is moe_inter→hidden (`ne=[896,1280,64]`). A validator
  (or any per-tensor reshape) that hardcodes one shape breaks on down. And down's
  `ne[0]=896` is not 256-divisible, so Q4_K falls back to Q4_0 — but the OLD
  per-expert down tensors had the same `ne[0]=896`, so this is byte-for-byte the
  same quantization, not a regression.
- **DS_MOE_CPU fallback needs per-expert views, and the view's `->buffer` must be
  set.** `ggml_backend_tensor_get` reads a view via `view_src->buffer`, but
  `to_f32`'s fast path gates on `t->buffer` (null on a fresh view) and would then
  deref the raw device pointer → Metal segfault. Set `view->buffer = parent->buffer`.
- **The quantizer already handles 3D** (`n_dims>=5` copies as-is; 3D falls through
  to the standard per-row path, `nrows = nelements/ne[0]`), so no quantizer change.
- **Keep the loader backward-compatible** (probe `ffn_gate_exps`; else legacy
  per-expert) so old GGUFs still load — HARD-RULE-4 "never delete the working path."

**Metric lesson: judge process memory by PEAK FOOTPRINT (`phys_footprint`), not
max RSS.** On the M1 A/B, `maximum resident set size` was *noisy and even inverted*
(old 1.83 GB vs new 4.22 GB) because RSS counts mmap'd GGUF pages resident in the
page cache, which swings with cache state. `peak memory footprint` — the process's
own committed anonymous memory — was stable and showed the real win: **5.27 → 3.97
GB (−1.30 GB, −25%)**, matching the removed duplication exactly. Decoded output was
identical ("The quick brown fox…" cer 0.0) on all three loader paths. See the
[[deepseek-stacked-experts-memory]] memory. On `main`, HF `-stacked` files.

**Generalizes to any per-expert-load + runtime-stack engine.** Applied verbatim to
`unlimited_ocr` (same DeepSeek-V2 MoE, `baidu/Unlimited-OCR`): output byte-identical
on all 3 loader paths, peak footprint 4.32 → 3.11 GB (−1.21 GB). `crispembed.cpp`'s
BERT/NLLB MoE embedders already load pre-stacked 3D experts (`expert_fc1_w
[H,inter,N_exp]`) from their converter — no duplication, nothing to do. Those three
are the *only* `ggml_mul_mat_id` paths in the tree. One Kaggle gotcha the port
surfaced: the numpy expert **accumulate-then-stack** (holding ~10 GB of f32 experts)
**thrashes for hours under multithreaded OpenBLAS** — the v1 reconvert hung ~3h with
no progress. Prefix converters with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1` (the dev-guide HARD rule) — the deepseek run
got lucky without it, unlimited did not.

## A parity stage downstream of a topk/argsort selection craters by query PERMUTATION under tiny backend FP deltas — not a compute bug (2026-07-13, layout-heron dec_0_cross_out)

`test-layout-diff`'s `dec_0_cross_out` stage looked like a flaky GPU bug:
cos_min **−0.08 on Metal**, **0.977 on CPU**, and "non-deterministic" across
Kaggle P100 runs (0.977 then −0.034 on the *same* box). Every instinct said
"Metal/CUDA numerical divergence in the deformable cross-attention." All wrong.

**The cross_out values are correct on every backend — they're just in a
different order.** RT-DETRv2 selects its 300 decoder queries with a
`std::partial_sort` over ~8400 **near-tie** encoder proposals
(`layout_detect.cpp` ~1318). A minuscule backend FP delta in `enc_output`
(Metal/CUDA vs the CPU/Python reference — max_abs 0.02, **cos 0.99999**, passes
its own 0.99 gate) reshuffles the near-tie ranks. So "query *i*" in our output
is a *different physical proposal* than "query *i*" in the reference. The
per-query, index-aligned cosine the harness computed then craters purely from
the reordering. The final detections are unaffected — they go through
score-sort + NMS, which is order-invariant.

**What nailed it (generalizable diagnostic):**
1. **Dump the intermediate that FEEDS the stage.** The *initial* decoder queries
   already showed per-query cos mean 0.78 / 111-of-300 below 0.9 — matching
   cross_out's 0.79 mean exactly. That located the divergence *upstream* of the
   CPU-side deformable sampling everyone suspected.
2. **The bijection signature.** Best-cosine matching each reference query to our
   output recovered **cos_mean 0.999 with 299/300 unique targets** — a clean
   bijection is the fingerprint of "right values, wrong order." A real scramble
   has no such matching (all best-matches collapse toward 0).
3. **Backend-independent of the impl choice.** Both manual `layout_attn` and
   `flash_attn_ext` gave the same ~0.79 — ruling out the attention kernel and
   pointing at a *selection*, not a *computation*, difference.

**Fix pattern (durable):** any parity stage downstream of a topk/argsort/greedy
*selection* must be compared **permutation-tolerantly** — best-cosine match each
reference vector against the full candidate set (`perm_tolerant_cos` in
`tests/test_layout_diff.cpp`, gate 0.85). This still catches a genuine
regression: simulated scrambles (feature-shuffle / sign-flip / spatial-roll) all
collapse to ≤0.08. And the encoder-scramble class it nominally guarded is
*already* covered strictly (0.99) by the upstream `s3..enc_output` stages, so the
looser downstream gate loses no coverage. Same "the metric was wrong, not the
model" class as the "bidirlm-vision parity: gate on per-token MEAN cosine" entry
below (worst-row min misleads under a legitimate outlier). Fixed on `main`
d7f0480.

## `ggml_concat` silently corrupts q4_k weights — QKV fusion needs load-time byte-stacking (2026-07-13, got_ocr QKV probe)

Tried to fuse got_ocr's LLM-decoder Q/K/V into one matmul via a graph-time
`ggml_concat(q_w, k_w, v_w, dim=1)` on the q4_k projection weights. It **compiles
and runs (no abort) but produces garbage** — the decode ran away to 1023 tokens
and "recognition failed". `ggml_concat` has no k-quant path; it mishandles the
super-block layout, so concatenating q4_k tensors yields wrong bytes. Two takeaways:

- **Never `ggml_concat` (or any elementwise op) a k-quant weight** and feed it to
  `mul_mat`. A correct QKV fusion must **byte-stack the row blocks at LOAD time**
  (q4_k rows over ne[1]=out are whole blocks, so stacking output rows is a valid
  q4_k tensor) into a persistent tensor — not rebuild it in the graph each step
  (which was also **3× slower**: 42.8 vs 12.9 ms/step from re-concatenating).
- **It isn't worth it anyway** on these decoders: a T=1 decode step is
  memory-bound `mul_mv`, so 3 separate q4_k matmuls move the same bytes as one
  fused q4_k matmul — fusion only saves ~2 kernel launches/layer (~4% of the
  ~11% host slice on a compute-bound decode). Metal also auto-fuses the norm/GLU
  elementwise chains already. `GOT_OCR_QKV_FUSE` probe reverted; see HISTORY.
  (got_ocr's *vision* tower ships a fused `attn_qkv` from the converter — the
  right place to fuse is the GGUF, not the runtime graph.)

## Fusing per-conv graphs helps only OVERHEAD-bound models; measure first (2026-07-13, SR ports)

CrispEmbed's SR/restoration engines dispatched a **fresh ggml graph per conv**
(init → alloc → compute → read-back, scalar glue between). Fusing the forward
into one graph is a real win **only when graph-setup overhead dominates**:

- **SAFMN** (228K params, tiny convs) is *overhead-bound* → full fusion gave
  **2.2× (6.1s→2.8s) + cos 1.0 vs 0.994**. Metal LOSES here (per-dispatch +
  host↔device copy > the tiny compute) → default CPU.
- **NAFNet / InstructIR** (32–256-ch convs) are *compute-bound* → per-block
  fusion is correct (cos ≥ 0.999998, byte-identical output) but **perf-neutral**
  (the conv math, not setup, is the cost). NAFNet defaults to Metal for a modest
  ~15%; InstructIR stays CPU (GPU conv_2d hits a Metal f32×f16 mul_mv pipeline
  issue on this arch).

So **measure the baseline first** — `grep forward_expand / conv2d_ggml` to tell a
genuinely per-conv engine from an already-fused one (Restormer, scunet, swinir,
etc. were already single-graph, just mislabeled "CPU-scalar"), and time it to see
if it's overhead- or compute-bound before investing in a fused-graph port.

Two gotchas that cost real time on these ports:
1. **erf vs tanh GELU.** SAFMN's reference uses exact erf GELU; `ggml_gelu` is the
   **tanh approximation**, and using it alone dropped output cos 1.0 → 0.947. Use
   `ggml_gelu_erf` when the reference does exact GELU (SAFM/SR/many ViTs).
2. **Conv weight layout scramble.** GGUF stores conv weights as `[OC,IC,KH,KW]`
   bytes, but `ggml_conv_2d` wants an `[KW,KH,IC,OC]` kernel. A plain
   `ggml_reshape_4d` on the raw leaf claims the right ne but keeps the wrong byte
   order → a reshape ASSERT or a silently scrambled kernel (cos ~0.5). Copy the
   dequant bytes UNCHANGED into an explicit `[KW,KH,IC,OC]` tensor (nafnet_resident)
   or replicate the engine's own axis-order detection (instructir `ir_kernel`).
   Also: raw GGUF leaves live on `ctx->backend`; a CPU conv sched can't run ops on
   a foreign-buffer leaf, so park weights on `enc_backend` or include their backend
   in the sched. Env-gate every path (`<ENGINE>_LEGACY`/`_CPU`/`_METAL`).

## "Reads structure but not detail" across independent models = a preprocessing/input bug on MY side, not model quality (2026-07-12, SMT + TrOMR OMR)

Two published SOTA OMR models both scored ~30% vs ground truth through my harness
— each reading the *structure* (clefs, barlines) right but the *detail* (pitches,
key/time sig) wrong. That uniform cross-model failure was the tell: a **systematic
input bug on my side**, not two coincidentally-weak models. Both were "wrong
input," and fixing it took each from ~30% to ~96%:

- **SMT: a spurious invert.** I used the **SMT-plusplus** fork's
  `convert_img_to_tensor` (`RandomInvert(p=1.0)`), but `smt-grandstaff` is an
  **SMT-main** model whose `convert_img_to_tensor` is `Grayscale→ToTensor` with
  **NO invert**. Feeding white-on-black → 30%; black-on-white (correct) → **96.3%**.
  Also from SMT-main `prepare_data`: RGB (not cv2 BGR), `reduce_ratio=1.0`,
  `width=min(w,3056)`, `height=max(h,256)`.
- **TrOMR: fed the wrong file.** The repo's `N.png` are 4-channel *reference
  renderings*; `readimg`'s `255 - img[:,:,3]` on their opaque alpha yields an
  all-black image (`gray range 0 0`). The real inputs are `photoN.jpg`. On the
  photos, TrOMR reads clefs/keys/rhythms/pitches correctly.

The **accuracy metric** was ALSO wrong at first: comparing the model's dot-stripped
tokens (`8FL`) to raw GT with `·` (`8·F·L`). GT must be normalized exactly as the
model's `prepare_data`: `re.sub(r'(?<=\=)\d+','')`, spaces→`<s>`, strip `·`,
tabs→`<t>`, newlines→`<b>`.

The SMT port itself was correct the whole time (per-stage cos=1.0, C++==Python
100%, SMT-plusplus's unscaled forward confirmed vs SMT-main which gives 0% garbage
on this checkpoint). "C++==Python==30%" proved only the PORT — both shared my bad
preprocessing.

**Durable rules:** (1) when a published SOTA model scores implausibly low, suspect
your HARNESS, not the model; (2) a "reads-structure-not-detail" pattern — and
*especially* the same pattern across independent models — is an input/preprocessing
bug; (3) derive preprocessing from the model's OWN repo/commit (`smt-grandstaff`
↔ SMT-main, not the same-named SMT-plusplus fork), and run the model on its OWN
example images/expected outputs before trusting any number. See
[[validate-intermediates-and-outputs]].

## TrOMR engine port: three traps a "tested" converter + handover brief hid (2026-07-13, src/tromr_ocr.cpp)

Porting Polyphonic-TrOMR (ResNetV2 SAME-pad backbone + hybrid ViT encoder →
x-transformers 12-sublayer decoder with SIGLU attn-on-attn / GEGLU FF → 4 parallel
heads) reached full parity (every diff-harness stage cos=1.0, 100% teacher-forced
argmax agreement 66/66 & 85/85, byte-exact greedy decode vs the authors'
`examples/{1,2,3}.txt`, Metal==CPU). Three non-obvious traps, none caught by the
"tested" converter or the handover brief:

- **The handover brief's ViT scale was wrong.** It said `64^-0.5`; the correct
  value is **`32^-0.5`** because `head_dim = encoder_dim/heads = 256/8 = 32` (the
  qkv weight is `[768,256]` = 3×256, so the inner dim is 256, not 512). The decoder
  *is* `64^-0.5` (inner 512, 8 heads). `enc_context` cos only hits 1.0 with 32.
  Corollary: [[verify-handover-claims-independently]] — the brief also mislabeled a
  scale and I only caught it because the diff harness is the arbiter.
- **The Python `gguf` writer does not enforce `GGML_MAX_NAME` (64), the ggml C
  loader does.** The converter emitted 69-char names
  (`encoder.patch_embed.backbone.stages.0.blocks.0.downsample.conv.weight`) and
  every engine load aborted `tensor name … too long`. A converter can be "tested"
  (writes fine, round-trips in Python) yet produce a GGUF **no ggml engine can
  open** — because no engine existed yet to load it. Fix: shorten the prefix in the
  converter (`→ enc.bb`) and mirror it in `map_tensors`.
- **Quantizing flattened 4D conv weights breaks the in-engine reshape-to-4D.** The
  quantizer flattens `[kw,kh,ic,oc] → [ic*kh*kw, oc]` then quantizes; reshaping a
  q8_0 tensor back to 4D yields an `ne[0]` (e.g. 1 for a 1×1 conv) that is not a
  multiple of the 32-element block → `ggml_dup` abort. Fix per the SMT precedent:
  add the conv prefixes (`enc.bb`, `enc.proj`) to the `tools/quantize.cpp`
  keep-guard so they stay F32. q8_0 then decodes byte-exact (argmax 66/66); the
  backbone staying F32 costs compression (1.7x) but that is a correctness/size
  tradeoff, not a bug.

Also: the authors' `examples/N.txt` were sampled at **`temperature=0.2` (stochastic)**,
so neither my argmax nor the reference dumper's argmax is *expected* to match them
byte-for-byte on hard polyphonic passages — argmax faithfulness is proven by
per-position agreement under teacher forcing (100%), not by exact-match to a
stochastic sample. A single near-tie flip (F16 conv-cast, logits max_abs ~8e-3)
cascades the greedy path once the prefixes diverge — expected, not a regression.

## Flova engine port: the sibling-verbatim path, and a Donut patch-embed pad the handover glossed (2026-07-13, src/flova_ocr.cpp)

Flova/omr_transformer (Donut VED: DonutSwin encoder → 4-layer **pre-norm** mBART
decoder → LilyPond) is the only permissive handwritten-music OMR model. It reached
full parity on the **first build** — every diff-harness stage cos=1.000000
(enc_stage0..3, enc_output, dec_block0..3, logits), 40/40 teacher-forced argmax
agreement, and byte-exact greedy decode (`c'2 a''8 c''8 r4 c'1 e'8 …`) matching the
model card, including the native no-`transformers` Donut preprocessing path. What
made it fast and the two things worth recording:

- **The encoder was a config change, not a rewrite.** DonutSwin is the *same*
  windowed-attention Swin already in `mixtex_ocr.cpp` (scalar window_partition /
  cyclic_shift / window_mhsa + RPB, batched LN/linear on a small ggml CPU graph).
  `run_swin_encoder` already reads embed_dim / depths / heads / window_size from
  hparams generically, so copying it verbatim and pointing it at `flova.encoder.*`
  (embed 128, depths [2,2,14,2], heads [4,8,16,32], window 10) Just Worked. The
  `rpb_table [nh,361]` / `rpb_index [100,100]` tensors are read generically — no
  code change, only the tensors differ.
- **The one real encoder fix: patch-embed pads UP, mixtex truncated.** mixtex's
  input (400×500) is divisible by patch_size 4, so it used `pH = img_h/ps` (floor).
  Flova's 583×409 is not: DonutSwin zero-pads H,W up to a multiple of patch_size
  (583→584, 409→412) → grid **146×103** (not 146/103-anything the brief spelled as
  "584×416/104", an off-by-a-few). Use ceil-div `pH=(h+ps-1)/ps` and guard the conv
  reads (`if (iy>=h||ix>=w) continue`). Stage-0 PatchMerging then pads the odd W
  103→104 → 73×52 = 3796, matching `enc_stage0`. Getting this wrong shifts every
  token and tanks stage 0 immediately — the diff harness localizes it in one run.
- **The decoder is fresh but small: pre-norm mBART.** LN *before* each sublayer with
  the residual around it (unlike BART/RoBERTa post-norm), embed = tokens·√1024 +
  learned-positions[**pos+2**] then layernorm_embedding, GELU-erf FFN, untied
  lm_head, eps 1e-5 throughout. Teacher-forced full-sequence forward (causal self +
  unmasked cross) doubles as the greedy engine (re-run over the growing prefix; L≤128
  so O(L²) is free). eos is the tokenizer's `</s>`=**54**, not generation_config's
  stale mBART `2` — same class as the TexTeller decoder-start trap.
- **q8_0 needs no keep-guard here (unlike TrOMR).** Flova's Swin is all linear; the
  only conv is the patch embed, which the quantizer flattens to `[48,128]` — ncols
  48 %32≠0 so Q8_0 auto-skips it (kept F32), and `rpb_index` (ncols 100) is likewise
  auto-kept, preserving its integer indices. Result: byte-exact decode on all three
  samples (573→162 MB, cos_mean 0.9997). The diff harness flags enc_stage3/enc_output
  `cos_min` 0.987 — a single worst-token per-row cosine (max_abs 0.93 on one
  high-activation token), the honest q8_0 floor, NOT a decode error (argmax 40/40,
  greedy byte-exact). Don't chase it with F16; the output is already exact.

## Transcoda engine port: clean-room from an oracle, and the four things that made greedy diverge despite cos=1.0 (2026-07-13, src/transcoda_ocr.cpp)

Transcoda-59M (ConvNeXt-V2-Tiny encoder → 2-layer projector + 2D-sinusoidal-PE
bridge → 8-layer pre-LN **RoPE** cross-attn decoder → Humdrum `**kern`) is the
first CrispEmbed engine written **clean-room**: the weights are CC-BY-4.0 but the
reference *code* is AGPL, so the engine was written only from the paper, the HF
config/data files, and an activation **oracle** (running the AGPL model = facts).
Every architecture question the handover left open was answered by the oracle in
one probe, not by reading source:

- **Introspection beats the handover.** `inspect.signature` + `named_modules` +
  `VisionFrontendOutput` field dump pinned every fact: grid is **46×32=1472**
  (handover said 47×33), norm is **[-1,1]** not ImageNet, RoPE is **torchtune
  adjacent-pair = ggml `ROPE_TYPE_NORMAL`** (not NEOX like every HF port here),
  cross-attn is **dual-memory** (K = projector-out+2D-PE, V = projector-out raw —
  the SMT pattern), the final encoder `layernorm` is **dead** (proj_input ==
  last_hidden_state exactly), and the 2D-PE matches SMT's `/C=512` formula. First
  build: all stages cos=1.000000, argmax 191/191, native preproc bit-exact.
- **Per-stage cos=1.0 does NOT mean the decode matches — four separate bugs made
  free-running greedy diverge while teacher-forced parity was perfect:** (1) the
  KV-step cached the RoPE'd K as a bare `reshape` **view**, whose source buffer is
  clobbered post-compute ([[set-output-on-view-stale]]) → `ggml_cont` it; (2) the
  output joined tokens with `/`, but kern tokens **contain** `/` (e.g. `*M2/4`) and
  the structure lives in literal `\n`/`\t` vocab tokens → concatenate directly, no
  separator; (3) `repetition_penalty=1.1` was applied once **per occurrence** of a
  token in the running sequence, so frequent tokens (the `\n` record separator) got
  ÷1.1^k and were crushed — HF applies it once **per unique token**
  ([[repetition-penalty-per-unique-token]]); (4) the oracle dump was capped at 192
  tokens, so "byte-exact for 442 chars then 18 extra" was the *oracle* truncating,
  not the engine. The tell that unlocked (1)–(3): teacher-forced argmax was 191/191
  but greedy still diverged ⇒ the bug is in the sampling loop / KV path, never the
  graph. With all four fixed, greedy is byte-identical to the HF reference.
- **q8_0 conv keep-guard, again.** Like TrOMR, the ConvNeXt-V2 stem/downsample/
  depthwise conv2d kernels are reshaped to 4D in-engine; the downsample convs have
  `IC·KH·KW`=384 (%32==0) so they slip past the `ncols%32` auto-skip and quantize
  → `ggml_cpy(q8_0→F16)` aborts (`ggml_dup` "fatal error"; Metal/CPU have no
  q8_0→F16 cast). The converter's short names (`enc.embed.patch`, `.ds.conv`,
  `.dw.`) didn't match the quantizer keep patterns; added them. Pointwise convs
  (pw1/pw2) are matmuls in-engine and quantize fine. q8_0 (65 MB): decode still
  byte-identical to the reference (enc cos 0.988 = the honest q8_0 conv floor).
- **Persistent device-KV cache = 2.4–4× faster decode, byte-identical.** The first
  KV path shuttled everything through host vectors and re-uploaded the cross K/V +
  growing self K/V **every step**, plus `ggml_concat`. Moving cross K/V (computed
  once) and self K/V (written in-graph via `ggml_cpy` into a position-view, read
  back via a `[C,pos+1]` view — the got_ocr pattern) into a persistent backend
  buffer removed all per-step host traffic. Byte-exact vs the host path on Metal
  AND CPU; 2.4–4× faster per back-to-back A/B (variance is machine load — measure
  both arms back-to-back, never idle-vs-loaded). Default flipped; the host path
  stays behind `TRANSCODA_OCR_HOST_KV=1` and the O(L²) recompute behind
  `TRANSCODA_OCR_FULL_DECODE=1` for regression bisection.

## Importing a llama.cpp LLM: un-permute q/k, because llama.cpp rewrites them for its interleaved RoPE (2026-07-12, SmolVLM import)

Merging a stock llama.cpp **SmolVLM-256M** (arch=llama LLM + idefics3 mmproj)
into CrispEmbed's `smoldocling` engine: the model loaded, ran the full vision +
connector + LLM pipeline, and produced **fluent garbage** ("The The The [ [").
Vision was fine; the LLM's attention was scrambled.

Root cause: **llama.cpp's `convert_hf_to_gguf.py` permutes the q_proj/k_proj
weights** (`LlamaModel.permute`: reshape `[n_head, 2, head_dim/2, …]` →
swapaxes → reshape) so that ggml's *interleaved* RoPE (`GGML_ROPE_TYPE_NORMAL`)
reproduces HF's *rotate_half* result. CrispEmbed's native converters read HF
weights **verbatim** and apply rotate_half RoPE directly. So a llama.cpp LLM's
q/k are in the wrong layout for a CrispEmbed loader → every RoPE'd dot product
is wrong → the decoder loops on a few tokens. Fix: **un-permute q and k back to
HF layout in the merge** (inverse of the reshape/swapaxes). It only reorders
OUTPUT rows, and Q8_0 quantizes each row independently, so it's a byte-exact
row-shuffle — no dequantization. q uses `n_head`, k uses `n_head_kv`; v and
everything else copy verbatim. After un-permuting, the merged SmolVLM OCR'd
`The quick brown fox…` correctly on Metal.

Durable rule: **un-permute q/k when ingesting a llama.cpp NORMAL-RoPE LLM
(`arch=llama`/mistral/gemma) into a CrispEmbed HF-layout graph.** Symptom is
always "fluent but wrong / repetitive," never a crash — shapes are identical,
only row order differs. (Same class as the [[flashattn-ext-already-permutes]]
bugs: right values, wrong arrangement.)

**CRUCIAL refinement (2026-07-12, InternVL import):** the un-permute is
**arch-dependent**. llama.cpp permutes q/k ONLY for interleaved/NORMAL-RoPE
arches; **NEOX-RoPE arches (`qwen2`) are already in HF layout** and must be copied
**verbatim**. InternVL2.5-1B's LLM is `arch=qwen2` — un-permuting it produced
"the! The title! It's not!" garbage; copying verbatim gave correct OCR + full
diff-harness parity with the native converter. So: `needs_unpermute = arch in
{llama, mistral, gemma}`, else verbatim. Two more recurring transforms: the
ViT/SigLIP/InternViT FFN is **name-inverted** in llama.cpp's clip export (map
fc1/fc2 by OUTPUT dim, NOT name — SmolVLM has `ffn_down`=fc1 while InternVL has
`ffn_up`=fc1, opposite, so name-matching is guaranteed wrong), and the 4-D Conv2d
patch weight flattens to 2-D by a pure C-order shape relabel (byte-identical).
InternVL also needs **QKV re-fusion** (mmproj splits attn_q/k/v; the loader wants
a fused `attn_qkv` — byte-concat [q;k;v], and vision has no RoPE so no permute).
See `models/merge-llamacpp-{smolvlm,internvl}-gguf.py` + their tests.

## Diff-harness: match the INPUT the harness feeds, and isolate against the native converter (2026-07-12, import validation)

The user's standing rule — **"always test against the diff-harness intermediates
AND ground-truth outputs, never just the output"** — earned its keep on the
InternVL import, but ALSO showed how a mis-run harness lies. First read of
`build/test-internvl2-diff` on my import showed `vis_patch_embed cos=-0.936`
(near anti-correlation) while OCR was perfect — alarming. **Root cause was NOT a
real defect: I dumped the HF reference with `--image test_text.png` (a real,
tiled image) while `test-internvl2-diff` feeds a SYNTHETIC GRADIENT** (a clean
single 448² tile, no dynamic tiling). Apples-to-oranges inputs → garbage cosine.
Re-dumped WITHOUT `--image` (gradient, matching the harness): `vis_patch_embed`
jumped to **cos 0.999999**, and my import was **identical to the native converter
at every stage** (both 0.999999 patch-embed; both `vis_proj_output` −0.098; LLM
identical modulo my Q8_0 vs native f16). So the import is genuinely correct.

Fix (this commit): the dump tool stamps `diff.input_mode` (`gradient` or
`image:<name>`) into the reference GGUF; the internvl2 harness **refuses** a
non-gradient reference with a clear message instead of reporting a misleading
anti-correlation. Prevents the exact trap.

Two durable rules: (1) **a diff harness is only valid when both sides see the
same input** — the gradient-vs-tiled-image mismatch produced a confident, wrong
−0.936. (2) **Isolate my-bug-vs-baseline by running the native converter on the
same HF model and diffing IT against the same reference**: identical cosines ⇒
import ≡ the blessed path, and any residual gap is pre-existing. Here that gap is
`vis_proj_output cos=-0.098` — a **pre-existing InternViT-vs-HF projector-stage
parity gap present in the native converter too** (doesn't break OCR; the
`pixel_unshuffle_v2` order matches between dump and engine, so it's a deeper
layer/projector divergence, not the interop). Ground truth for an imported GGUF
is the source engine itself: `llama-mtmd-cli` on the same file. And the original
point still stands — the LLM read correct text over that mis-measured vision
cosine, so the output check alone would have hidden it. See
[[validate-intermediates-and-outputs]].

## VL "runs but ignores the image": use the inject-embeds discriminator BEFORE diffing the vision tower (2026-07-12, mmproj reverse interop)

Loading a stock llama.cpp Qwen2-VL-2B into CrispEmbed, `--ocr` ran and produced
fluent-but-wrong output ("The text in the image is not visible"). I spent a long
time building an HF diff-harness on the *vision tower* (fed CrispEmbed's exact
patches to HF's Qwen2-VL vision model) and found cos 0.957 — close but imperfect,
which nearly sent me chasing a phantom ViT bug for hours.

**The decisive test was one line of thinking:** if the vision were the problem,
better embeds would help. So I added a `CRISPEMBED_LOAD_MERGER=path` override
that swaps the computed image embeds for a dumped tensor, and fed it three
inputs: HF's *perfect* embeds, all-zeros, and random. **All three produced the
IDENTICAL output.** That instantly proves the image is being *ignored entirely* —
the bug is LLM-side conditioning (splice / prompt / positions), NOT the vision
tower, and the cos-0.957 was a red herring.

Root cause: `qwen2vl.image_token_id` is absent from llama.cpp GGUFs. CrispEmbed
had **two mismatched defaults for the same concept** — the prompt builder's
`image_pad_id` defaulted to `<|image_pad|>=151655`, while the vision-text splice's
`image_token_id` defaulted to `0`. So the prompt emitted 151655 pads while the
splice searched for token 0, matched nothing, and never replaced any embedding.
Fix: default `image_token_id` to 151655 (match the prompt) + write the token IDs
in the converter so the GGUF self-describes.

Durable rule: **for any "multimodal model runs but ignores the modality," run the
inject-{perfect, zeros, random} discriminator FIRST.** Identical outputs ⇒ the
modality is dropped (conditioning bug — check the token-id/splice/positions);
different outputs ⇒ the encoder is genuinely wrong (then diff the tower). It
separates encoder bugs from conditioning bugs in a single cheap test, before any
per-layer harness. Sibling gotcha from the same session: two independent
defaults for one metadata key silently disagree — grep for every place a special
token id is read and make the defaults identical (or better, write the key).

Also from this port (llama.cpp qwen2vl mmproj → CrispEmbed): **llama.cpp INVERTS
the ViT `ffn_up`/`ffn_down` names vs the projection direction** — `ffn_down` is
fc1 (hidden→intermediate), `ffn_up` is fc2, provable by bias widths
(`ffn_down.bias`=[intermediate], `ffn_up.bias`=[hidden]). Map fc1/fc2 by the
output dim, never the name. And to localize a `ggml_can_mul_mat`/`ggml_can_repeat`
abort, temporarily print `a->ne`/`b->ne` right before the assert in ggml.c
(revert after) — it names the exact tensor + shapes in one run.

## Two "inverse" interop scripts drift silently unless a round-trip test runs the REAL scripts end-to-end (2026-07-12, mmproj hardening)

Adding a regression test for the llama.cpp⇆CrispEmbed Qwen2-VL mmproj interop
immediately exposed two silent bugs that had shipped, both invisible to the
existing per-script self-tests:

1. **Name-map divergence.** The merge script (mmproj→CrispEmbed) was changed to
   keep llama.cpp-native tensor names verbatim (`v.blk.*`, `mm.*`, `v.post_ln.*`)
   because that's what the `qwen2vl_ocr` loader actually reads — but the export
   script (CrispEmbed→mmproj) still read the *legacy* `vis.blocks.*`/`proj.*`
   names the merge script no longer produces. So `export --in <real-merged-gguf>`
   found **zero** vision tensors and errored. The export's own `--self-test`
   passed the whole time because it round-tripped *synthetic legacy names it
   generated itself*, never touching a file the merge script wrote. A per-script
   self-test that fabricates its own input can't catch cross-script drift.

2. **Hardcoded patch dtype.** The merge's temporal-patch concatenation did
   `np.frombuffer(..., dtype=np.float16)`, silently corrupting any **F32**
   `v.patch_embd.weight` (common in real mmproj files). Fix: view by the tensor's
   real element *width* (`GGML_TYPE_META` byte size → `uint8/16/32/64`); the
   concat only reorders whole elements, so a width-correct integer view is
   byte-exact for any unquantized dtype and never interprets the float value.

Durable rule: **for a pair of scripts claimed to be inverses (A→B, B→A), the
only test that matters feeds a fixture through the REAL A then the REAL B (via
subprocess) and asserts the output equals the input** — here, a synthesized tiny
mmproj → `merge` → `export` → mmproj, all 40 vision tensors byte-identical, for
each patch dtype (F16 and F32). Self-tests that validate one script against its
own synthetic data give false confidence. See `tests/test_mmproj_interop.py`
(pure Python, no model download, wired into the `regression.yml` smoke tier).

## `ggml_set_output` on a reshape/view does NOT protect the underlying source buffer — snapshot reads back garbage (2026-07-12, C4)

Building the C4 cross-call prefix KV cache, the plan was: run a prefix-only
graph, mark each layer's post-rope **K** and **V** as outputs, read them back
after compute, reuse them in a suffix-only graph. K read back correct; V read
back *garbage* — and the corruption differed between a P=9 and a P=T=19 build of
the same prefix, so it wasn't a value error, it was a **stale buffer**.

Root cause: in the graph, `K = ggml_rope_ext(...)` is a **fresh contiguous
tensor** (rope allocates its own output), so `ggml_set_output(K)` pins a buffer
that is genuinely K's. But `V = ggml_reshape_3d(v_proj, ...)` is a **VIEW** that
aliases the `v_proj` matmul output. `ggml_set_output` on the view flags the view
tensor, not `v_proj`'s buffer — so gallocr freely reuses `v_proj`'s buffer for a
later op, and the post-compute `tensor_get(V)` reads whatever overwrote it. The
*forward* was fine (flash-attn consumed V via its permute-view before the reuse,
so `prefix_hidden` read cos 1.0 vs the full path) — only the **snapshot** was
stale. Symptom: reused-prefix embeddings cos ≈ 0 (orthogonal), while the cached
K matched byte-for-byte and every other input (mask, positions, concat order,
rectangular flash) checked out. Cost most of a debugging cycle precisely because
the forward looked correct.

Fix: `K = ggml_cont(g, K); V = ggml_cont(g, V);` immediately before
`ggml_set_output` — `cont` materializes each into its own contiguous buffer that
`set_output` then protects. Rule: **before snapshotting a graph intermediate for
read-back, `ggml_cont` it if it is (or might be) a view** — reshape/permute/slice
results are views; matmul/rope/norm/add results are fresh. This is the
read-back-a-view sibling of the existing "sched aliases many set_output
snapshots to one buffer" and "gallocr reuses input buffers as scratch" gotchas —
localize it by printing per-layer norms (aliasing → identical norms) and by
diffing the snapshot against an independent recompute (here: cache built with
P=lcp vs P=T, first-P tokens must match; K L1=0, V L1=15075 pinpointed it).

## Per-step CPU→device KV cache re-upload is the #1 autoregressive perf killer (2026-07)

math_ocr's TrOCR decoder stored self-attention K/V in CPU `std::vector<float>`
and re-uploaded the entire growing cache via `ggml_backend_tensor_set` every
decode step. With 200 steps this is O(n²) total transfers — plus constant-cost
cross-attention K/V re-uploads (6 layers × n_enc × D × 4B per step).

On Metal/WebGPU this caused ~19s/region (device sync overhead dominates).
On WASM it caused OOM (ggml hash table overflow from graph rebuild pressure).

**Fix:** Adopt the lightonocr.cpp persistent KV cache pattern:
- Allocate `ggml_tensor` for K/V on the compute device once at max_seq
- Write new K/V per step via `ggml_cpy` into `ggml_view_2d` at offset `n_past` (O(1))
- Read full history via `ggml_view_2d` (zero-copy on device)
- Cross-attn K/V uploaded once before the decode loop

**Result:** 19s→4.4s/region on CPU (4.3x), WASM crash eliminated.

**Rule:** Any autoregressive decoder with `std::vector` KV cache + per-step
`tensor_set` should be migrated to persistent device tensors. The pattern is
in lightonocr.cpp (GQA + RoPE variant) and now math_ocr.cpp (simple MHA).

## A weight's `t->data` is a DEVICE pointer on CUDA/Vulkan — host reads SIGSEGV; Metal/CPU hide it; local Ampere reproduces the class but not arch-specific garbage (2026-07)

A model weight loaded on a device-local backend (CUDA/Vulkan/SYCL/ROCm-HIP) has a
**device pointer in `t->data`**. Any host-side dereference — `memcpy(dst, t->data,…)`,
`(const float*)t->data`, `traits->to_float(t->data,…)`, `return (const float*)t->data`
from a `*_to_f32` fast path — **segfaults**. It is invisible on CPU and on **Metal**
(Apple unified memory is host-visible, so `t->data` is a valid host pointer there),
which is why this whole class "works on Metal/CPU, crashes only on CUDA." The fix is
always the same: keep the zero-copy fast path only when
`!t->buffer || ggml_backend_buffer_is_host(t->buffer)`, otherwise copy through
`ggml_backend_tensor_get` (which does the device→host copy). Reference-correct
implementations already in-tree: `granite_vision_ocr.cpp` (host-visibility guard),
`glm_ocr.cpp` `read_model_w`, and the `surya_det` / `instructir` / nafnet / safmn
helpers.

Found + fixed across **8 engines** (deepseek-ocr2, unlimited_ocr, math_ocr,
smoldocling_ocr, parseq_ocr, tesseract_lstm, dat_sr, tbsrn_sr); deepseek/dat/tbsrn
runtime-verified on a local RTX A1000 (sm_86). A full `->data` census (52 refs / 14
files) plus a ggml-host-accessor check (`ggml_get_f32_1d`, `ggml_get_data_f32` —
none used) confirms no instance remains.

Lessons: (1) a `*_to_f32(t, buf)` fast path that **returns `t->data`** is doubly
dangerous — it silently skips fusion on F32 models (the DAT bug below) AND, once you
"fix" that by copying the returned pointer (`buf.assign(p, p+n)`), the copy now
dereferences a device pointer and crashes on GPU. Route the read through the backend
buffer, don't paper over it. (2) **A local NVIDIA GPU is worth having even if it's a
different arch than CI:** the RTX A1000 (Ampere sm_86) reproduced this whole class in
minutes, but does NOT reproduce the *arch-specific* faults — glm-ocr / internvl2 /
qwen2vl-3b vision garbage and the swinir/dat/tbsrn free-after-load teardown SIGSEGV
all only manifest on Kaggle's older Turing (sm_75) / Pascal (sm_60) GPUs. So a modern
local GPU cleanly *separates* backend-agnostic bugs (fixable + verifiable locally)
from older-arch numerical divergence (needs the old hardware). (3) Metal's
host-visible buffers make it a poor oracle for this class — a CUDA-only crash can sit
latent through all Metal/CPU testing.

## A cratered embedding (cos ~0) can be a stale SHIPPED GGUF, not the engine/ref/pooling — test a fresh re-export (2026-07)

`bidirlm-omni`'s text embedding read **cos 0.047** vs an independent HF mean-pool ref, and the
engine loudly warned `stale GGUF — recovered mrope_section=[24,20,20] … re-export with the latest
converter`. The obvious suspects were all wrong: (a) NOT pooling — `1_Pooling/config.json` confirms
`pooling_mode_mean_tokens: true`, which the ref used; (b) NOT `mrope_section` — that warning still
prints on a freshly-converted GGUF, but for **text-only** input all three mrope channels share the
same position, so it's a red herring; (c) NOT the engine — the **vision** tower on the very same
GGUF passed at 0.997. The actual bug was in the **tensor weights/layout the OLD converter produced**.
Proof and fix in one step: re-run the current `convert-decoder-embed-to-gguf.py` → text jumps to
**cos 1.000000 (f16) / 0.9992 (q8_0)** and vision still passes (0.9966). Lessons: (1) when a shipped
GGUF is old, **a fresh re-export is the cheapest ground-truth test** — don't spend cycles theorizing
about pooling/rope/quantization first; (2) an engine's own "stale GGUF" warning is a strong signal —
believe it; (3) a bug can hit one tower (text: 0.047) while another (vision: 0.997) is fine, so
**verify every path a shared GGUF feeds** before/after a re-export (one omni GGUF serves both
`bidirlm-text` and `bidirlm-vision`); (4) `crispembed-quantize <in.gguf> <out> q8_0` re-quantizes a
converted f16 without a separate llama.cpp — but imatrix k-quants still need the imatrix pipeline.

## A source file shared across two repos can't be kept byte-identical if the repos' clang-format differs — sync the LOGIC (2026-07)

`pcs.cpp` lives in both CrispEmbed (`src/pcs.cpp`, fallback) and CrispASR
(`crisp_punc/src/pcs.cpp`, the copy that ships). The old rule was "keep them byte-identical modulo the
`#include`" — but that is **unachievable**: the two repos have different `.clang-format` (CrispEmbed
`PointerAlignment: Middle` + single-line ifs; CrispASR `Left` + broken ifs), each enforced by its own
lint CI, so every commit reformats the shared file differently. Chasing byte-identity just fights the
formatters. The right invariant is **logical** identity: check with a whitespace/comment-insensitive
diff (strip comments, collapse whitespace, normalise the include name), not `diff`.

Doing that here surfaced the only *real* drift — one line: this copy called `ggml_backend_init_best()`
while CrispASR's called `crispasr_init_gpu_backend()` (the #214 `--gpu-backend` selector). Converged by
adopting `crispasr_init_gpu_backend()` here too (via the already-vendored `core/gpu_backend_pref.h`); it
falls back to `ggml_backend_init_best()` when no preference is set, so the change is **behavior-neutral
by default** and gains `--gpu-backend` support. Runtime-verified: `test-punct-diff` on
`pcs-xlmr-base-q4_k` (sha256-checked vs HF) punctuated + capitalised correctly on Metal. Same lesson
applies to the other cross-repo duplicates (`core/gguf_loader.{h,cpp}`).

## `ggml_set_output` cannot corrupt computed VALUES — verify a CUDA fix before believing a plausible pattern (2026-07)

Chasing a CUDA-only bug in `lfm2_embed`'s ColBERT path (`colbert_output` cos **0.57** on a
P100 vs **0.998** on CPU/Metal), the in-engine localizer showed the pre-projection backbone
`hidden` itself is anti-correlated on CUDA (cos −0.70, max_abs 13) — while the *identical*
backbone in the dense-encode graph passes 20/20 on the same CUDA. A tempting hypothesis: the
ColBERT path does `hidden = cur; ggml_set_output(hidden)` while `cur` also feeds the projection
`mul_mat`, so marking a live intermediate as an output "corrupts" it on the scheduler. **Wrong.**
Cont-copying it (`hidden = ggml_cont(cur)`) produced **byte-identical** CUDA numbers to 6
decimals, and mechanically it had to: **`ggml_set_output` only sets a flag that changes a
tensor's BUFFER LIFETIME after compute — it never changes the tensor's computed values.** Since
`colbert_output` is computed FROM `cur` during the graph, `cur` is being *computed* wrong on
CUDA — a real compute-time divergence (graph-structural: the extra projection output, the
conditional `ggml_backend_sched_reserve` absent in the dense path, or the two paths sharing one
`ctx->sched`), not an output-flag or read-back artifact.

Two transferable rules:
1. **Distinguish compute-time corruption from read-back corruption.** If a *downstream* result
   (here `colbert_output = mul_mat(proj, cur)`) is also wrong, the tensor is mis-*computed*;
   output flags / view reads are irrelevant. Only if *just* a post-compute `tensor_get` is wrong
   should you suspect buffer reuse / view residency.
2. **Never mass-apply a statically-swept pattern before ONE empirical confirmation.** A codebase
   sweep found ~9 engines with the same `set_output`-on-live-intermediate shape
   (got_ocr/glm_ocr/internvl2/qwen2vl/math_ocr/pcs) — but since the pattern was *refuted* as the
   cause here, none are confirmed bugs. Editing 9 core engines on an unvalidated theory would
   have been pure risk. (See also "independently reproduce a handover's root-cause claim before
   building on it" below.)

**RESOLVED (2026-07, `fix/lfm2-colbert-cuda-multivec`).** The real cause was the third candidate
above: `encode_multivec` **re-allocated the same graph object it had just passed to
`ggml_backend_sched_reserve`**. `ggml_backend_sched_reset` does not null `tensor->buffer`, so the
reserve pass's stale buffer/residency assignment was reused by the follow-up
`ggml_backend_sched_alloc_graph` — mis-computing the backbone (`hidden` cos −0.70) and hence
`colbert_output` (0.57). Metal tolerates it (0.998); CUDA corrupts silently. The **dense path
(`lfm2_embed_encode_to`) never hits this because it frees `g` and rebuilds a fresh graph after
reserve** — which is exactly why the same backbone passes 20/20 in the dense graph but fails in
the ColBERT graph on the same device. Fix: factor ColBERT graph construction into a `build_graph()`
lambda and rebuild a fresh graph after the bucket-change reserve, mirroring the dense path. General
rule (already stated below for graph reuse): **the graph you hand to `sched_reserve` is dead — never
`sched_alloc_graph` it; build a fresh one for alloc+compute.** **Confirmed on a Tesla P100 (compute
6.0) A/B** — same GPU, same q8_0 model + HF-f32 ref, github `main` vs the fix branch built side by
side: `main` `colbert_output` cos **0.571643** (FAIL, backbone `hidden` −0.702160, reproducing the
handover to 6 decimals) → fix cos **0.995885** (PASS, `hidden` +0.922054). The 0.99 regression
guardrail now passes on CUDA.

**Sweep — is it systemic? No.** All six `ggml_backend_sched_reserve` call sites were audited:
`lfm2_embed.cpp` dense (455, already rebuilt), `lfm2_embed.cpp` colbert (635, the bug — now
rebuilds), and the three `crispembed.cpp` encoder paths (1204 `encode_tokens`, 1495
`encode_tokens_packed`, 1842 `run_encoder_raw` sparse/colbert/rerank). The `crispembed.cpp` three
are safe: each reserves a separate `measure_gf = build_encoder_graph(T_bucket)` and then allocs a
**distinct** `gf = build_encoder_graph(T)`. Even though `build_encoder_graph` re-inits over the
shared `ctx->compute_meta` buffer (so `gf`'s tensors reuse `measure_gf`'s addresses), each call is a
fresh `ggml_init` + `ggml_new_tensor`, which nulls `buffer`/`data` — so the allocated graph never
carries the reserve pass's stale pointers. lfm2_colbert was the lone site that reserved and
allocated the *same* object. (Distinct from the InternVL2 cached-graph-reuse class below, which was
found and fixed separately.)

## Reading a QUANTIZED weight to CPU: size the copy by `ggml_nbytes(t)`, not `n_elem * 4` (2026-07)

The shipped `pcs-xlmr-base-q4_k.gguf` (the CLI `pcs` entry) **crashes on every inference** —
`ggml-backend.cpp:349 GGML_ASSERT(offset + size <= ggml_nbytes(tensor))`. `pcs_process` pulls its
SBD/truecase FC-head weights to CPU with `ggml_backend_tensor_get(t, buf, 0, n_elem*sizeof(float))`,
assuming F32. But the q4_k converter **quantizes those head tensors** (`head.post.fc1/fc2` Q4_K,
`head.sbd.fc2`/`head.tc.fc2` Q4_0), whose `ggml_nbytes` is ~0.56 B/elem ≪ `n_elem*4` → out of
bounds. fireredpunc is immune because it computes its head **in-graph** via `ggml_mul_mat` (ggml
dequantizes natively) with an F16 cls weight. Lessons: (a) any code that reads weights to CPU must
size by `ggml_nbytes(t)` and dequantize (`ggml_get_rows` / a tiny F32 `ggml_cpy` graph), never
assume F32; (b) prefer computing small heads **in-graph** so quantization is transparent; (c) a
"perf: cache the head weights at init" refactor (wave commit `4a498d1`) is exactly where this class
of assumption sneaks in. (Impl lives in sibling repo `CrispASR/crisp_punc/src/pcs.cpp`.)

## bidirlm-vision parity: gate on per-token MEAN cosine, not worst-row min (massive-activation deepstack) (2026-07)

Building the bidirlm vision `-ref.gguf` guard surfaced a metric trap. The vision
tower's **deepstack** slabs carry a few "massive-activation" rows (`max_abs ~5` vs
~0.1 typical). At q8_0 those quantize imperfectly, so the **worst-row** cosine
(`cos_min`) of `deepstack.1` is **0.43** while the **per-token mean** is **0.9938** —
the model is correct, but a min-over-rows metric reads as a catastrophic FAIL. The
validated Python parity (`test_bidirlm_vision.py`) gates on the mean, so the C++
diff must too: `crispembed_diff::Ref::compare` returns both `cos_min` and `cos_mean`
— gate on `cos_mean`. A graph-scramble regression still craters the mean to ~0, so
the guard keeps its teeth. Emit `<stage>: cos=<mean> max_abs=…` (the esrgan/safmn
`cos=` convention `run_one.py` parses; `cos_min=` would make it gate the worst row).
The same heavy tail is why **q4_k image_embeds sits at 0.97, not a bug** — the
massive dims exceed 4-bit range. Corollary for any vision-tower diff: check whether
the tensor has heavy-tailed rows before choosing min vs mean, and derive
`n_patches` from `grid_thw` (Σ t·h·w), not `pixel_values`' shape — GGUF stores dims
column-major, so a naive `shape[0]` reads the patch dim, not the count.

## Encoder batching: packed block-diagonal is O(T_total²); rectangular 4D per-item mask is O(B·T²) (2026-07)

Two ways to batch B bidirectional-encoder sequences into one graph, with very different
scaling:

**Packed (block-diagonal).** Concatenate all sequences into one length-`T_total = ΣTᵢ`
stream (`B` stays 1), build an F16 `[T_total, T_total]` mask that is 0 within each
segment and −∞ across, feed it to `flash_attn_ext`. Positions restart per segment. This
is the proven `bidirlm_vision` pattern and is **bit-parity** with per-sequence encoding
(masked keys contribute `exp(−∞)=0`). BUT ggml's `flash_attn_ext` still *computes* every
masked cell, so attention is **O(T_total²)** — for many short sequences that's
catastrophic (packing 128×15-token texts into one 1920-token stream measured a **3.7×
slowdown**). Capping the pack into greedy token-budget groups bounds it but never beats
the alternative.

**Rectangular 4D per-item mask.** Keep sequences as separate 4D batch items
`[hd, T_max, nh, B]` (pad to `T_max`), and mask *padding only* with a per-item mask
`pad_mask [T_max, T_max, 1, B]` (−∞ on key columns `k ≥ len_b`, independent of query;
padded query rows are discarded in pooling). Attention is **O(B·T_max²)** — a factor `B`
cheaper than packing when lengths are similar. Length-sort + chunk to keep `T_max` tight.
Measured **1.18×–1.48× faster than sequential AND packed**, parity cos 1.0/0.9999697.

**ggml/Metal support for the per-batch mask (the enabling fact).** The pinned ggml
`flash_attn_ext` asserts only `q->ne[2] % mask->ne[2] == 0` and `q->ne[3] % mask->ne[3]
== 0` (heads/batch broadcast) — **no** `GGML_KQ_MASK_PAD` and no `n_q` padding in this
version. A `[T,T,1,B]` mask (ne2=1 broadcast over heads, ne3=B per item) is legal, and
the Metal kernel indexes it per batch via `(iq3 % ne33)*nb33` (`ggml-metal.metal` ~5872).
So a per-item 4D mask runs on Metal. (Could not empirically confirm here — this sandbox
has **no GPU; the whole tree runs CPU with `GGML_METAL=OFF`**, so "Metal vs CPU"
benchmarks in this env are CPU-vs-CPU. Keep GPU-default paths opt-in until a real-Metal
A/B.) Both paths live behind env gates (`CRISPEMBED_ENCODER_PACKED` / `_4D`); default
stays the per-sequence loop.

## ModernBERT: three latent bugs — local-path converter, CLS-vs-mean pooling, missing SWA (2026-07)

`gte-modernbert-base` was code-supported (shared BERT-family encoder graph) but had never
been parity-checked; it was broken three independent ways, each masking the next:

1. **Garbage tokenizer (cos 0.46).** `convert-bert-to-gguf.py` defaults to **ollama mode**
   (`ollama_mode = not args.crisp`), which writes a WordPiece tokenizer and *never runs BPE
   detection*. Pass **`--crisp`**. Even in `--crisp` mode, BPE/CLS-pooling/Unigram-score
   detection called `hf_hub_download(repo_id=args.model)`, which **throws on a local path**
   and was swallowed by a bare `except` → silent WordPiece + mean fallback. Fixed with a
   `_resolve_file()` helper (local dir → hub) at all three sites.

2. **Pooling metadata (cos 0.84 with correct tokens).** Once tokens matched, per-layer diff
   showed **all 22 layers cos ≥ 0.99995** — the backbone was perfect — yet the pooled
   embedding was 0.84. Root cause: the loader read `bert.pooling_type` (ollama enum, 1=mean)
   instead of `bert.pooling_method` (crisp enum, 1=CLS), so a CLS model mean-pooled. The
   telltale: `cos(crispembed, HF *mean*-pool) = 0.99999` while `cos(…, HF *CLS*) = 0.84`.
   **Lesson: when the backbone matches per-layer but the final embedding doesn't, suspect
   pooling/metadata, not the graph.**

3. **Missing sliding-window local attention (long-doc only).** ModernBERT alternates global
   (every Nth) and local layers; only the RoPE θ alternated in our graph — the local layers'
   `±local_attention/2` (=64) window mask was absent, so they attended globally. Invisible
   for short inputs (window ≥ seq len), but a 113-token doc dropped to 0.9826. Added a
   per-layer `swa_mask` ([i,j]=0 iff |i−j|≤64) fed to `flash_attn_ext` on local layers only;
   converter emits `bert.local_attention`. Gated by `CRISPEMBED_ENCODER_NO_SWA=1` as an A/B /
   regression-bisection lever. Result: 113-tok 0.9826 → **0.999998**; the compiled
   `test-modernbert-diff` guard uses a >64-span text so disabling SWA craters cos to −0.87.

## Public raw encode APIs must mirror the main tokenizer dispatch (2026-07)

`crispembed_encode_tokens_raw` (and a sibling raw path used by the diff harness) branched
only `use_sentencepiece ? SPM : WordPiece` — **missing the `use_bpe` case** that the main
`encode` path has. So BPE encoders (ModernBERT) were silently tokenized with WordPiece in
the raw API (113 → 103 tokens, garbage), even though `crispembed_encode` worked. Any code
path that re-tokenizes must use the same `use_bpe → SPM → WordPiece` dispatch as the primary
one; a partial copy is a latent per-arch bug. (Separately: the CrispEmbed BPE tokenizer still
diverges from HF on some longer/varied texts — an edge case, not yet chased.)

## EmbeddingGemma: a non-orthogonal Dense bottleneck amplifies tiny backbone discrepancies (2026-07)

EmbeddingGemma-300m is Gemma3 → mean-pool → **Dense(768→3072) → Dense(3072→768)** → L2 →
Matryoshka. Parity vs HF sat at **~0.997, identical at f16 and f32** — so it's *not*
precision. The Dense/pooling code and f32 weights match HF exactly (verified by reading +
a pooling-variant probe: BOS+content+EOS mean is the best match). The residual traces to a
sub-0.9999 discrepancy in the *Gemma3 backbone* pooled output that the learned, non-norm-
preserving 768→3072→768 projection **amplifies** (a ~0.9995 pre-Dense cosine becomes ~0.997
post-Dense). **Lesson: models with a post-pooling projection head amplify backbone error —
0.997 end-to-end can hide a 0.9995 backbone bug; diff the *pre-Dense* pooled vector to
localize.** (The gap was deemed acceptable and left open; per-stage backbone diff is the
next step.)

## ggml v0.10.0 (8be60f83) GPU-teardown regressions: Metal residency abort + CUDA use-after-free + sched CPU-fallback assert (2026-07)

Three aborts appeared *only after* the ggml submodule bump to v0.10.0 (`8be60f83`),
turning GPU runs that worked "two weeks ago" into crashes. All FIXED on
`fix/metal-v0.10-regressions`. None is a CrispEmbed logic bug — v0.10.0 changed
ggml's contract (a stricter GPU-device teardown + scheduler requirement).

**1. Metal residency-set teardown abort (`ggml-metal-device.m:612`).** v0.10.0 added
Metal *residency sets*: a GPU keep-alive cache (default `keep_alive = 180 s`, a
background heartbeat dispatch thread) that wires buffers resident to avoid OS
eviction. It also added a hard teardown assert `GGML_ASSERT([rsets->data count]==0)`
in `ggml_metal_device_free`. Buffers register a set on alloc and *deregister on
free* (device.m:906/920) — so the assert only fires if a Metal buffer is still
alive when the **process-global** device is torn down by a C++ static destructor at
`exit()`. CrispEmbed's one-shot binaries (CLI, `test-*-diff`) leak their backend at
exit (relying on process teardown), which was benign pre-v0.10.0 but now aborts
(SIGABRT / exit 134) **after results are printed** — corrupting exit codes and
making `tests/regression/run_one.py` report false "died from signal 6" on passing
runs. Diagnostic: NaN/abort at *exit* with correct stdout ⇒ teardown, not compute.

Fix (not a ggml patch): ggml exposes a kill-switch, `use_residency_sets =
getenv("GGML_METAL_NO_RESIDENCY") == nil`. A library **constructor** in the
always-linked core TU `src/core/gguf_loader.cpp` sets `GGML_METAL_NO_RESIDENCY` by
default (overwrite=0), running at library load — before any ggml Metal device init.
This restores pre-bump behavior and covers every entry point (CLI, engines, tests,
and all bindings, which load `libcrispembed` as a shared lib so the constructor
runs at dlopen). The residency cache only benefits a **long-lived** process (the
server; a one-shot run is a fresh process, so it buys nothing there); such a host
opts back in with **`CRISPEMBED_METAL_RESIDENCY=1`** and is safe because it frees
its contexts via `crispembed_free` on shutdown (verified leak-clean with residency
re-enabled — jina embed exits 0). The one-shot binaries that *do* leak are handled
by the universal backstop (#3).

**1b. Same root cause on CUDA — use-after-free, no kill-switch.** The identical
"GPU buffer outlives the process-global device static-dtor teardown" leak crashes
CUDA too, as **SIGSEGV** (swinir/dat/tbsrn) or **SIGABRT** (gliner/lfm2_colbert/
layout-heron/lfm2_embed) — correct output, then crash on exit. CUDA has no
`NO_RESIDENCY` equivalent, so #1's Metal switch can't touch it. The fix is the
backstop: skip the static-dtor teardown for one-shot binaries.

**3. Universal backstop — `core_util::clean_exit(rc)`.** New header
`src/core/clean_exit.h`: flush stdout/stderr then `std::_Exit(rc)`, terminating
WITHOUT running static destructors (so ggml's global GPU-device teardown never
runs) — the same os._exit trick downstream already used for the PyTorch-MPS case,
generalized. Applied to the **one-shot** binaries only: the CLI's `main` (rename
real body to `cli_main`, thin `main` routes through `clean_exit`; any
`crispembed_free`/engine-free inside still runs first) and all 88 `tests/*.cpp`
mains. Backend-agnostic: fixes Metal *and* CUDA teardown crashes and preserves the
pass/fail exit code (missing-model still exits 1). Long-lived hosts (server,
bindings) do NOT use it — they free via `crispembed_free`. Verified on Metal with
residency force-enabled (which reproduces the leak crash): test-vit-embed-diff /
test-lfm2-diff / CLI all exit 0 (were 134).

**2. Scheduler CPU-fallback assert (`ggml-backend.cpp:1736`).** v0.10.0's
`ggml_backend_sched_new` now asserts the LAST backend is CPU. Engines that build a
Metal-only sched abort at load. The modern multi-backend engines already append CPU
(got-ocr/layout comment it, the "issue #68" pattern in `fireredpunc.cpp`), and the
SR/OCR `n=1` engines use a dedicated CPU sched — both fine. Only `lfm2_embed` passed
a GPU backend as its single sched backend. Fix: append a CPU fallback (owned+freed
by the ctx). This finally let lfm2 run on Metal (test-lfm2-diff per-layer cos
≥0.9999) — previously the abort masked it entirely.

Takeaway: after any ggml submodule bump, re-run a **Metal** smoke of the one-shot
CLI + a couple of `test-*-diff` binaries and check **exit codes**, not just stdout —
teardown/scheduler contract changes are invisible to numerical parity.

## unlimited_ocr "CLI can't OCR" was a truncated download, NOT a routing/crash bug (2026-07-02)

RESOLVED — no code change. A prior session reported that `crispembed --ocr FILE`
gave empty output / SIGSEGV on `unlimited-ocr` and concluded the CLI `--ocr`
auto-detect path "doesn't handle unlimited (needs `--ocr-pipeline`)". That was
**wrong on both counts**:

1. **The `--ocr` path already handles unlimited.** `crispembed_ocr_model_init` →
   `detect_arch()` maps `general.architecture == "unlimited_ocr"` to
   `OCR_MODEL_UNLIMITED_OCR` (`crispembed.cpp:3316`), and init/recognize/free are
   all wired to `unlimited_ocr_*`. The engine injects its own prompt
   (`document parsing.`) and the **required sliding-window `no_repeat_ngram`
   (35/128)** inside `run_llm_decoder` (`unlimited_ocr.cpp:2541`), so every entry
   path — `--ocr` and `--ocr-pipeline` alike — gets the correct decode config.
2. **The real cause was a truncated GGUF download** — 483 MB of a 2.25 GB q4_k
   file. The loader now reports this cleanly (`core_gguf`: "truncated/corrupt
   GGUF … tensor 'l.blk.1.attn_v.weight' extends past EOF"); an older binary
   SIGSEGV'd reading past the short file, which is exactly what commit `dd9dd2e`
   ("fail cleanly on truncated/corrupt GGUF instead of SIGSEGV") guards against.

**Lesson (adds to [[verify-handover-claims-independently]] / [[dont-assume-env-capabilities]]):**
before diagnosing a model load crash / empty output as a code bug, verify the
GGUF's on-disk size against the HF `X-Linked-Size` (`curl -sIL … | grep -i
x-linked-size`). A memmove/`ggml_backend_tensor_get` EXC_BAD_ACCESS reading a
*bad source address* during load is the signature of either a truncated file or
the quant-size class of bug (cf. the `pcs` q4_k `n_elem*4 >> ggml_nbytes`
finding) — check file integrity first.

## PaddleOCR-VL SIGSEGV was ERNIE head_dim≠D/n_heads + an SPM tokenizer loaded as GPT-2 BPE — NOT a GQA-broadcast bug (2026-07)

RESOLVED. `paddleocr-vl-0.9b` crashed in the shared `qwen2vl_ocr` engine on
both backends. The audit entry below guessed the "8:1 GQA broadcast" hazard from
`fbae7ba`; that was wrong. Two independent, unrelated bugs, found by building a
debug binary (the Release SIGSEGV is a ggml_reshape assert under `-O0`):

1. **The crash: ERNIE-4.5 uses `head_dim=128` while `hidden_size/n_heads =
   1024/16 = 64`.** The engine assumed `head_dim = D/n_heads` everywhere, so the
   Q/K/V reshapes (`attn_q.weight` is `[1024, 2048]` → q_dim=2048, not 1024) and
   the post-attention reshape-to-D overran the tensor → `memmove` SIGSEGV in
   Release, `GGML_ASSERT(nelements(a)==ne0*ne1*ne2)` in debug. Corroborated by
   the mRoPE sections `[16,24,24]` summing to 64 = head_dim/2. Fix: add
   `llm_hparams.head_dim`, read from an explicit `*.attention.head_dim` /
   `key_length` key or derive from `q_w->ne[1] / n_heads` at load time; reshape
   attention output to `q_dim = head_dim*n_heads`, not `D`. No-op for Qwen where
   head_dim == D/n_heads.

2. **End-to-end: the ERNIE vocab is SentencePiece-style (`▁` for spaces,
   `<0xXX>` byte tokens) but was loaded as byte-level GPT-2 BPE.** That silently
   dropped every space/newline in the chat template, so the model saw
   `OCR:Assistant:` and greedily emitted `</s>` (token 2) as the *first* token.
   Plus the chat tokens were hardcoded to Qwen's `<|im_*|>` (151644/5) which are
   out of range for the 103424-row ERNIE embed table → a second `get_rows`
   assert. Fix: detect PaddleOCR-VL, emit the ERNIE template
   `<|begin_of_sentence|>User: <image>OCR:\nAssistant: ` (trailing space is
   load-bearing — dropping it makes the model emit `</s>` immediately), stop on
   `</s>`=2 (per `generation_config.json`, **not** `<|end_of_sentence|>`=100272),
   auto-detect the `▁` vocab → load SPM + add_dummy_prefix, and decode with a
   `▁`→space / `<0xXX>`→byte SPM decoder. fox.png → "The quick brown fox jumps
   over the lazy dog." on CPU+Metal, stops cleanly. qwen2.5-vl-3b (same engine)
   unaffected.

**Meta-lesson (again):** the handover's suspected commit + "GQA broadcast" root
cause were both red herrings. A debug/-O0 build turned an opaque `memmove`
SIGSEGV into an exact reshape assert in one step; the fastest path was to read
the real tensor dims out of the GGUF, not to reason about the suspected commit.
## restormer — RESOLVED (2026-07): broken ggml conv-weight layout + fake-attention block graph

The restormer garbage (below) is fixed in `src/restormer.cpp`. Two layout/graph
bugs, both corrected and validated against a PyTorch ground-truth value and an
end-to-end denoise test (mid-gray σ=25 noise: mean|err| 19.84 → **2.15**; scalar
path==ggml path). **NB (2026-07-02): the original "CPU==Metal to 0" claim held only
on CPU** — a third, residency bug still aborted restormer on Metal until 2026-07-02
(weights on init_best vs a CPU conv sched; see the June-audit section's Fix-notes
Correction below). Now genuinely CPU==Metal.

1. **Conv-weight layout was scrambled for EVERY conv (the real garbage source).**
   The GGUF converter writes conv weights raw as numpy `(OC,IC,KH,KW)` C-order and
   the loader keeps `ne` = the stored dims (NOT reversed). The correct transform to
   a `ggml_conv_2d` kernel is therefore a **plain reshape of the contiguous bytes to
   ggml `[KW,KH,IC,OC]`** — no permute, no transpose, no shuffle. The old load-time
   pre-permute (an oc-fastest shuffle) + the `rst_prep_w`/`rst_conv2d_ggml`
   2D-reshape heuristics all mis-laid-out the kernels. Proof: PyTorch
   `patch_embed[0,0,0]` ground truth = **0.645721**; the old ggml conv produced
   0.161163. **Correction to the prior handover/notes:** `RESTORMER_SCALAR=1` was
   NOT a clean reference — `rst_forward_tile` runs the U-Net convs (patch_embed,
   down/up, reduce, output) through `rst_conv2d_ggml` in *both* modes, so the
   scalar-blocks path was *also* garbage (100px crop: mean 168.9). The
   "convs are fine, bug is only in the block graph" theory was wrong; convs were
   the primary bug. The load-time pre-permute is now deleted entirely.

2. **The ggml MDTA block graph was a fake single-head attention.** It used
   `ggml_norm` (mean-subtract + std) as a stand-in for L2-normalize, did a single
   full `C×C` attention (no per-head split — wrong for the 2/4/8-head levels),
   and **dropped the learned per-head `temperature` entirely**. Rewrote it to match
   the scalar reference: reshape `[HW, d_k, n_heads]`, `ggml_rms_norm` over spatial
   (= L2normalize·√HW, with the √HW² folded into a `temperature/HW` scale),
   per-head batched `mul_mat`, softmax over the key axis. Also `rst_ln2d_ggml`
   used `ggml_norm` for the BiasFree LayerNorm (denoise model is BiasFree,
   `has_bias=0`) — that wrongly subtracts the mean; now computes `x/sqrt(var+eps)·w`
   with no mean-centering, matching `rst_layernorm_bf`.

## nafnet_denoise — RESOLVED (2026-07-02): scrambled conv kernels + Metal/CUDA residency abort

nafnet was the `--denoise` tier-2 engine and the one conv→ggml-wave engine with **no
diff harness** (only reachable via the OCR pipeline), so its regression shipped
unseen. Added `test-nafnet-diff` (mirrors test-restormer-diff; feeds the ref input,
compares the 64×64 output) + a `diff_only` regression-manifest entry (ref
`cstr/nafnet-sidd-GGUF/nafnet-ref.gguf` from `tools/dump_nafnet_reference.py` on
`NAFNet-SIDD-width32.pth`). Disambiguated engine-vs-dumper with a new `NAFNET_SCALAR=1`
A/B gate: scalar conv path scored **0.999998** vs the ref, ggml path **0.538** →
engine bug, dumper faithful. Three sub-bugs in `conv2d_ggml`, all now fixed
(ggml==scalar==ref, cos **0.999998** on Metal AND CPU):

1. **Kernel layout scramble (the 0.538 crater).** Same root cause as restormer: the
   hand-rolled converter writes conv weights as numpy `[OC,IC,KH,KW]` row-major, so
   ggml loads `ne=[OC,IC,KH,KW]` over bytes whose true fastest axis is KW. The old
   4D branch did `ggml_permute(w,3,2,1,0)+cont`, which *physically reorders* the
   bytes as if OC were innermost → every kernel scrambled. The fix copies the
   dequant bytes UNCHANGED into an explicit `ne=[KW,KH,IC,OC]` tensor (mirrors
   swinir/hat/restormer). **Subtlety that wasted a first attempt:** a bare reshape
   only moved 0.538→0.588, because 1×1 convs (conv1/3/4/5, ups) collapse to
   `ggml_n_dims()==2` and were silently taking a *second* (also wrong) 2D branch.
   Building the kernel from the known `oc/ic_g/kh/kw` args sidesteps the dim-collapse
   entirely.
2. **Depthwise needs an F16 kernel.** `ggml_conv_2d_dw` hardcodes its im2col to F16,
   so a F32 depthwise kernel makes `mul_mat(F32 kernel, F16 im2col)` — an unsupported
   type combo that trips `GGML_ASSERT(cur_backend_id != -1)` in sched split (the CPU
   backend can't place the node). Depthwise (conv2) kernels are now F16; regular
   convs stay F32 for best parity (ggml_conv_2d derives im2col from the kernel type).
3. **Metal/CUDA residency abort** (same class as restormer's third bug): weights on
   `init_best` (Metal) referenced from the CPU conv sched aborted graph alloc on
   Metal / segfaulted on CUDA. Fixed by dequantizing each conv weight once into an
   `enc_backend`-resident tensor (cached per source pointer) — which also avoids
   re-dequantizing on every per-block conv call.

Verified end-to-end: `crispembed --ocr --denoise` (got-ocr2 q4_k + nafnet q4_k) on
Metal reads the fox line cleanly; `run_one.py --name nafnet` passes with the diff
live (worst cos 0.999998).

## Metal/CUDA residency-abort class: init_best weights + a CPU enc_sched = crash on every GPU build (2026-07-02)

Auditing whether nafnet's residency abort was shared, I found it is a **systemic
pattern** across the conv-front-end engines, independent of the layout bug. Shape:

```
ctx->backend = init_best();      // Metal/CUDA — used ONLY to load weights (often freed right after)
ctx->enc_backend = cpu_init();   // the conv/graph scheduler is CPU
... conv graph references the GGUF weight *leaf* (w = weight_t / get_raw) ...
ggml_backend_sched_alloc_graph(enc_sched, gf);   // ← ABORTS
```

The CPU sched can't run an op whose input tensor lives in a Metal (`MTL0`) buffer:
`pre-allocated tensor (<name>) in a buffer (MTL0) that cannot run the operation` →
`ggml_abort` on Metal, SIGSEGV on CUDA. These engines do **no** GPU compute (the
Metal backend is pure weight storage), so `<ENGINE>_FORCE_CPU=1` / a CPU-only build
hid it — which is exactly how every one of them passed its "audit."

**Two fixes, pick by whether the main backend does GPU compute:**
- *Pure-CPU engines* (all of these): load weights on CPU (`ggml_backend_cpu_init()`).
  Behavior-preserving — the convs already ran on the CPU sched; only the weight
  buffer moves. This is what restormer/esrgan/safmn/bttr/hmer/posformer/mixtex/
  ppformulanet now do.
- *Engines whose conv sched must stay CPU but that DO use the GPU elsewhere*:
  dequantize each conv weight once into an `enc_backend`-resident tensor (swinir/hat
  preload; nafnet caches via `nafnet_resident`). Don't force the whole model to CPU.

**Affected + fixed (2026-07-02):** nafnet, restormer, esrgan, safmn, bttr_ocr,
hmer_ocr, posformer_ocr, mixtex_ocr, ppformulanet_ocr. **Safe (verified):** swinir,
dat, hat, pan, tbsrn, adair, scunet (preload conv weights onto enc_backend);
instructir (CPU weights); text_sr (fully scalar convs, enc_sched is vestigial/unused).

**Lesson: audit conv→ggml engines on the DEFAULT (GPU) backend, not just CPU.** A
diff harness run under FORCE_CPU is blind to this entire class. The math-OCR engines
(bttr/hmer/posformer/mixtex/ppformulanet) had **zero** regression coverage, so the
abort shipped unseen — same coverage-gap story as nafnet.

## mixtex_ocr decode bugs — RESOLVED 2026-07-02 (two independent bugs + a wrong handover claim)

mixtex_ocr "ran but produced garbage: correct formula start then runaway repetition,
never stops." Two *independent* bugs, and one handover claim that was wrong:

- **Bug A — detok leak (cosmetic).** The decode concatenated raw byte-level BPE piece
  strings, so `Ġ`/`Ċ` leaked verbatim. Fixed by routing through the now-shared
  `core_bpe::unicode_to_bytes` (see the DRY sweep note below).
- **Bug B — never emits EOS (the real one).** `mixtex.decoder.vocab_size` in the GGUF
  was **25678** (the base `tokenizer.tokens` count) but the tied LM head / word
  embedding is **25681** rows. The decoder's argmax + logit loop used `vocab_size`, so
  they only scored tokens 0..25677 — the EOS token **`</s>` = 25678** was **outside the
  scored range and could never win** → decode ran to `max_len` and degenerated. Content
  was always correct (all real tokens < 25678); only the *stop* token was excluded.
  Fix: derive `vocab_size` from `dec.word_embed.weight->ne[1]` (authoritative LM-head
  width), and fix the converter to write `word_embed.shape[0]`, not `len(tokens)`. This
  is the sibling of the [[texteller-decoder-start-token]] trap: VisionEncoderDecoder
  models split token bookkeeping between the base tokenizer, the nested decoder config,
  and the added-special-tokens table, and converters routinely grab the wrong one.
- **Wrong handover claim (verify independently).** The handover asserted "the encoder
  is correct — it recovers `\frac{-b\pm\sqrt{b^2-4ac}}{2a}`." Running **HF itself** on
  the same `formula.png` produced *different* garbage (`\bigvee\overline{...`) — the
  image is out-of-domain / preprocessing-mismatched (HF resamples **bicubic**,
  `resample:3`; the C++ port resizes **bilinear**), and the C++ formula-looking output
  was a coincidental post-divergence hallucination, not evidence of a correct encoder.
  A C++-vs-HF token trace matched exactly for 5 tokens then flipped at a low-confidence
  step — expected from f16-vs-f32 + the resize mismatch, not a graph bug. On in-domain
  formula images the port now decodes cleanly and terminates.
- **Bicubic preprocessing (done).** Ported the resize from bilinear to the shared
  `image_preproc::resize_bicubic_u8_hwc` (a=-0.5 Catmull-Rom + antialias, matching PIL
  `resample:3` on uint8). Verified: C++ now matches HF's greedy token trace through
  **9** tokens (incl. HF's `\bigvee`=12724) vs **5** with bilinear; the residual flip is
  f16-vs-f32 (the GGUF is f16, HF f32), not preprocessing. Bilinear is retained behind
  `MIXTEX_BILINEAR=1` for A/B bisection (dev-guide "keep both paths" rule).

### TexTeller was garbage: decoder FFN activation was hardcoded ReLU (2026-07-13)

TexTeller (ViT + TrOCR VED on the shared `math_ocr.cpp` engine) shipped emitting
garbled LaTeX, yet had a passing regression fixture-shaped history because it was
only ever checked per-stage (encoder cos ~0.999), never on the **decoded output**
(HARD RULE #3). Two pix2tex-specific constants were hardcoded in the shared engine:

1. **Decoder FFN activation hardcoded `ggml_relu`** — but TexTeller's TrOCR decoder
   uses **GELU** (`config.decoder.activation_function == "gelu"`; pix2tex/TrOCR-small
   use relu). Wrong activation on every FFN layer accumulated into output that
   *partially recovered structure* (`\[x \} \} = \frac{-b \pm \sqrt{...}}{2a}` — the
   `\frac` skeleton is right, the rest drifts). **That partial-correctness is the
   diagnostic signature of a small-but-systematic per-layer error** (wrong-but-similar
   activation, subtle norm), NOT a gross wiring bug. Now data-driven from
   `decoder.activation_function` (default relu → pix2tex unchanged); graph uses
   `ggml_gelu_erf`, scalar uses exact `erff`. HF "gelu" == erf, not tanh.
2. **Preprocessing hardcoded mean=std=0.5 + squash-resize** (TrOCR). TexTeller needs
   mean=0.9545467/std=0.15394445 + trim-white-border + aspect-preserving resize
   (short edge→S-1, long edge capped at S) + **white pad** to 448 (its torchvision
   transform; pad value 0 in normalized space == mean grey). Now
   `encoder.image_mean/std` + `encoder.preprocess_pad` from GGUF (defaults reproduce
   pix2tex). TexTeller ships **no `preprocessor_config.json`** — constants live in its
   source; the converter takes `--image-mean/--image-std/--preprocess pad`.

**Isolation that nailed it (dev-guide diff-harness):** injecting the *reference*
`pixel_values` still garbled → not preprocessing; dumping the encoder memory and
diffing vs HF's `ViTModel.last_hidden_state` gave per-token cos mean **0.99933**
(CLS 0.99997) → encoder correct → bug is in the decoder; f16 == q8_0 garbage → not
quantization → graph. GELU flipped it to a **byte-exact** match on both formula
fixtures, f16==q8_0, CPU==Metal; pix2tex-mfr stays byte-identical. Env A/B:
`MATH_OCR_{MEAN,STD,PAD,FFN_GELU}`; graph-isolation hooks `MATH_OCR_{PV_BIN,DUMP_ENC}`.

**Bug class — audit any converted-model engine for hardcoded, model-dependent
constants** (activation fn, input mean/std, resize mode, prefix-token count, RoPE
variant). Sibling status: `ppformulanet_ocr`/`ppformulanet_l` use **tanh-approx**
GELU where MBart's `gelu` is **erf** (minor cos ~0.999 mismatch, and both carry
`expected_text: null` fixtures — unvalidated end-to-end); `mixtex` correctly uses
`gelu_erf`. **Root systemic hole: a fixture pinned from the engine's *own* output
"passes" while enshrining garbage** — golden text must trace to the HF reference,
and `expected_text: null` means "never validated," not "fine." See
[[validate-intermediates-and-outputs]], [[never-blame-quantization]].

## DRY: byte-level BPE decode centralized in core/bpe.h (2026-07-02)

Every OCR/VLM engine hand-rolled the *inverse* of `core_bpe::byte_encoder()`
(`Ġ`→space, `Ċ`→newline, all 256 bytes). Three variants existed: full-correct-but-
duplicated (deepseek/unlimited/glm/internvl2), full-but-re-implementing the byte table
too (smoldocling/granite/qwen2vl), and partial `Ġ`-only that silently dropped `Ċ` and
other bytes (ppformulanet ×2, lightonocr, mixtex=none). Centralized as
`core_bpe::byte_decoder()` + `core_bpe::unicode_to_bytes(piece[, out])`; each engine
keeps only its own *special-token skip policy* (which pieces to drop), since that
genuinely varies. `math_ocr` is intentionally excluded — it's SentencePiece `▁`, whose
codepoint isn't a byte-encoder output. Verified non-regressing by A/B on deepseek
(Cat1), granite (Cat2, identical old-vs-new), ppformulanet-l (Cat3, now decodes `Ċ`).

## June-2026 scalar→ggml wave audit: 3 new regressions, all invisible to numerical guards (2026-07)

Systematic re-audit of the ~15-engine June scalar→`ggml_conv_2d` refactor wave
(handover `june20-ggml-refactor-regression-audit.md`), each engine on **both**
Metal and CPU. Result: the wave broke **well more than the 3 known** engines —
**three** *new* regressions (restormer, paddleocr-vl, qwen2vl-3b), all
both-backend, and every one slipped the automated guards. Only looking at the
actual output (rendered pixels / read transcript) caught them:

- **restormer — garbage output, both backends. [RESOLVED 2026-07 — see the
  dedicated section at the top of this file.]** `restormer-denoise-f16` on a
  clean image emitted blocky rainbow noise (mean 147 / std 120 vs a clean ~242),
  **bit-identical on Metal and CPU** → a both-backend `ggml_conv_2d` conversion
  bug (not the usual Metal-only failure mode). The garbage aligned with the 128px
  tiling. Correct call on the site (`rst_conv2d_ggml` weight-reshape heuristic),
  wrong scope: the reshape was scrambled for **every** conv (not a
  transposed-conv edge case), so `RESTORMER_SCALAR=1` was *not* a clean reference
  either — the U-Net convs run through ggml in both modes. Also a second bug in
  the ggml MDTA block graph (fake single-head attention, no temperature). Fixed.
- **paddleocr-vl — SIGSEGV, both backends.** `paddleocr-vl-0.9b` loads via the
  shared **`qwen2vl_ocr`** engine (vision 27L merge=2, llm 18L, **16/2 GQA =
  8:1**) then crashes `EXC_BAD_ACCESS` in `_platform_memmove` during forward —
  exit 139, zero output, reproducible in isolation on Metal AND CPU. The 8:1
  head/kv ratio is exactly the "native-GQA broadcast wrong for specific ratios"
  hazard `fbae7ba` introduced. Bad memmove ⇒ a tensor-size mismatch overrun.
  **qwen2vl-3b (the primary user of this shared engine) does NOT crash** → the
  SIGSEGV is a PaddleOCR-VL-specific branch, not the common path.
- **qwen2vl-3b — hallucinated OCR, both backends.** `qwen2.5-vl-3b-q4_k` on the
  clean fox image outputs a fabricated description ("mathematical symbols and
  equations, Greek letters α/β/γ, summations, derivatives or integrals, the text
  is distorted…") instead of reading "The quick brown fox…". **Identical on Metal
  and CPU** → both-backend engine bug; the garbled-vision → LLM-hallucination
  signature matches the `got_ocr` neck-permute regression. got/internvl2/lightonocr
  read the same image correctly through the *same* `--ocr` path, so it's the
  qwen2vl vision pathway, not the harness/prompt. `patch_embed` is stored 2D here
  so `5c8cb1b`'s 4D-flatten branch does not fire (ruled out); exact site needs a
  per-stage ref (none on HF — generate via `tools/dump_qwen2vl_reference.py`).
  `expected_text` was `null` (never baked), so this may also be a never-validated
  path; either way qwen2vl-3b OCR is currently broken. Not yet tested at q8_0.

### qwen2vl-3b hallucinated OCR — RESOLVED (2026-07-02): never-worked path, 4 bugs, NOT the ggml wave

The `null` `expected_text` was the tell: qwen2.5-vl-3b OCR **never worked** —
this was a never-validated path, not a scalar→ggml-wave regression (the wave
theory above was wrong). Four independent Qwen2.5-VL-specific bugs, all fixed in
`src/qwen2vl_ocr.cpp`; after the fix both Metal and CPU read
`The quick brown fox jumps over the lazy dog. 12345` (cer≈0) at q4_k.

1. **Vision RoPE built in raster order, but patches are merge-block ordered.**
   The preprocessor (`image_preprocess.cpp`) *always* emits patches in
   `(h//m,w//m,m,m)` merge-block order, and HF's `rot_pos_emb` applies the same
   permutation to the position ids. `compute_vision_rope`'s `merge_order` arg was
   `is_qwen2_vl` → **false** for Qwen2.5-VL (RMSNorm variant) → rope in raster
   order → every patch rotated with a neighbour's position → scrambled spatial
   structure. This is the dominant bug (fixing it alone flips pure hallucination
   into reading real words). Fix: `merge_order = true` (unconditional — see the
   gate correction below).
2. **Merger grouped patches by the wrong branch.** The CPU spatial merge keyed
   the consecutive-vs-raster grouping off `is_qwen2_vl`, sending Qwen2.5-VL
   through a raster gather (`normed_data[row*w_p+col]`) that mis-groups
   merge-block-ordered data (the deepstack path already assumed consecutive —
   the tell). Fix: unconditional consecutive grouping.
3. **Windowed attention was entirely unimplemented.** Qwen2.5-VL's ViT does
   window attention on all but `fullatt_block_indexes` ([7,15,23,31]); the loaded
   `window_size`/`fullatt` fields were never used — every block did full
   attention. Implemented as an equivalent **in-place additive mask** (0 within a
   window, -inf across) applied via `soft_max_ext` on windowed blocks — no
   physical reorder/reverse-permute needed, because window attention only
   restricts the *set* a patch attends to, which is storage-order-independent.
   Full-attn blocks keep the fused `flash_attn_ext`. Opt-out: `QWEN2VL_OCR_NO_WINDOW=1`.
4. **No OCR prompt for arch `qwen2vl`.** Only `qwen3vl` got the transcription
   prompt; `qwen2vl` fell back to `"Describe this image."` → verbose prose
   (fails a bare-text CER match). Fix: apply the OCR prompt to both archs.

Per-stage HF ref (`tools/dump_qwen2vl_reference.py`) not regenerated here —
end-to-end transcript on both backends was the verification.

**Gate correction — the `deepstack_indexes.empty()` gate regressed Qwen3-VL
(caught + fixed 2026-07-02, same day).** The first fix (commit `86d0830`) gated
rope order and merger grouping on `deepstack_indexes.empty()`, on the wrong
assumption that Qwen3-VL is `is_qwen2_vl=false`. It is **`is_qwen2_vl=true`** (its
ViT uses LayerNorm) *and* it has deepstack — so it was already reading the
*correct* merge-block/consecutive path via the old `is_qwen2_vl` gate, and the
new gate flipped it to raster → **garbage OCR** (`qwen3-vl-2b-q4_k` on fox.png
emitted `T11123456789…`). Root truth: `patchify_qwen_layout` emits merge-block
order for **every** Qwen2VL-family model, so rope order and merger grouping are
**unconditionally** merge-block/consecutive — no gate. Final fix makes both
unconditional; verified `qwen3-vl-2b` AND `qwen2.5-vl-3b` read the fox line on
CPU and Metal. Lesson: `is_qwen2_vl` is a *ViT-norm* flag (LayerNorm vs RMSNorm),
NOT a family selector — Qwen3-VL trips it true; don't repurpose it (or a proxy
like deepstack) for preprocessing-order decisions. The window-attention gate
(`!is_qwen2_vl && …`) is unaffected: Qwen3-VL correctly stays full-attention.

**Methodology lesson — a "non-degenerate output" numeric guard is not an output
check.** restormer's noise has high std, so a std>1 guard *passes* it; the
garbage was only visible by rendering the pixels. Likewise paddleocr's crash is
invisible to a per-stage cos diff (no reference was ever uploaded, and the engine
dies before producing one). The regression suite's exit-code + garbage-guard is
necessary but not sufficient for the SR/restoration engines — they need a golden
image + PSNR/SSIM (or a per-stage ref with the *output* tensor), and the VLMs
need a real transcript baked (not `expected_text: null`).

**Manifest ref bugs found while (re-)verifying the "no ref on HF" claim** — the
refs mostly *do* exist, the manifest just pointed at wrong names/repos:
`glm-ocr-ref.gguf`→`glm-ocr-ref-full.gguf`; qwen2vl ref repo
`cstr/qwen2vl-3b-crispembed-GGUF` (nonexistent)→`cstr/qwen2.5-vl-3b-crispembed-GGUF`;
paddleocr ref repo `cstr/paddleocr-vl-GGUF`→`cstr/paddleocr-vl-0.9b-GGUF`. Fixed
in `tests/regression/manifest.json`. (glm's ref exists and enables its per-stage
diff; qwen2vl/internvl2/paddleocr per-stage refs still need generating from the
VPS torch dumpers.)

**`fbae7ba` correction:** it removed GQA head-expansion in **lightonocr / got_ocr /
glm_ocr**, NOT qwen2vl (handover table was wrong). got_ocr was a no-op (MHA,
nh==nkv); lightonocr + glm now rely on native GQA broadcast. Verified clean:
lightonocr decodes correctly both backends (minor cosmetic `ĠĊ` BPE-marker leak).

**Verified CLEAN, both backends (no regression):** SR per-stage diff —
swinir, dat (scalar + `DAT_SR_GGML_CONV`), hat, pan, tbsrn, adair, scunet,
instructir; output+Metal==CPU — safmn, esrgan; OCR — got-ocr2 (full 20-stage
diff, cer 0.000), internvl2, lightonocr.
**nafnet_denoise + restormer = the two conv→ggml regressions still open after the
first audit; both RESOLVED 2026-07-02** (see the dedicated sections at the top of
this file). nafnet was the coverage gap (no diff harness, no standalone CLI output
— only reachable via `--denoise`); it had the same scrambled-kernel bug **plus** a
Metal/CUDA residency abort. restormer's earlier "CPU==Metal fixed" claim was wrong:
its **layout** fix was real, but a **second, independent residency bug** (weights
loaded on the freed init_best backend, referenced from the CPU conv sched) meant it
still *aborted on Metal* and segfaulted on CUDA — it only ever passed under
`RESTORMER_FORCE_CPU=1` / CPU-only builds, which is how the audit missed it. Both
now verified Metal==CPU (nafnet output cos 0.999998, restormer 0.999997).
**granite-vision — NOT a wave regression, but OCR broken by a packaging bug.**
Per-stage diff vs `granite-vision-ref.gguf` is **healthy and identical on both
backends**: `vis_patch_embed` cos 1.000 → gradual accumulation → `vis_layer_26`
0.958 / `projector` 0.956 (max_abs ~2.7–4.3). That's expected q8_0-vs-f32 drift —
all within granite's calibrated manifest threshold **0.95** (only the harness's
strict 0.99 default flags them; the decay is gradual, not a crater, so no scramble
regression). BUT end-to-end OCR degenerates to raw token IDs `<322><322>…` because
the gguf ships **`tokenizer=MISSING (0 tokens)`** → prompt encode + output decode
both fail. That's a **converter/packaging bug** (re-embed the tokenizer), separate
from the ggml wave; `expected_text` was `null` so granite OCR was likely never
validated end-to-end.

**RESOLVED (2026-07-02).** The tokenizer is now folded into
`models/convert-granite-vision-to-gguf.py` (new `array<string>` KV writer +
`load_tokenizer()`, writing `tokenizer.tokens`/`tokenizer.merges` +
`attention_multiplier`/`rms_eps`/`bos`/`eos`), so a fresh convert produces a
complete gguf and the separate `patch-granite-gguf-tokenizer.py` step can't be
forgotten (that patch script is now idempotent — it skips the KVs it re-adds).
The three published GGUFs (q4_k/q8_0/f16) in `cstr/granite-vision-crispembed-GGUF`
were re-patched + re-uploaded (Xet deduped each to ~84 MB of genuinely new data —
the tokenizer strings). Load banner now reads `tokenizer=embedded (49156 tokens)`
and `--ocr fox.png` returns readable text on **both CPU and Metal**: q4_k reads
`The quick brown fox jumps over the lazy dog, 12345` and q8_0 is an exact match
`… dog. 12345`. The regression manifest now bakes that `expected_text` (max_cer
0.15). Validation detail: token count 49156 == `vocab_size`, `token[49155]` ==
`<image>` (the runtime's `image_token_index`), no vocab gaps.

**Fix notes.** restormer's **layout** is DONE (see the RESOLVED section at the top):
the "3-site weight-layout unification" framing was a red herring — the converter
stores conv weights raw as numpy `(OC,IC,KH,KW)`, so the correct kernel is a
*plain* reshape of the contiguous bytes to ggml `[KW,KH,IC,OC]` at each site (no
permute/transpose), and the load-time pre-permute was deleted outright. paddleocr
+ qwen2vl are also fixed (see their own RESOLVED sections / HISTORY). The
`ne=[432,3]` abort was self-inflicted by a half-applied fix, not a real second
weight bug; the real second bug was the ggml MDTA block graph.
**Correction (2026-07-02):** restormer had a *third* bug the layout work never
touched — it loaded weights on `ggml_backend_init_best()` (Metal/CUDA) but runs
every graph on a CPU `enc_sched`, so referencing those GPU-buffer leaves aborted on
Metal (`pre-allocated tensor (patch_embed.weight) in a buffer (MTL0) that cannot
run`) and segfaulted on CUDA. Fixed by loading restormer's weights on CPU (the
convs run on CPU regardless of where the weights live). Now passes on Metal in both
default and `RESTORMER_SCALAR=1` paths.

**HF download gotcha (this box, 2026-07):** the `huggingface_hub` client wedged on
the Xet CDN (`cas-bridge.xethub.hf.co`) with 10s read-timeouts even with
`HF_HUB_DISABLE_XET=1`, while the HF *API* stayed fast. Plain
`curl -L https://huggingface.co/<repo>/resolve/main/<file>` (AWS CDN redirect,
`--retry 6`) is the reliable fallback for the diff/regression harnesses.

## InternVL2: caching a ggml graph on a SHARED scheduler is unsafe; "re-alloc" fixes it on CPU but not Metal (2026-07)

`c714758` ("cache vision encoder graph across tile invocations") built the
InternViT graph once and reused the `ggml_context` + `ggml_cgraph` + input/output
tensors for every dynamic tile, only re-uploading input data. This crashed:
InternVL2 dynamic-tiles an image into 1–5 tiles, and within each tile
`encode_vision_tile` and `project_vision` share the **same** `ctx.sched`
(`ggml_backend_sched`). `project_vision` calls `ggml_backend_sched_reset` between
tiles, which invalidates the vision graph's allocation. From the 2nd tile on,
vision compute read freed/realloc'd scheduler buffers → `EXC_BAD_ACCESS`
(SIGSEGV/SIGBUS) on Metal, and a non-terminating decode (87-min "hang") on CUDA.
Bisected: the parent `f0df10c` OCRs the exact 5-tile page correctly.

**The subtle part — the fix that only works on one backend.** A first, tempting
fix (shipped in parallel as `f0ddbf9`) keeps the built graph cached but calls
`ggml_backend_sched_reset` + `ggml_backend_sched_alloc_graph` *before each tile*
to "re-establish" the allocation. That **works on the CPU backend but still
SIGSEGVs on Metal** (verified head-to-head on the same 5-tile page, M1): ggml
frees the graph's input-tensor memory after its first consumer, and re-allocating
the *same* cgraph object doesn't restore a valid input buffer on Metal's
residency-set/shared-buffer allocator, though CPU tolerates it. The only fix that
holds on **both** backends is a full **fresh build per tile**
(`build_vision_graph` → `sched_reset` → `alloc` → compute → `ggml_free`) — the
pre-`c714758` behaviour.

**Lessons:**
1. Don't cache a ggml graph (context/cgraph/tensors) that is computed on a
   scheduler shared with other graphs — the other graphs' `sched_reset`/`alloc`
   invalidate it. Cache the *weights*, not the graph.
2. A memory-lifetime fix that passes on one backend can still crash another —
   ggml's CPU and Metal allocators have different free/reuse semantics. Validate
   allocator-sensitive fixes on **every** backend you ship (the re-alloc fix was
   green on a CPU-only VPS and would have shipped the Metal crash).
3. The same reuse anti-pattern can surface on a *different* backend and **need
   not crash** — it can silently corrupt. `lfm2_embed`'s ColBERT path (2026-07,
   top-of-file entry) hit the reserve-variant: `encode_multivec` re-allocated the
   same `ggml_cgraph` it had passed to `ggml_backend_sched_reserve`. That was
   tolerated on CPU *and* Metal (colbert_output cos 0.998) yet **silently
   mis-computed on CUDA** (cos 0.571643, backbone `hidden` −0.702160 on a P100) —
   the mirror image of this InternVL2 case (CPU-OK, Metal-crash). So "passes on
   the backend I happened to test" proves nothing, and the failure may be a
   quiet numerical drift, not a SIGSEGV. A graph handed to
   `sched_reserve`/`sched_reset` is dead: rebuild a fresh one before the next
   `sched_alloc_graph`+compute (the dense `lfm2_embed_encode_to` already does),
   and A/B on the backend you actually ship to.

## GOT-OCR2: the "colorcolor…" garbage was a vision-neck permute, not quantization (2026-07)

The got-ocr2 garbage output (`colorcolorcolor…` repeated forever) and the
belief that "F16 decoder fixes it, quantized decoder breaks it" were **two
different bugs tangled by a confound**. The actual garbage root cause is a
single wrong axis-permutation in the vision neck's final flatten
(`src/got_ocr.cpp`, commit **7f43e4d**):

```
- x = ggml_cont(ng, ggml_permute(ng, x, 2, 0, 1, 3));  // (C,W,H) → produced (H,C,W)  ✗
+ x = ggml_cont(ng, ggml_permute(ng, x, 1, 2, 0, 3));  // (W,H,C) → (C,W,H)            ✓
  x = ggml_reshape_2d(ng, x, vis_D, n_vis_tokens);      // (1024, 256) = (channel, token)
```

The old `(2,0,1,3)` produced `(H,C,W)` instead of `(C,W,H)`, so the 256 vision
tokens handed to the projector were scrambled. The decoder then received
meaningless image embeddings and degenerated into a repeated token.

**Proven by bisection, not asserted.** Building the pre-fix commit (`ba74093`,
old permute) against the *current, known-good* q8_0 GGUF — GGUF held constant,
only the runtime code varied — reproduced the exact `colorcolor…` garbage on
the CPU backend. The same old code garbles **f16 too**, so the bug is
independent of decoder precision and of backend (Metal/CPU alike).

**The confound:** a colleague on a CPU-only VPS saw an F16-decoder GGUF work and
a quantized-decoder GGUF produce garbage, and concluded "the decoder must stay
F16." In reality the F16 build was run with newer (post-7f43e4d) code while the
quantized build was a stale download run with older code — a code change and a
GGUF swap were varied together. `--decoder-f16` never fixed anything; it only
correlated with the real fix. (This is the *same* wrong conclusion the
diff-harness artifact produced from the other direction — see next entry.)

**Lesson:** when two things change between a "broken" and a "working" run
(here: local code version *and* the GGUF), you have not isolated the cause —
hold one constant and vary the other. A backend-/precision-agnostic bug
(a graph-shape error) can masquerade as a precision-sensitivity bug when the
code version silently rides along with the weight file.

## GOT-OCR2: a wrong-axis parity cosine faked "decoder can't be quantized" (2026-07)

`tests/test_got_ocr_diff.cpp`'s `compare()` reduced the cosine over the wrong
tensor dimension — it used the **token count** (5) as the row length instead of
the **feature dim** (1024). On the got-ocr2 Qwen2-0.5B decoder this reported
`llm_layer_0` cos ≈ **0.936** for Q8_0 weights, which read as catastrophic
quantization sensitivity and drove a real, wasteful workaround: shipping an
F16-only decoder (`--decoder-f16`) at 1.03 GB, ~2× slower per-token than Q4_K.
The same wrong-axis harness had earlier produced a bogus "bf16 compute is the
bug" theory.

With the axis fixed (`row_dim=0`) the plain Q8_0 decoder matches the f32
reference at cos ≥ **0.99996** across all layers, and Q4_K / Q8_0 / F16 produce
byte-identical OCR. Two independent cross-checks would have caught the false
alarm immediately: (1) the decoder graph is functionally identical to
`internvl2_ocr`'s Qwen2-0.5B path, and (2) `internvl2-1b` already ships that
exact decoder at Q4_K. got-ocr2 now defaults to Q4_K (445 MB, ~20 ms/tok on M1).
Full writeup: [`docs/got-ocr2.md`](docs/got-ocr2.md).

**Lesson:** a parity harness is only as trustworthy as its reduction axis — a
cosine taken over the wrong stride can masquerade as model sensitivity and
justify a precision workaround that isn't needed. Before believing "this small
model can't be quantized," sanity-check against a sibling model that ships the
same architecture at that quant. (Separately, this investigation surfaced that
Q8_0 `mul_mv` is anomalously slow on M1 — see
[`docs/metal-q8_0-mul_mv-slow-m1.md`](docs/metal-q8_0-mul_mv-slow-m1.md).)

## DAT: Conv+BN fusion silently skipped on F32 models (to_f32 returns t->data) (2026-06)

`dat_sr.cpp` parity vs a *genuine* reference (the real PyTorch DAT-light run on
weights reconstructed from the f32 GGUF — `tools/dump_dat_reference_from_gguf.py`)
was only cos 0.9906 (vs the other SR engines' 0.999+). Root cause: the init-time
Conv+BN fusion dequantized weights with the file's `to_f32(t, buf)` helper, which
— like several `*_to_f32` helpers in this repo — **returns `t->data` directly for
GGML_TYPE_F32 tensors and leaves the passed `buf` empty** (only the quantized/F16
paths fill `buf`). The fusion code then read `cw = dequant(...)` and guarded with
`if (!cw.empty() && !bw.empty())`, so on an **F32 model** every conv's `cw` was
empty, fusion was skipped, and the BatchNorm in the AIM dwconv / channel- and
spatial-interaction branches was **dropped entirely** (there is no separate BN
application — fusion is the only BN path). On F16 models the bug is hidden because
`to_f32` fills `buf`. Fix: in the dequant lambda, when `to_f32` returns a pointer
≠ `buf.data()`, copy it in (`buf.assign(p, p + ggml_nelements(t))`). Output cos
0.9906 → 0.999995; all 20 captured stages ≥0.99998.

**Lesson:** when a `*_to_f32(t, buf)` helper has a fast F32 path that returns
`t->data` without touching `buf`, never use `buf.empty()`/`buf.size()` as a
proxy for "did I get the data" — use the returned pointer. This silently
miscompiles only on F32 (not F16/quant) models, so it survives F16-only testing.
Genuine ground truth (real model run) was needed to catch it: a self-consistent
ref reverse-engineered from the same engine would have hidden it.

## SwinIR shifted-window: shift sign must match the precomputed attn_mask (2026-06)

SwinIR's `swinir_sr.cpp` produced an output whose `test-swinir-diff` cosine read
−0.91 (looked anti-correlated). Two distinct issues were tangled together:

1. **The real bug — `cyclic_shift` sign convention.** The shifted (odd-index)
   Swin blocks roll the feature map by `ws/2`, partition into windows, and add a
   *precomputed* `attn_mask` (loaded from the GGUF) that blocks attention between
   the regions that wrap around at the bottom/right edges. That mask is built for
   `torch.roll(-ws/2)` (= numpy `np.roll(x, -ws/2)`, i.e. `out[y]=in[(y+ws/2)%H]`).
   The engine's `cyclic_shift(..., -ws/2)` computed `in[(y-ws/2)%H]` — the
   **opposite** roll. Forward and reverse shifts still cancelled (the round-trip
   was self-consistent), so the bug was invisible except where it interacts with
   the mask: the **edge / wrap-around windows** got the mask meant for the other
   convention, mixing token regions that should be blocked. Result: divergence
   localised at image edges in the shifted blocks, compounding through the four
   RSTBs (rstb_0 cos 0.9994 → rstb_3 max_abs 147, engine ≈ 2× ref at edges).
   **Fix:** forward shift `+ws/2`, reverse `−ws/2` so partition order and mask
   align. All stages then cos ≥ 0.99997, output (pre-clamp float) cos 0.999996.
   Lesson: when a precomputed mask encodes a geometric convention, the shift /
   partition / reverse must all match *that* convention end-to-end; a self-
   consistent-but-opposite round trip silently corrupts only the masked windows.

2. **A red-herring test metric.** `crispembed_diff::Ref::compare` reports
   `cos_min` = the worst per-row cosine, where row size = `shape.back()`. For the
   SR output stored CHW with gguf-reversed shape `[256,256,3]`, that "row" is **3
   horizontally-adjacent pixels in one channel**, and the C++ side is uint8-
   clamped to `[0,1]` while the ref is raw float (can be negative). A single
   near-zero edge triple where clamping disagrees in sign drives `cos_min` to
   −0.91 even when the image is essentially identical (global cos 0.999996). The
   previous session correctly flipped the shift, saw `cos_min` go −0.91 → −0.001,
   and wrongly concluded the flip "wasn't the fix." Always sanity-check a diff
   harness's reduction (worst-row vs global) before trusting a single scalar —
   especially on quantized/clamped outputs. `test_swinir_diff.cpp` now gates on
   the image-level (global + per-RGB-channel) cosine.

## Porting a scalar conv engine to ggml_conv_2d: reverse the kernel ne (2026-06)

When converting an SR/restoration engine from scalar conv nested loops to a
`ggml_conv_2d` graph (pan_sr was the latest), the recurring trap is the conv
kernel's `ne`. The Python converters write each PyTorch weight `[OC,IC,KH,KW]`
with a plain `astype` — no permute — so in the GGUF the *data* is KW-innermost
(row-major `[OC][IC][KH][KW]`, exactly what the scalar `weight[o*ic*kh*kw +
c*kh*kw + ky*kw + kx]` indexing assumes) but the stored `ne` is `[OC,IC,KH,KW]`.

`ggml_conv_2d(ctx, kernel, input, ...)` wants the kernel as `ne=[KW,KH,IC,OC]`
over that same byte layout. So at weight-prep time **reverse the four ne axes
and copy the raw dequantized buffer unchanged** — do not permute the data:

```cpp
// src ne = [OC,IC,KH,KW] (PyTorch order), data KW-innermost
int64_t ne[4] = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] }; // -> [KW,KH,IC,OC]
ggml_tensor * w = ggml_new_tensor(gw_ctx, GGML_TYPE_F32, 4, ne);
ggml_backend_tensor_set(w, dequantized_src, 0, ggml_nbytes(w)); // bytes as-is
```

Feeding the native ne instead aborts in `ggml_im2col` with `GGML_ASSERT(OW>0)`
because it reads `OC` as the kernel width (e.g. 40 > input W=32 → negative OW).
Do **not** use `ggml_n_dims` to detect conv kernels — a 1×1 weight `[OC,IC,1,1]`
reports 2 dims and would be left un-reversed. Key off the `.weight` name suffix
and always treat those as 4D. Biases (`.bias`, 1-D) are copied as-is.

Verification corollary: the diff harness feeds the C++ engine a uint8-quantized
input (`round(x*255)/255`), so the torch reference generator must snap its input
to the same 1/255 grid. Skip this and a ±1/255 input perturbation amplifies
through a 4× SR network to ~0.4 max-abs, dropping one image row to cos ≈0.9959
even though the math is correct. Matched, pan's graph and scalar both reach
cos_min 0.999997. See `tools/dump_pan_reference_from_gguf.py`.

## ggml_flash_attn_ext accepts non-contiguous Q/K/V (2026-06)

`ggml_flash_attn_ext` on both Metal and CUDA handles non-contiguous inputs
natively — it does NOT require `ggml_cont` wrappers around `ggml_permute` on
Q, K, or V before the flash attention call. Adding `ggml_cont` is redundant
and forces an extra allocation + copy at the Metal side. The correct pattern:

```cpp
Q = ggml_permute(gctx, Q, 0, 2, 1, 3);   // [hd, T, H, 1] — non-contiguous OK
K = ggml_permute(gctx, K, 0, 2, 1, 3);
V = ggml_permute(gctx, V, 0, 2, 1, 3);
ggml_tensor * attn = ggml_flash_attn_ext(gctx, Q, K, V, mask, scale, 0.f, 0.f);
```

Previously the code had `ggml_cont(ggml_permute(...))` — removed in
`decoder_embed.cpp` (`29d8a08`) and `bidirlm_vision.cpp` (`fd8cd09`).

Note: for KV-cache *outputs* that you `ggml_set_output` and read back via
`ggml_backend_tensor_get`, you DO still need `ggml_cont` (see the cross-
backend KV-cache audit entry below) — the non-contiguous exception only
applies to flash_attn *inputs*.

## SIMD dot_product as the universal optimization lever (2026-06)

A single `dot_product()` function with AVX2+FMA/NEON in `core/cpu_ops.h`
accelerated 30+ runtimes because every CPU-side matmul, attention dot
product, and LSTM gate computation reduces to the same inner loop:
`sum += a[i] * b[i]`. By making `linear_cpu` and `mha_1q_cpu` delegate
to `dot_product()`, all existing callers got SIMD for free without any
per-engine changes. Key pattern: dual-accumulator unroll (16-wide) with
single-accumulator cleanup (8-wide) plus scalar tail — handles any
dimension without alignment requirements.

## DequantCache: init-time vs per-call weight dequantization (2026-06)

The biggest single performance win across the codebase was caching
dequantized weights. The anti-pattern: `to_f32(tensor)` returns a new
`std::vector<float>` each call, so a 30-layer decoder running 512 steps
re-dequantizes the same immutable weights `30 * 4 * 512 = 61K` times.
The fix: `DequantCache` struct with `unordered_map<void*, vector<float>>`
keyed on `tensor->data`. First call dequantizes; subsequent calls return
the cached pointer. Thread-safety: one cache per inference context (not
global static — that was the math_ocr bug).

## Thread-local buffers eliminate per-call allocations in shared primitives (2026-06)

Hot-path functions like `mha_1q_cpu`, `swiglu_ffn`, and `layernorm2d_cpu` were
allocating `std::vector<float>` on every call (per head, per position, per step).
`thread_local std::vector<float>` with lazy resize eliminates all heap allocation
after the first call while remaining thread-safe. Pattern: check `size() < N`,
resize if needed, use `.data()`. The buffer persists across calls on the same
thread. For callers with pre-existing scratch buffers, offer an optional
`float* scores_buf = nullptr` parameter so they can pass their own.

## BatchNorm fusion into conv weights at load time (2026-06)

For Conv→BN sequences in eval mode, BN can be algebraically folded into the
preceding conv's weights: `new_W[o] = bn_scale[o] * conv_W[o]`,
`new_b[o] = bn_scale[o] * conv_b[o] + bn_shift[o]`, where
`bn_scale = bn_weight / sqrt(bn_var + eps)` and
`bn_shift = bn_bias - bn_mean * bn_scale`. This eliminates the BN pass
entirely at runtime. Applied to TBSRN (11 conv+BN pairs) and DAT SR (54
conv+BN pairs across 3 patterns: dwconv, channel_interaction, spatial_interaction).
Key detail: the
fused weights go into a separate `fused` map checked first by `get()` — if
found, returns the fused version; otherwise falls through to DequantCache.
Output differs by ±1-2 pixel values due to float associativity
(`scale * (W*x)` ≠ `(scale*W) * x`), which is expected and acceptable.

## ggml_backend_sched vs ggml_gallocr for repeated inference (2026-06)

`ggml_gallocr` checks whether reallocation is needed on every `alloc_graph`
call (walks the graph to compare sizes). `ggml_backend_sched` with
`sched_reserve` pre-allocates for a bucket size, then `sched_alloc_graph`
just assigns pointers — much faster for repeated same-shape calls. The
T-bucketing pattern (8/16/32/64/128/256/512) reduces re-reserves when input
lengths vary slightly. For LFM2 (350M, 16 layers), graph+alloc overhead
dropped from ~2ms to ~0.7ms per call, but compute at ~700ms dominates —
the win is primarily architectural (aligns with the encoder path, enables
future GPU dispatch via the scheduler's multi-backend support).

## BPE merge: linked list + priority queue for O(N log N) (2026-06)

The standard BPE merge loop finds the lowest-rank adjacent pair, merges it,
and repeats. Naive implementation: O(N) scan per iteration × O(N) erase = O(N²).
Fix: doubly-linked list of symbol nodes + min-heap of `(rank, left_node_id)`.
Pop the best pair, merge left←right, requeue affected neighbors. Stale entries
(where the pair text no longer matches the rank) are filtered on pop by
re-checking the merge table. Total: O(N log N). Applied to both `bpe.h`
(used by lightonocr, granite_speech) and `tokenizer_bpe.cpp` (all BPE models).

## T5 attention uses scale=1.0 (no 1/sqrt(d)) (2026-06)

Unlike standard Transformer attention where `scores = QK^T / sqrt(d_k)`,
T5 uses raw dot products with scale=1.0. When using `ggml_flash_attn_ext`,
pass `scale=1.0f` explicitly. The relative position bias is added as a mask
tensor. In the pix2struct encoder, there's no relative bias (positions come
from row/col embeddings) — pass `nullptr` mask + `scale=1.0f`.

## Hann-window tiling for SR/restoration models (2026-06)

All SR/denoise models can process arbitrary image sizes via overlapping
tiles with raised-cosine blending. Key formula for the blend weight at
each pixel: `w = hann_y * hann_x` where `hann(p) = 0.5 - 0.5*cos(pi*p/overlap)`
in the overlap zone, 1.0 elsewhere. Accumulate weighted outputs + weight
map, divide at the end. Tile size must be aligned to the model's
downsample factor (e.g. 16 for a 4-stage U-Net with 2x per stage).

## Per-engine confidence: softmax without modifying logits (2026-06)

When adding per-token confidence to all 15 OCR engines, the key insight:
compute `exp(logits[best] - max_logit) / sum_exp` in a separate block
WITHOUT overwriting the logits array. Many engines have debug prints
referencing raw logit values after the argmax. Overwriting logits for
softmax breaks those prints. The one-pass approach
`conf = 1.0 / sum_exp` (since `exp(best - max) = exp(0) = 1` when best
IS the max) is both faster and non-destructive.

## llama.cpp GGUF ≠ CrispEmbed GGUF (2026-06)

llama.cpp splits VLMs into LLM + mmproj GGUFs with different tensor
names (`blk.N.attn_q` vs our `llm.layers.N.attn.q`, `v.blk.N` vs
`vis.blocks.N`). When a model is only available as llama.cpp GGUFs
(e.g. german-ocr-3.1), use `merge-llamacpp-qwen2vl-gguf.py` to:
1. Read both GGUFs (no gguf pip dependency — standalone reader)
2. Map tensor names via regex
3. Merge metadata from both sources
4. Write single combined GGUF with our naming convention

## Qwen2-VL engine handles 4+ architectures (2026-06)

The `qwen2vl_ocr` engine reads all hyperparams from GGUF metadata with
prefix probing for `qwen2vl.*` and `qwen3vl.*`. This
means it handles:
- Qwen2-VL (original)
- Qwen2.5-VL (updated)
- Qwen3-VL / FireRed-OCR (new attention patterns)
- Nanonets-OCR2 (pruned Qwen2-VL, 16L vs 28L)

Key: never hardcode layer counts or dimensions — always read from GGUF.

## Reference GGUF consistency trap (2026-06)

MixTex showed cos=-1.0 on decoder parity, causing 3 hours of debugging.
Root cause: the reference dumper captured encoder stages from a
preprocessed synthetic image, but decoder stages from `model.generate()`
which used ViTImageProcessor preprocessing — different encoder outputs
feeding the same decoder reference. Fix: always run encoder+decoder
from the SAME input in the reference dumper, or use the reference's
own `enc_layernorm` output to drive the Python decoder comparison.

## ggml_add doesn't support mixed types (2026-06)

`ggml_add(f32, f16)` crashes with `binary_op: unsupported types`.
`ggml_cast` in the graph SHOULD convert F16→F32, but some code paths
read model weights directly via `ggml_backend_tensor_get` assuming F32
layout — this reads wrong data sizes from F16 tensors (`tensor read out
of bounds`). Fix: use `tensor_to_f32()` helper that reads raw bytes via
`ggml_nbytes()` then dequantizes via `ggml_get_type_traits()->to_float`.

**Quantized version (2026-07-03): the same rule bites `position_embd` on
quant.** clip_text/SigLIP-text quantized cleanly on paper (75 tensors, imatrix
fired) but produced `cos_vs_f32 = 0.0000` and aborted at inference:
`binary_op: unsupported types: dst f32, src0 f32, src1 q8_0`. The quantizer
stores `position_embd.weight` (a 2-D embedding table) as Q8_0, and the graph
added it via a raw `ggml_view_2d(pos_embd)` — `ggml_add`'s src1 must be F32, and
a Q8_0 view is not. The token embedding next to it was *fine* because it goes
through `ggml_get_rows`, which dequantizes to F32; only the position path used a
raw view. Fix: `if (ggml_is_quantized(pe->type)) pe = ggml_cast(g, pe,
GGML_TYPE_F32);` before the view+add. **Lessons:** (1) an embedding table added
(not matmul'd) must reach the binary op as F32 — either `get_rows` it (LiLT does
this for all of pos/x/y/w/h/type and never hits the bug) or `ggml_cast` it; a
raw `view_2d` of a quantized weight is a latent crash that only appears once the
model is quantized. (2) "the collector fired and quantize succeeded" does NOT
mean the quantized model runs — always run inference on the quant and check
cos-vs-f32, per CLAUDE.md "build verifies compile, not correctness."

## Quant/imatrix A/B needs a CONTINUOUS metric, not a thresholded one (2026-07)

Evaluating imatrix on a classification model (punctuation, NER, KIE) with a
**thresholded** metric — restored-string exact-match, or per-token argmax-label
agreement — is blind to it. The argmax saturates to "perfect" long before the
model is lossless: fireredpunc scored 5/5 restored-string match for *both* plain
q4_k and q4_k+imatrix, so imatrix looked worthless (n=5 → "no value").

imatrix acts on the **logits / probability distribution**, not the argmax. Dump
the pre-argmax per-token class logits (an env hook like `FIREREDPUNC_DUMP_LOGITS`)
and, over HUNDREDS of tokens, compute **mean per-token prob-cosine** (softmax vs
gold softmax, →1) and **mean KL(gold‖quant)** (→0). Over 490 tokens that showed
q4_k+imatrix cutting KL-from-f16 ~2.8× (0.0093→0.0033) — a real, monotone win the
exact-match hid. Report those, not exact-match. (Embedders already use cosine, a
continuous metric — this gap only bit the discrete-output models.)

Two corollaries burned real time here:
1. **Never A/B a gguf that is still being quantized.** A half-written iq4_xs read
   as 0/5 exact-match ("iq4_xs breaks punct"); the completed file is argmax-perfect.
   Gate the eval on a DONE sentinel + a file-size-stable check.
2. **Measure against the highest-precision gold you actually have, and say which.**
   fullstop-punc has no f16 base on HF (only q8_0), so its imatrix could only be
   calibrated+quantized from q8_0 and measured vs q8_0 — a near-lossless gap
   (KL 0.0012) where imatrix genuinely can't help. That's a real "no benefit", but
   for a different reason than fireredpunc's "exact-match can't see it"; don't
   conflate the two.

## DeepSeek-OCR-2: from a never-run port to character-perfect OCR (2026-06)

> **Status: WORKING again after a perf-sweep regression (fixed 2026-07-02).**
> The Jun-20 "perf sweep" silently regressed OCR to garbage on BOTH backends —
> see "Perf-sweep regression" below. Fixed by restoring `deepseek_ocr2.cpp` to
> the last-known-good-AND-fast commit `c58913c` (Metal vision graphs, correct
> output), reverting the post-`c58913c` perf commits. Character-perfect OCR on
> Metal + q4_k confirmed on the recovered Jun-19 q4_k. The 2026-06 history below
> is preserved for its instructive failure modes.

### Perf-sweep regression — garbled OCR, both backends, no A/B gate (found + fixed 2026-07-02)

`qwen2.5`-style symptom (garbled multilingual tokens `章的 flix Bailly …` / decode
repetition `&# &#`) on **both** Metal and CPU (deterministic), on the recovered
character-perfect Jun-19 q4_k. Not the q4_k (the exact character-perfect model
garbled), not Metal (CPU byte-identical), not the engine *file* per se — ggml SHA
and the converter were unchanged. **Bisect (`git bisect ... -- src/deepseek_ocr2.cpp`,
range `38e3801..e803e9f`) pinned it to the Jun-20 perf sweep**, which introduced
MULTIPLE regressions with **no env gate and no A/B test**:
- `c75b95d` "flash_attn_ext + remove GQA repeat in Qwen2 encoder": replaced the
  encoder's manual masked GQA attention with `ggml_flash_attn_ext` but kept the
  manual path's trailing `ggml_permute(attn, 0,2,1,3)`. **`ggml_flash_attn_ext`
  already returns `[hd, nh, T]` (permuted internally), so that trailing permute
  is spurious and scrambles the features** (see memory [[flashattn-ext-already-permutes]];
  the Jun-2026 flash wave left the same spurious permute in layout/math/deepseek).
  The precise perf re-add fix is therefore a one-liner: keep flash_attn, drop the
  trailing permute, reshape `[hd,nh,T]→[D,T]` directly — but verify per the A/B
  rule before flipping the default. (Reverting *only* this wasn't enough —
  `910d036` had also rebuilt the encoder as a single graph, and even the
  `DS_QWEN2_SCALAR` fallback garbled on the post-sweep tree, i.e. the regression
  was spread across several commits.)
- decode degeneration (repetition) appeared between the good `c58913c` and
  `402b38d`/`e65e73c` (flash_attn LLM / persistent decode) — HF `infer()` uses
  `no_repeat_ngram_size=20`; the greedy decoder had no repetition blocking.

Last-fully-good commit = **`c58913c`** (Jun-19): after the Metal vision-graph
speedups (~15×) yet before the regressions; reads a document verbatim on Metal.
**Fix = restore `deepseek_ocr2.cpp` to `c58913c`.** The reverted perf paths
(persistent decode, F16 KV, flash_attn encoder/LLM, single-graph encoder) must be
re-added ONE AT A TIME behind an env gate, each A/B-tested against decoded output
before flipping the default — see the new rule in `crispasr-crispembed-dev.md`
("A/B-test every perf optimization"). deepseek had **zero** regression coverage,
which is why a broken default shipped unseen; add a golden entry when a stable
test image is chosen (fox.png's 800×200 strip is a weak-signal global-view case).

`deepseek_ocr2.cpp` + `convert-deepseek-ocr2-to-gguf.py` were committed in a
single feat commit and **never ran end-to-end** — the published GGUF
(`cstr/deepseek-ocr2-crispembed-GGUF`) will not even load. Diagnosed via the
HF blueprints (`deepencoderv2.py`, `modeling_deepseekocr2.py`,
`modeling_deepseekv2.py`) + a metadata dump:

1. **Converter is a stub — no tensor renaming.** It does
   `for name in header: writer.add_tensor(name, ...)`, emitting raw HF names
   (`model.sam_model.*`, `model.qwen2_model.model.model.layers.N.*`,
   `model.layers.N.*`). The engine loads short names (`v.*`, `qe.*`, `l.*`), so
   it finds *zero* tensors. An audit of all converters shows this is the only
   complex-VLM converter with no rename map (lightonocr=20, pix2struct=19,
   firered=17, layout=14, decoder-embed=10, qwen2vl/internvl2/glm/got=6-8). The
   renames=0 outliers otherwise are simple SR/restoration nets (esrgan, swinir,
   scunet, …) whose engines match the raw names — those are fine.
2. **merges written as array-of-arrays.** tokenizer.json stores merges as
   `[a, b]` pairs; the converter wrote them as a GGUF nested array (elemtype 9),
   which ggml rejects ("invalid GGUF type 9"). Must flatten to `"a b"` strings
   (same fix as Qari). FIXED in the converter.
3. **Tensor names exceed GGML_MAX_NAME (64).** Because of (1), names like
   `model.qwen2_model.model.model.layers.N.post_attention_layernorm.weight` (70
   chars) blow the ggml limit — a free consequence of not renaming.

Engine bugs found by blueprint comparison (use_mla=False → standard
`LlamaAttention`, so an agent's MLA RoPE/scale findings were false positives):
- **MoE gate**: config `norm_topk_prob=False`, `routed_scaling_factor=1.0` →
  use the raw top-k softmax probs; the engine renormalized them. FIXED.
- **Qwen2 vision encoder** (`CustomQwen2`): blueprint concatenates
  `[visual, queries]`, applies a token-type mask (visual↔visual bidirectional;
  queries→all-visual + causal-among-queries), and returns `y[:, n_query:]`. Our
  engine has the order reversed, is fully bidirectional, and returns the first
  half. NOT fixed (needs a loadable GGUF to verify).

**Progress (2026-06):** the converter now has the full HF→engine rename map and
deepseek_ocr2 is wired into the `--ocr` arch dispatcher, so the model **loads
(2707 tensors) and runs through the SAM stack without crashing**. Two crashes
fixed along the way: (a) the SAM downsample `net_3` outputs 896 channels (the
Qwen2 dim), not the config's nominal `downsample_channels` 1024 — derive the
channel counts from the weight `ne[1]`, not a hardcoded 1024 (was an OOB read);
(b) the LLM rmsnorm multiplied an f32 activation by the f16 norm weight, which
ggml's elementwise ops reject — `ensure_f32` the weight.

The Qwen2 vision encoder forward had 5 bugs (all confirmed vs `deepencoderv2.py`
`CustomQwen2Decoder`, which subclasses `Qwen2Model`, `rope_theta=1e6`):
(1) concat `[visual, queries]`, not `[queries, visual]`; (2) a token-type
attention mask (visual↔visual bidirectional; queries→all-visual + causal-among-
queries) — the engine was fully bidirectional; (3) **RoPE is applied**
(positions 0..T-1) — the engine omitted it; (4) return `y[:, n_vis:]` (the query
half), not the first half; (5) apply the final `qe.output_norm`. All five FIXED.

**RESOLVED (2026-06): character-perfect OCR on Metal + q4_k.** The key
unlock was the user's insight — quantize to q4_k and run on the Metal GPU
(`cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`) instead of
fighting CPU speed. With Metal active (~20 s prefill, MoE expert dispatch
parallelized across `n_threads` with `std::thread`), the pipeline became fast
enough to iterate. The remaining bugs, in the order they unblocked the output:

1. **KV-cache axis scramble.** K/V were `permute(0,2,1,3)`'d before flattening
   to `[nkv*hd, T]`, transposing the token and head axes vs the reload's
   `reshape_3d(hd, nkv, n_past)`. Flatten with a plain `cont()+reshape`, no
   permute — exactly the verified qwen2vl_ocr path. This alone fixed the first
   generated token.
2. **KV-cache buffer aliasing** (the "first token right, rest garbage"
   signature). `k_out`/`v_out` were views sharing the *same* `cont(K/V)` buffer
   the attention path consumes; under the no-alloc scheduler that buffer is
   recycled once attention reads it, so prefill computed the right first token
   but **cached garbage**. Give the cache outputs their own `cont` and
   `ggml_build_forward_expand` them (they are not ancestors of `layer_output`).
   Isolation that nailed this: `DS_NO_KV` (recompute the full sequence each
   step) produced " Paris." for "The capital of France is" while the cached
   path produced "Paris vro vro…".
3. **Prompt construction.** The decoder was fed 256 placeholder tokens with no
   bos, no view-separator and no instruction. The HF `infer` + plain template
   builds `[bos] + <image>*256 + <view_sep> + tokenize("Free OCR.")`; the 257
   image/sep slots are masked-scatter-replaced by `[global_features(256),
   view_seperator(1)]`. Assemble that as an embedding matrix directly (text
   slots from `embed_tokens`, image slots from the projector, separator from the
   learned `v.view_separator`). `image_token_id` is **128815**, not the Qwen
   `151643` the old heuristic assumed; eos is **1** (`<｜end▁of▁sentence｜>`).
4. **Byte-level BPE I/O.** Added a `core_bpe` encoder (merges loaded from the
   GGUF) to tokenize the instruction, and the **inverse** GPT-2 byte map on
   decode so pieces render as text (`Ġ`→space) instead of raw byte-unicode.
5. **Image preprocessing.** DeepSeek-OCR2 uses `mean=std=0.5` ([-1,1]), **not**
   CLIP normalization, and `ImageOps.pad` (aspect-preserving resize + gray
   border), not a stretch resize. With CLIP mean/std the model hallucinated a
   markdown table; with the correct preprocessing it reads the page verbatim.

The diagnostics added for this (`DS_DBG`, `DS_NO_KV`, `DS_TEXT_TEST`) are
env-gated and left in. The `crispembed-quantize` tool keeps the MoE router
(`*.mlp_gate.weight` / `ffn_gate_inp`) and the `qe.*` Qwen2 encoder at Q8_0,
which is what makes q4_k safe for this MoE.

### Speed (~15×) + portability + a numerics red herring (2026-06)

Three findings from making it fast and verifying it on macOS/Metal:

1. **The qwen2 encoder was the one stage NOT on the GPU** — `encode_qwen2` was
   pure CPU-scalar (naive O(T²) attention + per-token `linear_cpu`/`swiglu_cpu`),
   ~9 min of vision on an M1 while SAM and the decoder were already ggml graphs.
   Ported it to `build_qwen2_enc_layer_graph` (one graph over all `T=n_vis+n_query`
   tokens, no KV cache), reusing the in-file `build_llm_layer_attn` pattern: NEOX
   `ggml_rope_ext`, GQA interleave, `ggml_soft_max_ext` + a precomputed F16
   bidirectional/query mask. **Vision (SAM+enc+proj) ~9 min → ~37 s.** Verified
   bit-equivalent to the scalar path (per-layer `cos_min` matched to 5 digits);
   `DS_QWEN2_SCALAR=1` keeps the scalar reference for A/B.

2. **Decoder use-after-free, masked by the slow encoder.** `build_llm_layer_attn`
   built its graph in a **local** `std::vector meta` buffer and *returned the
   graph* — the buffer freed on return, leaving `gf` dangling. Latent UB: fine on
   Linux (freed heap intact), `EXC_BAD_ACCESS` in `ggml_backend_sched_alloc_graph`
   on macOS. It never surfaced before because every run died/timed-out in the
   9-min scalar encoder, *before* reaching decode-step-0. The fast encoder exposed
   it immediately. Fix: own the meta buffer **in** the returned struct (move
   preserves `data()`), matching the SAM pattern (caller-scoped buffer). With this
   the full OCR runs end-to-end in ~2 min and is character-perfect.

3. **The encoder's parity "failure" is a metric artifact, not a bug.** Against an
   fp32 PyTorch reference the encoder output looks broken (`cos_min`→−0.07,
   `cos_mean`~0.5 over 24 layers). But an **independent naive-fp32 NumPy
   reimplementation diverges identically** (cos_mean 0.57 vs the C++ 0.50) — so
   the gap is inherent fp32-vs-PyTorch-SDPA sensitivity on this model's
   **attention-sink massive activations** (token 0, channel 570, growing to ~410),
   not a code error. Confirmed not quantization: q8_0-roundtripping the weights in
   PyTorch keeps cos_mean 0.9995. `cos_min` is misleading here (dominated by the
   one massive channel) — judge the encoder by `cos_mean`, or just by the OCR
   text, which is correct. Lesson: before chasing a "bug" on a massive-activation
   model, reproduce the reference path naively in fp32 — if *it* also diverges,
   the divergence is numerical, not yours.

**Then the decoder MoE was the whole budget.** With the encoder on the GPU,
`moe_ffn_cpu` (scalar, per-token, re-dequantizing q4_k experts every step) was
~99% of LLM time: ~2000 ms/layer prefill, ~47 ms/layer/token decode; attention
was already negligible. Ported it into the layer graph via `ggml_mul_mat_id`,
reusing crispembed.cpp's BERT-MoE pattern (router→softmax→`ggml_top_k`→`get_rows`
weights→`mul_mat_id` gate/up/down→weighted sum + a combined shared expert).
The 64 per-expert tensors are stacked once at load into `[in,out,n_exp]`
(`stack_moe_experts`, a `memcpy` of each quantized expert into its slice — same
shape/type so blocks stay aligned; +~1.3 GB, gated so the CPU path doesn't pay
it). Per-layer prefill ~2015 ms → ~50 ms (~40×); **full OCR ~121 s → ~43 s**,
byte-identical output. `DS_MOE_CPU=1` keeps the scalar path.

**Then vision — and the surprise was the convs, not the attention.** Per-stage
timers showed SAM's 12 attention layers (already Metal) were only ~3 s; the rest
was scalar CPU: the **neck/downsample `conv2d_cpu`** (3.7-8 s, thread-variance-
prone) and patch embed (~2 s). Threading both (exact) roughly halved them, then
porting the neck/downsample to `ggml_conv_2d` (`build_sam_neck_graph`: 4 convs +
2 channel-axis LayerNorm-2d via permute→`ggml_norm`→affine→permute) dropped it to
**~150 ms** (~20-40×). Gotcha: the conv kernels are Q8_0 (vision floor), and you
**cannot reshape a quantized `[768,256]` to `[1,1,768,256]`** (`ne[0]=1` breaks
the 32-block) — dequant to F32 and feed as graph inputs. SAM ~12 s → ~4.7 s,
`sam_output` cos unchanged (0.999253). `DS_SAM_CONV_CPU=1` keeps the CPU chain.
Net: full OCR ~9 min (start of session, never completed) → **~23 s**, character-
perfect. Remaining costs: model load (~5-12 s, cold disk + Metal buffer copy) and
the SAM attention (~4 s, hard to flash-attn due to decomposed rel-pos bias).

Harness notes from the hunt: the per-stage diff was comparing the **wrong
tensors** (pre-neck 4096×768 vs the final 256×896 SAM output; pre-norm full-seq
vs the post-norm query-half encoder output) — the dead `diff_ref_path` is now
wired to a `DS_REF` env var with corrected comparison points + per-layer
bisection. `tools/dump_deepseek_ocr2_reference.py` was rewritten to instantiate
the vision modules standalone from `deepencoderv2.py` (the bundled MoE
`modeling_deepseekv2.py` won't import on transformers ≥4.48 — `LlamaFlashAttention2`
was removed), so the reference GGUF builds on CPU without the 3.4B decoder.

## Qwen2-VL (Qari-OCR) parity: four independent bugs, four layers

Qari-OCR (a Qwen2-VL-2B Arabic OCR fine-tune) produced garbage. The diff
harness (`test-qwen2vl-diff`) plus a PyTorch ground-truth comparison localized
**four** independent bugs, one per layer of the stack. Methodology: dump the
HF model's vision-merger output, token_ids and per-layer LLM hidden states for a
real image, inject them into the C++ engine via test hooks (`GEN_FROM_REF`,
`LLM_FROM_REF`), and bisect.

1. **Vision MLP activation.** Qwen2-VL `VisionMlp` uses `ACT2FN[hidden_act]`
   with the *vision* config default `hidden_act="quick_gelu"` (`x·σ(1.702x)`),
   NOT the merger's exact `nn.GELU()`. The engine used `ggml_gelu_erf` for both.
   Fix: `ggml_gelu_quick` for the ViT block, keep `ggml_gelu_erf` for the
   merger. (vis_layer_0 cos 0.995 → 0.99999.)

2. **Vision 2D-RoPE inv_freq.** `VisionRotaryEmbedding` is built with
   `dim = head_dim/2`, so `inv_freq[j] = theta^(-2j/(head_dim/2))`. Using
   `head_dim` in the denominator makes the frequencies decay half as fast —
   a subtle error (layer 0 still ~0.995) that compounds over 32 layers to
   destroy the merger (cos 0.06). Shared by Qwen2.5-VL (same
   `VisionRotaryEmbedding(head_dim//2)`), so the fix is correct for both.
   (vis_merger 0.37 → 0.99; last_logits 0.96 → 0.9999.)

3. **`last_logits` validates only the LAST position.** The prefill's final-token
   logit matched HF at 0.9999, yet generation was garbage. For the last token,
   causal == bidirectional attention, so a correct last-token logit does NOT
   prove the per-position KV is right. Always validate intermediate positions
   (per-layer hidden states across the WHOLE sequence) before trusting prefill —
   `LLM_FROM_REF` (inject HF embeds) gave cos 0.99997 across all 714 positions
   and proved the LLM forward correct, isolating the bug to generation.

4. **No-cache decode dropped the image; KV-cache outputs were pruned.** The
   single-token decode fell back to a full-recompute path that called
   `run_llm_forward` WITHOUT the image (image-blind → the model "describes a
   blank page" / answers conversationally). Pass the image through. The KV-cache
   path itself had two bugs that only surfaced once the recompute was fixed:
   - **(a) side outputs pruned.** `k_out_N`/`v_out_N` are not ancestors of the
     logits, so `ggml_build_forward_expand(gf, logits)` dropped them and
     `ggml_graph_get_tensor` returned null → silent no-cache. Call
     `ggml_build_forward_expand` on each side output explicitly.
   - **(b) cached V was a view into a reused buffer.** In the decode step
     `K_new` comes from `ggml_rope_multi` (materialized) but `V_new` was a
     `ggml_reshape_3d` *view*. Marking a view as a graph output and reading it
     back under a no-alloc scheduler returns GARBAGE — the scheduler reuses that
     buffer for later ops. Symptom: the cached K matched HF to ~1e-3 but the
     cached V was off by 6-9 (massive-activation magnitudes) at scattered
     elements; generation matched for 1-2 tokens then collapsed. Fix:
     `ggml_cont` K_new/V_new before `ggml_set_output`. (The prefill already did
     `reshape_2d(ggml_cont(V), ...)`; the decode forgot the cont.)
   General rule: **any tensor you `ggml_set_output` and read back via
   `ggml_backend_tensor_get` under a no-alloc scheduler must be materialized
   (`ggml_cont`), not a view** — otherwise its storage is fair game for reuse.

### Cross-backend audit of the view-as-output KV bug (2026-06)

Two KV-cache designs exist across the autoregressive decoders:
- **Read-back into a host vector** (qwen2vl, lightonocr, deepseek_ocr2): compute
  K/V in the graph, `set_output`, `ggml_backend_tensor_get` into `std::vector`,
  feed back as inputs next step. **Vulnerable** to both the prune bug (side
  outputs not expanded) and the view bug (V is a reshape view). This is the
  pattern that broke Qari.
- **`ggml_cpy` into a persistent KV buffer** (got_ocr, glm_ocr, internvl2_ocr):
  allocate persistent K/V tensors, `ggml_build_forward_expand(gf, ggml_cpy(K,
  k_view))`. No host read-back, cpy materializes. **Safe** — prefer this pattern
  for new backends.

Audit result: `lightonocr.cpp` had the cont bug AND three more (see next
section); now fixed and matching HF. `deepseek_ocr2.cpp` had the same read-back +
reshape-view pattern — the cont fix is applied (provably correct), but it is
NOT verified end-to-end (no local model) and, like lightonocr, may have further
architecture bugs needing a diff-vs-HF pass. got/glm/
internvl2 are safe (cpy pattern). All CPU-scalar decoders (mixtex, bttr, hmer,
posformer, ppformulanet*, granite_vision, math_ocr, decoder_embed) have no ggml
decode graph → immune.

## LightOnOCR-2-1B (Pixtral ViT + Qwen3): four bugs, fixed via HF diff

LightOnOCR looped ("ALIEN…", then digit garbage) despite a git note claiming the
KV cache was "confirmed working". A PyTorch diff (dump `vision_tower` /
`multi_modal_projector` / `language_model` layer outputs, compare per-row cos)
localized **four** bugs — the projection ones were the blockers (proj cos was
≈0, i.e. random):

1. **KV-cache V was a reshape view** marked `set_output` and read back → garbage
   (the cross-backend bug above). `ggml_cont` before `set_output` + expand the
   side outputs.
2. **Pixtral 2D-RoPE built interleaved but applied rotate-half.** The cos/sin
   table used interleaved layout (`dim[2i]`=h, `dim[2i+1]`=w) and `freqs[i]` for
   both axes, but `apply_rope` is rotate-half. Pixtral
   (`PixtralRotaryEmbedding`) is rotate-half: `freqs=1/theta^(2k/dim)`, height
   uses `freqs[::2]`, width `freqs[1::2]`, angle vector `[h(dim/4)|w(dim/4)]`
   **repeated** across the two halves (idx j and j+dim/2 share the angle).
   Verified numerically equal to HF (max diff 0.0).
3. **Projector RMSNorm in the wrong place.** `Mistral3MultiModalProjector` is
   `norm → patch_merger → linear_1 → gelu → linear_2`; the C++ applied the norm
   *last*. The norm is an RMSNorm over the per-patch D-dim vision features,
   **before** the 2×2 merge.
4. **Patch-merger merge order was patch-major, not channel-major.** Mistral3's
   `Mistral3PatchMerger` uses `F.unfold`, which lays out the `D·merge²` vector
   **channel-major**: `[c0·(k00,k01,k10,k11), c1·…]`. The C++ concatenated whole
   patches (`[patch00's D ch | patch01's D ch | …]`) — a permutation the merging
   weight can't undo (→ proj cos ≈ 0, random). Fix: `dst[c*msq + kpos] = src[c]`.

Result: first token and OCR text match HF exactly ("Qari OCR parity smoke test /
Invoice number: QA-2026-0616 / Total due: $42.75 / Please return plain text
only."). `vis_out` cos ≈0.99 / `proj_out` ≈0.88 vs fp32 HF is just q4_k
quantization — the greedy output is still exact.

**Diagnostic tells:** a VLM that says "I'm just a text-based assistant" or
paraphrases the OCR instruction (often in Chinese, the Qwen base language) is
not seeing the image — suspect splice/decode, not vision. Conversely, output
that's coherent-but-wrong-content with a correct first token points at
per-position KV, not the prefill logit.

**Out-of-distribution caveat:** Qari is an *Arabic full-page* OCR model. Test it
on rendered Arabic (PIL + raqm, `direction='rtl'`), not sparse English — with a
describe-style prompt and a tiny image it legitimately answers "blank page",
matching HF.

## GELU variant matters for token classification

HuggingFace/PyTorch uses erf-exact GELU (`torch.nn.functional.gelu`), not the
tanh approximation from the original BERT paper. For embedding retrieval the
difference is negligible (cos ~0.9999), but for token classification (NER) the
small per-value differences (~1e-4) compound through 12 layers and flip argmax
on borderline tokens. Fix: always use `ggml_gelu_erf` for BERT/XLM-R models.

Before fix: 2/4 entities detected (missing Apple ORG, Cupertino LOC).
After fix: 4/4 entities match Python exactly, all scores > 0.997.

## Cased vs uncased BERT tokenizer auto-detection

BERT-cased models (e.g. `dslim/bert-base-NER`) require case-preserving
tokenization. The GGUF doesn't store `do_lower_case`. Detect from vocab:
if single-letter uppercase tokens ("A", "B", ...) exist in the vocab, it's
a cased model. WordPiece tokenizer must skip `tolower()` in that case.

Wrong casing: "Barack" → ["bar", "##ack"] (6 subwords, all predicted O).
Correct casing: "Barack" → ["Barack"] (1 token, predicted B-PER with 0.999).

## BiACM: attention score fusion, not embedding fusion

LiLT's BiACM (Bidirectional Attention Complementation) adds text and layout
attention **scores** before softmax — NOT the embeddings themselves. Each stream
maintains separate Q/K/V projections and separate FFN layers. Only the attention
score matrices are shared (added element-wise at each layer). This means
`ggml_flash_attn_ext` cannot be used — scores must be computed manually
(`Q @ K^T`), summed, then passed to `ggml_soft_max` before applying to V.

## Layout embedding concatenation order

LiLT layout embeddings: 6 position lookups concatenated in this exact order:
`x(x0), y(y0), x(x1), y(y1), h(y1-y0), w(x1-x0)` → 6×128 = 768d.
Getting h/w swapped (w before h) causes cos=0.28 at the embedding level,
cascading to cos=-0.35 at layer 11. Getting it right: cos=1.000000 on all layers.

## RoBERTa position IDs start at 2, not 1

RoBERTa uses `padding_idx=1`, so positions start at `padding_idx + 1 = 2`.
Using offset 1 (like standard BERT) causes cos=0.97 at layer 0, degrading
to cos=0.80 by layer 11 — subtle enough to look like a precision issue
but actually a systematic bug. Fixed: `pos_ids[i] = i + 2` for RoBERTa/XLM-R.

## crispembed-diff test: always use matching inputs

Parity tests must use identical inputs between Python reference and C++.
The LiLT diff test initially used hardcoded bboxes in C++ that didn't match
the Python dumper's dynamically-generated bboxes — cos dropped to 0.20 at
the layout embedding level. Fix: store `input_ids` and `bbox` in the
reference GGUF and read them in the C++ test.

## Shared library pattern: conditional fallback

When extracting code into shared `crisp_*/` libraries from CrispASR:
- Guard CrispASR-specific code (auto-download, model registry) with
  `#ifdef CRISPASR_BUILD` — set via CMake `target_compile_definitions`
- Both repos check `EXISTS "${CRISP_*_DIR}/CMakeLists.txt"` and fall
  back to local copies when the sibling repo is absent
- Link order matters on MinGW: consumer before provider
- Always run full unit tests (439 in CrispASR) after the refactor

## Qwen2-VL vs Qwen2.5-VL config field names

The vision config schema differs between Qwen2-VL and Qwen2.5-VL:

| Field | Qwen2-VL | Qwen2.5-VL |
|-------|----------|------------|
| ViT block dim | `embed_dim` (1280) | `hidden_size` (1280) |
| Merger output | `hidden_size` (1536) | `out_hidden_size` (2048) |
| FFN size | `mlp_ratio` (4) → computed | `intermediate_size` (3420) |
| Input channels | `in_chans` (3) | `in_channels` (3) |

Critically, Qwen2-VL's `hidden_size` is the **merger output** (= LLM input dim),
not the ViT block dim. The GGUF `vision.hidden_size` must be set to `embed_dim`
(the ViT block dim), not `hidden_size`. Getting this wrong means every attention
head_dim computation in the engine is wrong → garbage output, no obvious crash.

Fix: use `getattr(vc, 'embed_dim', vc.hidden_size)` for the block dim,
`getattr(vc, 'intermediate_size', None) or embed_dim * mlp_ratio` for FFN.

## NAFNet dequant: always use ggml_get_type_traits for quantized tensors

When loading quantized GGUF weights for CPU-scalar inference, the `to_f32()`
helper must handle all quantized types — not just F32 and F16. The pattern:

```cpp
const auto * traits = ggml_get_type_traits(t->type);
if (traits && traits->to_float) {
    traits->to_float(t->data, buf.data(), n);
}
```

Failing to do this (e.g. `memset(buf, 0, ...)` for unknown types) silently
produces zero weights → the model runs but outputs garbage. The cosine drops
from 0.998 to 0.932 — high enough to look "plausible" but clearly wrong.
This was caught because Q8_0 and Q4_K produced *identical* cosines (both
were reading zeros), which shouldn't happen if real dequantization is working.

## NAFNet quantizer: per-channel scale factors are add operands

NAFNet's `beta` and `gamma` tensors `[1, C, 1, 1]` are used as element-wise
multiply operands in residual connections (`x = input + block_output * beta`).
Despite being 4D, they have only C elements and are extremely sensitive to
quantization. The quantizer's `is_add_operand` guard must include them:
`.beta` and `.gamma` patterns alongside `.ls1`/`.ls2` (LayerScale).

NAFNet conv weights (1x1 and 3x3 DW) all have ne[0] < 32 (the Q8_0 block
size), so the quantizer's existing `ncols % qk != 0` guard already skips
them. The beta/gamma guard is defense-in-depth.

## Hough deskew: run Sobel on grayscale, not binarized image

For document deskew via Hough transform, running Sobel edge detection on
the raw grayscale image works much better than binarizing first then running
Sobel. Binarization destroys gradient information at text boundaries,
especially for thin lines and anti-aliased edges at small angles. The top-5%
edge threshold (not top-10%) gives enough votes for reliable angle detection.

## ggml quantized reshape: dequant BEFORE reshape

ggml quantized types (Q8_0, Q4_K, etc.) store data in fixed-size blocks
(e.g. Q8_0 = 32 elements/block). `ggml_reshape_4d` changes shape metadata
without moving data. If the reshape creates `ne[0]` smaller than the block
size (e.g. ne[0]=3 for a 3×3 conv kernel), subsequent operations that
read the quantized data will access invalid block boundaries → crash.

**Rule**: Always dequant quantized tensors to F32 BEFORE reshaping to
arbitrary dimensions. Then reshape, then cast to F16 if needed.

Also: ggml only supports Q→F32 dequantization (`ggml_cast`), not Q→F16.
Attempting `ggml_cast(Q8_0, F16)` hits `GGML_ABORT("fatal error")` in
`ggml_compute_forward_dup`. Always go Q→F32→F16 (two casts).

## GPU backends: read weights via ggml_backend_tensor_get, not t->data

Models that mix a ggml graph (GPU) with CPU-scalar fallback code — like
`surya_det.cpp`, whose LiteMLA + decode head stay scalar — must not read
weight bytes through `tensor->data`. On a CPU backend `t->data` is the host
buffer, but on a GPU backend (Metal/CUDA) it is not a valid host pointer, so a
direct `memcpy(out, t->data, …)` or `traits->to_float(t->data, …)` reads
garbage (or crashes). Route every host-side weight read through
`ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t))`, which copies out of the
tensor's own buffer regardless of backend, then dequantise from that staging
buffer. (Apple Silicon's unified memory happens to make `t->data` work for
Metal in many cases, but relying on that is non-portable and breaks on CUDA.)

Companion lesson: switch the backend with `ggml_backend_init_best()` (not a
hardcoded `ggml_backend_cpu_init()`) and gate `ggml_backend_cpu_set_n_threads`
behind `ggml_backend_is_cpu()` — it is a CPU-only call. Provide an env override
(e.g. `SURYA_DET_FORCE_CPU=1`) so parity debugging can pin the CPU path.

## Per-layer parity comparison must happen inside the layer loop

When comparing C++ per-layer outputs against a reference GGUF, the
comparison must happen immediately after each layer completes — NOT
in a separate loop after all layers finish. The hidden state buffer
is overwritten by each subsequent layer, so a post-loop comparison
would compare every layer's reference against only the final layer's
output (producing spurious failures on early layers and a spurious
pass on the last layer).

## SAM ViT windowed attention: LN before partition

SAM ViT-B applies LayerNorm BEFORE window partitioning. For windowed
layers in a ggml per-layer graph:
1. Apply LN1 on CPU to the full (unpartitioned) hidden state
2. Window-partition BOTH the LN'd state (graph input) and the original
   state (residual input)
3. The graph uses `skip_ln1=true` to avoid double-normalization
4. The residual connection uses the original (pre-LN) partitioned data

Getting this wrong (LN after partition) introduces zeros from padding
tokens into the LayerNorm statistics, corrupting edge windows.

## DeBERTa rel_embd must be dequantized for CPU-side expansion

DeBERTa's relative position embeddings are expanded on CPU (log-bucket
indexing → [H, T*T] tensor) before the ggml graph runs. With quantized
models (Q8_0/Q4_K), the `rel_embd.weight` tensor is no longer F32 —
reading it via `ggml_backend_tensor_get` gives raw quantized bytes.
Must use `tensor_to_f32_backend()` which reads raw bytes then calls
`ggml_get_type_traits(type)->to_float()` to dequantize. Same applies
to `encoder_ln_w/b` used in the LayerNorm applied to rel_embd.

## Dual-backbone GLiNER: parameterize span mode and hidden dim

GLiNER models differ in span representation mode:
- markerV1 (LFM2): concat(proj_start, proj_end, proj_first) → 3*hidden
- markerV0 (DeBERTa): concat(proj_start, proj_end) → 2*hidden

The out_project MLP input dimension changes accordingly. Parameterize
`span_cat_dim` based on span_mode rather than hardcoding `3*hidden`.
Also parameterize `head_dim_gl` (GLiNER head dimension) separately from
`enc_hidden` (encoder output dimension) to handle the 768→512 projection.

## PARSeq two-stream decoder (XLNet-style attention)

PARSeq's decoder uses a two-stream design from XLNet where both position
queries and content tokens are maintained separately. Key details:

1. **Token ordering is non-standard**: `[EOS=0, chars=1..94, BOS=95, PAD=96]`.
   The head output excludes BOS and PAD, so it has 95 classes (0=EOS + 94 chars).
   This is because `BaseTokenizer` puts `specials_first=(EOS,)` before charset
   and `specials_last=(BOS,PAD)` after.

2. **Context construction**: The content stream at decode position k is NOT just
   the token embedding. It's `pos_queries[k-1] + embed(token_k)` for k>=1, and
   just `embed(BOS)` for k=0 (no position query for BOS — it's "null context").

3. **norm_c is essential**: Context K/V in self-attention are normalized by
   `norm_c` (LayerNorm), while queries are normalized by `norm_q`. Skipping
   norm_c produces garbage.

4. **Efficient AR decode**: At step i, only one query position is used
   (`pos_queries[i]`), with context tokens 0..i. No causal mask needed since
   T=i+1 and all positions are visible. The paper's `query_mask` only matters
   for the full N-step forward (training/refinement).

5. **Non-square patch kernel**: Patch embedding uses Conv2d with kernel [4,8]
   (height 4, width 8). ggml_conv_2d doesn't support non-square kernels, so
   patch embedding runs CPU-side as a manual extract+matmul.

## ggml GQA broadcasting (critical for decoder models)

`ggml_mul_mat` natively broadcasts ne[2] when `b->ne[2] % a->ne[2] == 0`.
For GQA (16 Q heads, 8 KV heads): **do NOT explicitly repeat K/V**.
`ggml_repeat` tiles `[h0..h7, h0..h7]` which is WRONG for GQA (should
be `[h0,h0,h1,h1,...]`). Just let mul_mat broadcast — it handles the
interleaved head mapping correctly internally.

Also: after attention, reshape to `q_dim = n_heads × head_dim` (NOT
`hidden_size`). For GQA models, q_dim ≠ hidden_size (e.g. 2048 vs 1024).

## BERT post-LN vs pre-LN

BERT uses post-LayerNorm: `attn → residual_add → LN → FFN → residual_add → LN`.
Many newer models (GPT, LLaMA) use pre-LN. Getting this wrong produces
output that looks plausible but has completely wrong magnitudes.

## RoPE application order

For Qwen3: RoPE is applied on `[head_dim, n_heads, T]` tensor (BEFORE
permute to `[head_dim, T, n_heads]`). `ggml_rope_ext` requires ne[2]=T
(the position dimension), which matches the unpermuted layout. Applying
RoPE after permute crashes with dimension mismatch.

At position 0, RoPE is identity (cos=1, sin=0), so position-0 values
match regardless of whether RoPE is applied. Debug with position > 0
to verify RoPE correctness.

## Tokenizer types for embedding models

| Model family | Tokenizer | Implementation |
|---|---|---|
| BERT/MiniLM/GTE | WordPiece | Greedy longest-match with ## prefix |
| XLM-RoBERTa/E5/Arctic/PIXIE | SentencePiece Unigram | Viterbi DP (NOT bigram merge) |
| Qwen3/Octen/F2LLM | GPT-2 BPE | core_bpe byte-level BPE with merges |
| Gemma3/Harrier-270M (our GGUFs) | SentencePiece BPE | `BPETokenizer` spm_style: BPE **merges** + ▁ + BOS/EOS |
| gemma-embedding (llama.cpp export) | SentencePiece BPE | `SentencePieceTokenizer` spm_bpe: bigram merge **from scores** (no merges array) |

Auto-detected from GGUF metadata: `tokenizer.ggml.type` (0=WP, 1=BPE, 2=SP),
`tokenizer.ggml.model` string, or heuristic (vocab > 100K → SentencePiece). Note
the two Gemma rows: our own converter bakes a `merges` list (BPETokenizer path),
but community llama.cpp SPM exports ship `scores` and NO merges, so the SP
tokenizer reconstructs the same segmentation by bigram-merging on scores (see the
gemma-embedding learning above).

### Critical: SentencePiece Unigram needs Viterbi; SentencePiece BPE needs bigram-merge

Two SentencePiece algorithms, opposite requirements — pick by the model, not by
"it's SentencePiece":

- **Unigram (XLM-R/E5/T5):** scores are log-probs → **Viterbi DP** (max-sum path).
  The llama.cpp-style bigram merge does NOT produce correct tokenization here.
- **BPE (Gemma/Llama):** scores are merge RANKS → **bigram greedy-merge** (merge
  the highest-scoring adjacent pair). Viterbi over ranks OVER-segments (it sums
  ranks and prefers many small pieces to one big token — e.g. `▁w+or+ld` beats
  `▁world`). `SentencePieceTokenizer` has both: default Viterbi, `spm_bpe` mode
  for the merge algorithm (set from `tokenizer.ggml.model` ∈ {llama, gemma}).
Example: "▁world" exists as token 8999, but bigram merge breaks it into
["▁w", "or", "ld"] because greedy pair merging can't find the global optimum.

**Viterbi DP**: For each position i, try all vocab tokens ending at i,
pick the segmentation with the highest total score. O(n × max_token_len).
This matches HuggingFace's `tokenizers` library exactly.

### SentencePiece BPE vs GPT-2 BPE

These are different tokenizer families with different pre-processing:
- GPT-2 BPE: byte-level encoding (spaces → Ġ), no BOS/EOS by default
- SentencePiece BPE (Gemma): spaces → ▁ (U+2581), BOS/EOS tokens

### Vocab scores for SentencePiece

SentencePiece Unigram models need per-token scores for Viterbi. These come from:
1. `tokenizer.sp_model.GetScore(i)` — but not available for all tokenizer classes
2. `tokenizer.json` → `model.vocab` → list of `[token, score]` pairs

If scores are missing (all zeros), the tokenizer degenerates to random merging.

## Per-op debugging methodology

Same as CrispASR: dump every intermediate tensor from BOTH HF reference
and our ggml graph, compare at each stage. The divergence point identifies
the exact broken operation. For Octen-Embedding-0.6B, this revealed:
- input_ln: MATCH
- q_proj/k_proj: MATCH
- q_norm/k_norm: MATCH
- o_proj: MISMATCH → GQA repeat was wrong
- Fix: remove ggml_repeat, let mul_mat broadcast → MATCH

## RoBERTa/XLM-R position embedding offset

RoBERTa-family models (XLM-R, PIXIE-Rune, arctic-embed-l-v2) offset position
IDs by `padding_idx + 1 = 2`. Position IDs for a 4-token sequence are
`[2, 3, 4, 5]`, not `[0, 1, 2, 3]`. Position embedding index 1 is all-zeros
(padding), index 0 is low-norm. Getting this wrong produces ~0.74 cosine sim
instead of 0.999.

Stored as `bert.position_offset` in GGUF metadata.

## Gemma3 architecture specifics

Gemma3 (Harrier-270M) differs from Qwen3/LLaMA in several critical ways:

1. **RMSNorm uses `(1 + weight)`**: Gemma3 RMSNorm computes
   `output * (1.0 + weight)` instead of `output * weight`. The stored weights
   do NOT include the +1 offset. Missing this makes all layer outputs wrong.

2. **Embedding scale**: Token embeddings are multiplied by `sqrt(hidden_size)`.
   The exact value is stored in `embed_tokens.embed_scale` (f16 precision:
   `sqrt(640) ≈ 25.25` not `25.298`).

3. **Extra norms**: 4 norms per layer (not 2):
   - `input_layernorm` → before attention
   - `post_attention_layernorm` → after attention, BEFORE residual add
   - `pre_feedforward_layernorm` → before FFN
   - `post_feedforward_layernorm` → after FFN, BEFORE residual add

4. **Attention scaling**: Uses `query_pre_attn_scalar` (= head_dim) instead
   of `sqrt(head_dim)`. Scale = `1/sqrt(qpas)`.

5. **gelu_pytorch_tanh**: Activation function; ggml_gelu uses tanh approx.

6. **head_dim != hidden_size/n_heads**: Gemma3 has head_dim=256, hidden=640,
   n_heads=4. Standard calculation gives 160, but explicit head_dim is 256.

7. **SentencePiece BPE tokenizer**: Uses ▁ space marker (not GPT-2 Ġ),
   needs BOS(2) at start and EOS(1) at end.

## Ollama integration learnings

### Architecture: Ollama uses ggml via CGO (same as CrispEmbed)

Both Ollama and CrispEmbed use ggml for tensor computation. Ollama wraps ggml
ops in Go structs via CGO (`C.ggml_mul_mat`, `C.ggml_rms_norm`). CrispEmbed
calls ggml directly from C++. The computation graphs are functionally identical.

### Phantom-space token vocabulary (critical for WordPiece)

Ollama's WordPiece tokenizer expects tokens in SentencePiece-style format:
- `"hello"` → `"▁hello"` (prepend ▁)
- `"##ing"` → `"ing"` (strip ##)
- `"[CLS]"` → `"[CLS]"` (keep special tokens)

Without this transformation, cos drops from 1.0 to ~0.19.

### GELU variant matters (exact erf vs tanh approximation)

BERT uses exact GELU (erf-based). Ollama's `.GELU()` uses tanh approximation
(`ggml_gelu_inplace`). Must use `.GELU_ERF()` for BERT/XLM-R encoder models.
Difference: cos 0.996 → 1.000.

### SentencePiece Unigram needs Viterbi DP, not pairwise merge

Ollama's existing `SentencePiece` tokenizer uses BPE-style greedy pairwise
merge (priority queue). This is WRONG for Unigram models (XLM-R, e5-small).
We added `SentencePieceUnigram` using Viterbi DP (same as CrispEmbed's
tokenizer_spm.cpp). Must also prepend space before tokenization.

### Gemma3 (1+weight) RMSNorm must be pre-baked for Ollama

Ollama's RMSNorm does `rms_norm(x) * weight`. Gemma3 needs `rms_norm(x) * (1 + weight)`.
CrispEmbed handles this at runtime with a `ones` tensor. For Ollama export,
pre-add +1 to all norm weights in the GGUF.

### Quantized token_types breaks Ollama binary ops

Ollama's ggml doesn't support `f32 + q8_0` in elementwise ops. The tiny
`token_types.weight` tensor (2 rows) must be kept as f32 during quantization.
Error: `binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`.

### Nil-guards needed for optional model components

Ollama's Qwen3 model.go unconditionally calls `QueryNorm.Forward()` — panics
for models without QK-norm (e.g. Jina v5). Gemma3 embed.go unconditionally
iterates `Dense` projection — panics for models without it (Harrier-270M).

### Jina v5 LoRA adapters need merge before export

Jina v5 models use task-specific LoRA adapters (retrieval, classification,
clustering, text-matching). Must call `model.set_adapter("retrieval")` then
`model.merge_and_unload()` before GGUF export. The `encode()` method does
more than standard forward+pool, so merged output won't exactly match HF.

### SentencePiece BERT models should use bert arch, not xlmr

Models like multilingual-e5-small report `model_type="bert"` with SentencePiece
tokenizer. These are BERT models (no position offset), not XLM-R. Only true
`roberta`/`xlm-roberta` types need the `xlmr` arch with position offset.

`paraphrase-multilingual-MiniLM-L12-v2` is another instance of this pattern —
BERT (post-LN) body + 250K-token XLM-R SentencePiece-Unigram vocab. The
converter detects this from `config.model_type == "bert"` and writes
`bert.position_offset = 0`. End-to-end cosine vs HF: **1.000000** on f16/f32,
**197/197 encoder tensors bit-exact** (max\|Δ\|=0) — see
`tests/parity_layers_bert.py`.

### SPLADE detection must look at checkpoint files, not state_dict

`AutoModelForMaskedLM.from_pretrained()` will *silently random-initialise*
missing `cls.predictions.*` keys instead of failing. Checking
`any("cls.predictions" in k for k in model.state_dict())` therefore returns
True for **every** plain encoder checkpoint, baking a random MLM head into
the GGUF (~600 KB of garbage tensors that load as "MLM/SPLADE head loaded"
at runtime).

The fix in `models/convert-bert-to-gguf.py` is to peek at the safetensors /
pytorch_model.bin header directly via `safe_open()` and only call
`AutoModelForMaskedLM` if `cls.predictions.` or `lm_head.` keys are
**actually present in the checkpoint**. `output_loading_info=True` looked
like an obvious alternative but returns inconsistent shapes (single model
vs 5-tuple) depending on `use_safetensors`, so the header-peek path is the
robust one.

This bug affected every plain `sentence-transformers/*` and `all-MiniLM-*`
conversion prior to 2026-05-11. Re-converting those models drops the file
size by ~1 MB each and removes the misleading "MLM head loaded" log line.

## PPFormulaNet-S / Texo-Distill OCR port

### MBart uses PRE-LN, not POST-LN

Despite MBart config saying `layer_norm_eps` and having `*_layer_norm` weights,
the HuggingFace MBart decoder applies **PRE-LN**: LayerNorm before attention/FFN,
with the residual connection skipping the LN. The TrOCR decoder (math_ocr.cpp)
uses POST-LN. Getting this wrong produces completely different logit distributions
— the first token diverges from logit 16.1 (correct) to 1.7 (wrong).

```
PRE-LN (MBart):                    POST-LN (TrOCR):
  residual = x                        Q = linear(x)
  x = LN(x)                          ...attn...
  Q = linear(x)                      x = x + attn_out
  ...attn...                          x = LN(x)
  x = residual + attn_out
```

The encoder diff test (cos=1.0) will NOT catch decoder LN ordering bugs —
you MUST also dump and compare decoder layer outputs from the Python reference.

### ODR violations from shared struct names

Multiple `.cpp` files defining `struct dec_layer` in the anonymous namespace
causes One Definition Rule violations. The linker may silently use the wrong
definition (144 bytes from decoder_embed_internal.h instead of 208 bytes from
ppformulanet_ocr.cpp), causing heap-buffer-overflow in `map_tensors`. ASAN
catches this immediately. Fix: use unique struct names (`ppfn_dec_layer`).

### UniMERNet preprocessing is NOT ImageNet

PPFormulaNet-S/Texo uses UniMERNet's image processor:
- Convert to grayscale, replicate to 3ch
- Resize preserving aspect ratio, pad with **black** (not white)
- Normalize: **mean=0.7931, std=0.1738** (NOT ImageNet 0.485/0.229)
- Input is always 384x384

Using ImageNet normalization produces garbage output even though the encoder
activations look reasonable — the model was trained with different pixel statistics.

### HGNetv2 StemBlock padding

StemBlock uses kernel_size=2 convolutions (stem2a, stem2b) with padding=0.
Before each, the input must be explicitly padded with `F.pad(x, (0,1,0,1))`.
Without this, the spatial dimensions mismatch at the concat step (pool output
vs stem2b output differ by 1 pixel).

### Conv-BN folding for CNN encoders

BatchNorm after Conv2d can be algebraically folded at conversion time:
```
fused_w = conv_w * (bn_weight / sqrt(bn_var + eps))
fused_b = bn_bias - bn_mean * (bn_weight / sqrt(bn_var + eps))
```
This eliminates all BN parameters from the GGUF, saving memory and compute.
The BTTR/HMER ports already did this; PPFormulaNet has ~150 BN layers to fold.

### 20M models are too small for Q4_K

The Texo-distill model (20M params, 384 d_model) produces identical output at
F32/F16/Q8_0, but Q4_K degrades noticeably — subscripts become wrong, tokens
repeat. The attention projections (384x384) and embedding table (1264x384) are
small enough that 4-bit quantization loses critical precision. Ship Q8_0 (22 MB)
as the smallest reliable variant.

### Debug prints: gate behind env vars, never remove

Decoder debug fprintf traces (`tok_emb+pos`, `after embed_ln`, `logits[91]`)
were essential for diagnosing the PRE-LN bug. Gate them behind
`getenv("PPFN_DEBUG")` rather than deleting. The crispembed-diff harness only
validates the encoder — decoder bugs require manual layer-by-layer tracing.

## llama.cpp implementation reference — what to borrow (2026-07)

Audit of `ggml-org/llama.cpp` @ ~`4fc4ec5` (July 2026). The support matrix and the
A/B-gated convergence backlog live in `PLAN.md → "llama.cpp parity, convergence &
A/B plan"`. This section is the technical deep-dive behind those steps. Verify any
file:line against the pinned ggml submodule before trusting it — upstream drifts.

### How llama.cpp structures encoders (one graph, GGUF-driven)

BERT / RoBERTa / XLM-R / NomicBERT / Nomic-MoE all share **one** builder
(`src/models/bert.cpp`); behavior is selected from GGUF metadata written at
convert time: `pooling_type`, `causal_attention=false` (→ bidirectional, no KV
cache, `build_attn_inp_no_cache`), RoPE presence, `moe_every_n_layers`. **Lesson:
make behavior data-driven in the GGUF, not hardcoded in the dispatcher** (our
`PLAN.md` C2 step). ModernBERT adds symmetric sliding-window attention
(`swa_type`, `set_swa_pattern`) + per-layer RoPE θ (global vs `rope_freq_base_
train_swa`) — a clean reference if we ever add it.

### Pooling & the RANK (reranker) head

`build_pooling()` (`src/llama-graph.cpp`) runs only when `cparams.embeddings`:
- NONE = pass-through per-token (late-interaction/ColBERT consumers pool client-side);
  MEAN = matmul with a normalized averaging matrix; CLS/LAST = `ggml_get_rows` of
  first/last token index.
- **RANK** = pooled vector → `cls` matmul(+b) → activation → optional `cls_norm`
  LayerNorm → `cls_out` matmul(+b). The activation is **GELU for ModernBERT, tanh
  otherwise**; ModernBERT pools MEAN, others CLS; Qwen3 rerankers softmax + LAST.
  **VERIFIED (2026-07-03) our `apply_classifier` matches:** 2-layer BERT/XLM-R
  rerankers (jina-v2, bge, ms-marco) use `tanh` (crispembed.cpp ~2926); the DeBERTa
  ContextPooler (mxbai) uses GELU-tanh (~2915). No ModernBERT reranker is in the
  roster (would need GELU), so no fix needed — the RANK heads are correct vs upstream.
  **⚠ CORRECTED (2026-08-05, G7c):** the ms-marco half of that claim was code-level
  only and did NOT hold for the shipped artifacts — their GGUFs carried a 1-layer
  head with NO pooler (the converter never emitted `bert.pooler.dense.*`), so
  native scored `dot(CLS,w)+b` instead of HF's `classifier(tanh(pooler(CLS)))`:
  calibration off by ~50x, tail ranking reordered. Fixed in the converter (fold
  pooler→`classifier.dense` + tanh, `63997e2c`) and re-shipped as `*-g7c.gguf`
  with re-pinned hashes; evidence in `tests/results/g7c/SUMMARY.md`. Two riders:
  (a) the DeBERTa gelu here is tanh-APPROX where HF `gelu` is erf — unmeasured,
  mxbai-only; (b) the local miniconda torch mis-executes BERT-class forwards
  (NaN/bus errors) — parity references on this box must come from ONNX Runtime.
- Qwen3-Embedding is trained **causal** (last-token/EOS pooling) — llama.cpp runs
  it causal and is *correct*. EmbeddingGemma and the LFM2.5 retrievers are
  bidirectional. Don't assume "decoder embedder ⇒ force non-causal".

### LFM2 ShortConv rides ggml_ssm_conv

LFM2 (`lfm2`, PR #14620) implements the short-conv block on the SSM path:
`in_proj` (3× expand) → chunk into 3 → causal shift → gated depthwise Conv1d
(kernel=3) → `out_proj`. Tensors `blk.N.shortconv.{in_proj,conv,out_proj}`; the
**conv weight stays F32** (special quant handling) — matches our own F32-cast
requirement for `ggml_mul` src[1] on Metal (see CLAUDE.md LFM2 note). If our LFM2
engine hand-rolls the conv, moving to `ggml_ssm_conv` gets better Metal kernel
coverage. Constraint: conv/recurrent state can't be partially erased, so KV/prefix
reuse must copy whole prefixes (upstream #19041) — relevant to `PLAN.md` C4.

### Qwen3-VL DeepStack + IMROPE = a reference for BidirLM-Omni

`tools/mtmd/models/qwen3vl.cpp` builds DeepStack exactly like our BidirLM-Omni
injection: per selected level, tensors `v.deepstack.%d.{norm,fc1,fc2}` →
reshape `[n_embd*merge, n_pos/merge, batch]` → norm → GELU-FFN → `ggml_concat`
across levels → concat onto the merger; final mmproj dim = base × (1 +
n_deepstack). Vision position encoding is **IMROPE** (interleaved M-RoPE) — the
same family we pin per-token (`pos_e = pos_t`) in `decoder_embed.cpp` (see CLAUDE.md
IMROPE landmine). This is a directly comparable implementation to diff our
DeepStack/MRoPE against. Distinguish from the **generic feature-layer concat**
(Granite/Gemma4: `clip.vision.feature_layer`, concatenates chosen intermediate
hidden states) — that's additive-vs-selective, two different mechanisms.

### mtmd image preprocessing internals (the Qwen2VLImageProcessor reference)

All preprocessing is C++ in `tools/mtmd/mtmd-image.cpp` (`struct img_tool`):
- Kernels: `resize_bilinear`, `resize_bicubic` (Catmull-Rom), and
  **`resize_bicubic_pillow`** — separable, precomputed coefficients, fixed-point
  `PRECISION_BITS=22`, filter **`a=-0.5`** (PIL-exact). In-code comment warns
  **GGML/PyTorch use `a=-0.75`** — this parity gotcha is very likely our observed
  sub-pixel resize residual (cos 0.999984). Test both when porting.
- `calc_size_preserved_ratio(inp, align_size, min_px, max_px)` = transformers
  `smart_resize`; `align_size = patch_size * n_merge`; `beta = sqrt(H*W/max_px)`
  rescales to the pixel budget. Budgets from `image_{min,max}_pixels` GGUF keys.
- `mtmd_image_preprocessor_llava_uhd`: `select_best_resolution` against
  `image_grid_pinpoints` (LLaVA-Next/Granite) or MiniCPM dynamic grid; per-model
  subclasses `_lfm2/_idefics3/_internvl/_granite/_deepseekocr`.
- Normalize: `(px - mean[c]) / std[c]` after u8→[0,1], mean/std from GGUF.
- Media injection: single `<__media__>` marker; `mtmd_tokenize` splits into
  TEXT/IMAGE chunks, wraps with begin/end vision tokens, emits grid nx/ny + pos
  type (NORMAL/MROPE/HUNYUANVL).

**Do not link libmtmd** — it PUBLIC-links all of `llama` and is oriented around
feeding a llama decoder. Align on formats (`mmproj-*.gguf` keys, `<__media__>`),
use the file as a spec, keep our own implementation. (Its resize is itself not yet
HF-byte-exact — open #16842/#17345/#17801.)

### imatrix quantization — the fix for our q4_k floor

The single highest-leverage quality lever, and offline-only (no graph risk).
`llama-imatrix` collects per-column sum-of-squared-activations (importance) over a
calibration corpus; `llama-quantize --imatrix` then minimizes **activation-
weighted** L2 error instead of unweighted, steering bits to weights actually
exercised (typ. 10–30% ppl reduction, largest at 2–4 bpw). Build the calibration
set from text resembling embedding-domain inputs. `IQ4_XS` (~4.46 bpw) needs a
good imatrix but is smaller/faster than `Q4_K_M`; `IQ4_NL` uses 32-weight blocks —
useful for our 256-alignment fallback (see `## Quantization notes → K-quant
fallback chain`, which currently drops small-dim tensors to Q4_0). MXFP4/ternary
are model-specific (FP4-trained / BitNet-QAT), not general q4 replacements. This
should lift LFM2 (~0.982) and BidirLM (~0.93–0.95) toward their q8_0 floor.

**Implemented as C1** (`src/imatrix.{h,cpp}` collector via eval-callback gated by
`CRISPEMBED_IMATRIX_OUT`; `crispembed-quantize --imatrix`). Rollout gotchas from
the Kaggle batch (`tools/kaggle/crispembed-imatrix-quant/`):
- **Source from the existing repo GGUF, not HF re-conversion.** Each `cstr/*-GGUF`
  repo already hosts a full-precision base (F32; base/q8 size ratio ~3.7). Using
  it sidesteps converter-arg guesswork — jina-v5's HF repo ships task LoRA
  adapters (`-classification/-clustering/-text-matching`); re-converting would
  pick the wrong weights. Auto-detect the base = largest non-quant `.gguf`.
- **Never overwrite baselines.** imatrix outputs upload under DISTINCT names
  (`*-q4_k-imatrix.gguf`, `*-iq4_xs.gguf`); q8_0/q4_k baselines are A/B-reference
  only. The registry default repoints to the A/B winner; `-q4k` serves imatrix.
- **CPU build, GPU only for internet.** A CUDA build compiles ggml-cuda's ~254
  `mmq-instance-*.cu` TUs (~15 min; the CrispASR ccache barely hits them) for
  kernels these ≤600M embedders never use. `-DGGML_CUDA=OFF`; keep `enable_gpu`
  true only because Kaggle CPU workers have no internet (kaggle_usage.md #3).
- **`kernels_output` only captures `/kaggle/working` files (not stdout, and in
  practice only `.ccache`)** — upload the A/B summary + `.imatrix` TO THE HF REPO
  for reliable retrieval.
- **Per-model winner varies.** q4_k+imatrix wins on the decoder embedders (lfm2,
  jina, octen, qwen3-embed), IQ4_XS+imatrix on the XLM-R/BERT encoders (bge, e5,
  gte, arctic) — smaller AND higher cos. Always A/B; don't assume one flavor.
  20-model table in `PLAN.md → C1`.
- **Auto-detect trap: LoRA task variants.** jina-v5 ships `-classification`/
  `-clustering`/`-text-matching` adapter GGUFs at the SAME size as the base
  retrieval model, so "largest non-quant .gguf" picked one → quantized the wrong
  weights. Fix: prefer the exact `{name}.gguf` and exclude task-suffix variants.
- **The collector wrote nothing — `clean_exit` bypasses `atexit` (2026-07-03).**
  The imatrix flush was registered via `atexit()`, but every one-shot CrispEmbed
  binary exits through `core_util::clean_exit()` → `_exit()` (deliberately skips
  ggml's slow Metal static-dtor teardown; see `src/core/clean_exit.h`), which also
  skips ALL atexit handlers and static destructors. So calibration collected the
  activation stats correctly, exited rc=0 with valid embeddings (431 KB JSON on
  qwen3-embed-8b), and threw the stats away at exit — the `.imatrix` came out
  empty and `crispembed-quantize` silently fell back to unweighted, producing a
  `-q4_k-imatrix.gguf` **bit-identical to the plain baseline** (`cos == cos` in the
  A/B). It looked model-specific only because `clean_exit` landed mid-rollout: the
  first 27 embedders were calibrated before it (collected fine), the last 3 big
  decoders (qwen3-embed-4b, octen-8b, qwen3-embed-8b) after it (empty). The eval
  callback *did* fire and match every weight — the bug was purely the flush never
  running. Fix (commit 07439db): call `crispembed_imatrix_flush()` from
  `crispembed_free()` (runs before `clean_exit`), guarded by `g_flushed` so the
  atexit fallback + explicit call write at most once per process. **Rule: never
  persist via `atexit` in any binary that ends in `clean_exit` — flush at a real
  teardown point.** The `fail-loudly-if-no-imatrix` guard in the Kaggle harness is
  what surfaced it (octen-8b/qwen3-embed-4b had slipped through pre-guard as
  silently mislabeled files, later corrected).
- **crispembed-quantize + SentenceTransformer Dense projections — FIXED (2026-07).**
  embeddinggemma-300m quantized `dense.0/dense.1` (ST Dense/Matryoshka heads) to
  q8_0; the output GGUF then failed to load — `GGML_ASSERT(offset+size <=
  ggml_nbytes) "tensor read out of bounds"` in `load_decoder_model`. PROVEN by
  diffing vs the working reference q8_0 (only dense.* differed: F32 vs Q8_0). Fix:
  a keep-F32 guard for `dense.*` in `tools/quantize.cpp`; re-quantized output
  loads + embeds cleanly (verified locally). Applies to any ST-Dense model.
- **Some models quantize poorly at 4-bit regardless of imatrix.** f2llm-v2-0.6b
  q4_k baseline 0.683 → +imatrix only 0.830; nomic-embed-text-v1.5 0.837 → 0.905.
  imatrix still helps, but for these keep q8_0 as the recommended flavor.
- **Rerankers get imatrix too — but the A/B metric is Kendall-tau, not cosine
  (2026-07-03).** Cross-encoders score (query, doc) pairs; there's no pooled
  embedding to cosine. The imatrix *collector* fires on the `--rerank` path with
  zero code change, so only the harness A/B differs: mean Kendall-tau on the doc
  ranking vs the q8/full-precision gold, mean|dscore| as tiebreaker (harness
  `MODE`/`rerank_ab`). All 7 rerankers quantized. imatrix reliably improves
  dscore; τ preservation is model-dependent — jina-v2 + ms-marco-L6/L12 stay
  τ=1.0 at 4-bit (wired to iq4_xs/q4_k+im), bge + mxbai drop to τ≈0.73–0.93 so
  their defaults stay q8_0. **Caveat: the eval set is small (n=5 queries × 4 docs),
  so τ is coarse (steps of ~0.033) and noisy — treat 4-bit-vs-q8 reranker calls as
  provisional; a larger paired corpus would firm them up.**
- **DeBERTa `rel_embd` must be dequantized on read — crashed ALL quantized DeBERTa
  models (2026-07-03).** `rel_embd` (disentangled-attention relative-position
  embeddings) is a 2-D weight the quantizer stores as Q8_0/Q4_K, but BOTH position-
  expansion paths (`run_encoder_raw` + `encode_tokens`) read it with a raw
  `n*sizeof(float)` get → `offset+size > ggml_nbytes` → "tensor read out of bounds"
  abort. So mxbai-rerank-base/xsmall-v1 and gliner-deberta could not run on any
  quantized GGUF (only the full-precision base worked). This surfaced as an
  "imatrix failure" but was unrelated — a plain `--rerank` on the q8 also crashed
  (masked earlier by reading `rc` through a `head` pipe). Fix: read via
  `core_cpu::to_f32` (dequant-safe), same pattern as the MLM/SPLADE head. Same
  class as the granite / pcs-q4k / MLM quant-read bugs — a quantized 2-D weight
  read as raw F32.
- **Fixed-label NER gets imatrix; the metric is micro span-F1 (2026-07-03).**
  bert-base-NER / xlmr-ner-hrl run the BERT-NER path, whose encoder is a *shared
  `crispembed_context`* — so the collector fires on `--ner` with no code change (A/B
  = exact (start,end,label) span-F1 vs full-precision gold; harness MODE `ner`).
  Both → iq4_xs (span-F1 1.0). **But first they needed a `bert_ner` classifier
  dequant fix (85feaeb):** `ner.classifier.weight` ships Q8_0/Q4_K and was read as
  raw F32 → "unsupported type 8" → **failed to load on ANY quant** (yet another
  instance of the quantized-2-D-weight-read-raw class). **GLiNER (gliner-deberta,
  gliner-lfm) is NOT covered:** it uses `ggml_gallocr` + `ggml_backend_graph_compute`,
  which has no eval-callback — the collector only attaches to a `ggml_backend_sched`.
  Collecting imatrix there needs routing GLiNER's compute through a sched when
  imatrix is active. **Reranker/NER A/B eval sets are small (n=5–6) → τ/F1 coarse;
  treat 4-bit-vs-q8 picks as provisional until a larger paired/labeled corpus.**
- **GLiNER imatrix needs a sched — the collector only hooks a `ggml_backend_sched`
  (2026-07-03).** GLiNER used `ggml_gallocr` + `ggml_backend_graph_compute` (no
  eval-callback), so the collector couldn't see its matmuls. Fix: build an opt-in
  sched (only when `CRISPEMBED_IMATRIX_OUT` is set) and route all compute sites
  through it via a small alloc/compute helper; keep the fast gallocr path otherwise
  (zero overhead in normal use). **Flush in `gliner_ner_free`** — GLiNER isn't freed
  via `crispembed_free`, so its imatrix would leak past clean_exit's `_exit`.
  General lesson: any engine with a *self-contained gallocr compute* is invisible to
  the collector until it runs through a sched.
- **gliner-lfm q4_k span-F1 0.94 was a coarse-metric artifact, not a bug.** Score-
  level diff showed uniform 2% quant shift with **max |Δscore| 0.031** (no outliers
  → no localized bug); the F1 dip was 3 detections scoring 0.50–0.51 crossing the
  0.5 threshold. Cross-check: the same LFM2 backbone scores 0.9975 on lfm2-colbert.
  **When a discrete/threshold metric (F1, τ) looks bad at small n, compare the
  continuous scores/vectors before blaming quant or a bug.**
- **Converter head detection must read the CHECKPOINT, not the loaded model
  (2026-07-03).** `convert-bert-to-gguf.py` tried AutoModelForTokenClassification
  before the MLM check, but HF **random-inits** a `classifier.weight` for a SPLADE
  model (config num_labels=2) — so `"classifier.weight" in state_dict` was True and
  SPLADE was mis-detected as a 2-label NER, dropping the real `cls.predictions.*`
  head. Every splade-pp GGUF shipped with no sparse head (functionally broken). Fix:
  decide the head from the **checkpoint files** (a real `classifier.weight`/`cls.
  predictions.*` there is authoritative) — real classifier wins (reranker/NER), else
  real MLM head → SPLADE, else embedder. HF silently invents missing heads, so the
  loaded `state_dict` lies for *any* head-detection heuristic.
- **SOTA permissive EN+DE eval corpora:** report scores against **MMTEB**
  (Apache-2.0 framework); for calibration/A/B *text* (no labels needed for the
  quant-vs-full-precision agreement metric) real-data options are **MIRACL**
  (Apache-2.0 card, but Wikipedia text underneath → effectively CC-BY-SA) and
  **Tatoeba** (CC-BY-2.0) EN–DE pairs; GermanQuAD/GermanDPR are CC-BY-4.0. Avoid
  XNLI/MultiNERD (NC), Flores/Wikipedia-derived (SA). **For a fully license-free
  bundle (usable under MIT/Apache/BSD-3), self-author the text and release it CC0**
  — that's what CrispEmbed ships (`tools/gen_eval_corpora.py`, EN+DE parallel pairs);
  quant-agreement A/B only needs diverse realistic multilingual text, so hand-written
  is as good as a benchmark here and carries no attribution burden. Practical trap:
  `datasets` 4.x dropped script datasets, so MIRACL/germandpr won't `load_dataset` —
  use their parquet mirrors or self-author.
- **imatrix calibration is ~language-agnostic — calibrate in EN, it generalizes to
  DE (2026-07-03).** Controlled A/B (`tools/kaggle/crispembed-calib-ab/`): quantize
  q4_k twice, imatrix from English-only vs English+German calibration, evaluate both
  on a German set. Deltas were **noise**: bge-m3 DE +0.0001 (EN −0.0007),
  xlmr-ner-hrl 1.0→1.0. **Why:** the imatrix is per-*column* sum-of-squared
  activations — which columns matter is set by the weights/architecture, not the
  calibration language — so a bilingual calibration corpus is NOT worth a
  re-calibration rollout. Use bilingual corpora for **A/B reporting** (they surface
  a real EN-vs-DE quality gap in the *model*), not for calibration. The A/B itself
  is the cheap way to check this before spending compute on any rollout.
- **Big models: calibrate on the q8_0, quantize from the f32 base.** A 4B/8B f32
  base (16/30 GB) can't be *loaded for inference* on Kaggle's ~13 GB RAM, which
  calibration needs. But the imatrix is **activation statistics**, and q8_0 is
  ~lossless (cos ~0.9998), so calibrating on the q8_0 (4.3/8 GB — fits RAM) yields
  essentially the same imatrix. Quantization itself is fine on the f32 base:
  `crispembed-quantize` reads it **tensor-by-tensor** (fseek+fread, no whole-model
  load), so RAM stays low regardless of file size. So the big-base recipe is:
  calibrate + A/B-gold on the q8_0, quantize from the f32 base, stage the big files
  in `/tmp` (~70 GB) not `/kaggle/working` (~20 GB, kaggle_usage.md #18). A/B is then
  cos-vs-q8 (the q8 is the practical gold). This avoids a CUDA build entirely — no GPU needed just to fit the model, since
  we never load the full-precision weights. **Caveat (measured):** quantizing
  from q8_0 instead of f32 is NOT free — on embeddinggemma-300m,
  cos(q4-from-q8, q4-from-f32) = **0.991** (double-quantization costs ~0.009).
  The resulting q4 is still good (~0.978 vs f32), and for 4B/8B models whose
  f32 base won't download (stalls at ~4 MB/s on Kaggle) it's the only viable
  path — but don't claim q8-source ≈ f32-source. `crispembed-quantize`
  dequantizes quantized sources (ggml `to_float`) before re-quantizing.

### Metal mul_mm F16 kernel selection (why set_prec doesn't help)

`ggml/src/ggml-metal/` picks the matmul kernel by **operand type + shape**, never
a precision flag: `mul_mv` (vector, small `ne11`) vs **`mul_mm` (simdgroup GEMM)
when `ne11 > 8`**. `mul_mm` casts activations to half → values >65504 → Inf/NaN
(our ×12 image-embed overflow). `ggml_mul_mat_set_prec(GGML_PREC_F32)` is a no-op
for Metal GEMM (only consulted for `flash_attn_ext`). Fix = scale ×1/256 before,
×256 after (memory `metal-mul-mm-f16-overflow`); same bug class on Vulkan (#18969).
Diagnostic: NaN only with many tokens, clean single-token ⇒ you're on `mul_mm`.
Confirm Metal is active (not CPU fallback): stderr `using embedded metal library`
+ `using device Metal`.

### Flash attention contract (cross-ref)

See `## ggml_flash_attn_ext accepts non-contiguous Q/K/V` and memory
`flashattn-ext-already-permutes`: output is `[hd, nh, T]` — already permuted —
reshape directly, **never** add a trailing `permute(0,2,1,3)` (that was the
`6027b56` RT-DETR crater). GQA: `n_head % n_head_kv == 0`, don't repeat K/V
(broadcast is internal). Pad the mask token dim to `GGML_KQ_MASK_PAD` (64 on
master, historically 32; verify against pinned ggml); padded rows must be `-INF`.

### GGUF / tokenizer conventions to align with (cross-ref)

See `## llama.cpp GGUF ≠ CrispEmbed GGUF`. Additionally: honor `general.alignment`
(default **32**), namespaced `{arch}.*` hparams, and tokenizer behavior flags
`add_bos_token`/`add_eos_token` (encode our LFM2 BOS-only rule as data, not code).
Use the `mmproj-*.gguf` sidecar convention for vision projectors so downstream
tooling recognizes them. (Landmine reminder: read every metadata string/array
*before* `gguf_free`.)

### What llama.cpp does NOT have that we do (differentiation)

MPNet, GTE-v1.5 (`NewModel`), DeBERTa-v2, SPLADE, bge-m3 sparse+ColBERT tri-head,
standalone CLIP/SigLIP text **and** image embeddings, GOT-OCR2, all specialized
math OCR (pix2tex/TrOCR/HMER/BTTR/PosFormer/MixTex/PP-FormulaNet/PARSeq/Tesseract/
Pix2Struct), and the entire face / detection-layout / NER-KIE / LID / punctuation
/ image-restoration surface. Only ESRGAN/RRDBNet exists in ggml (via
`stable-diffusion.cpp`), not the other SR models. These are genuine moat — keep
their guardrails green.

## Quantization notes

### Python gguf vs C++ quantizer

The Python `gguf` library (`pip install gguf`) only implements quantization
for basic types: Q4_0, Q5_0, Q5_1, Q8_0. K-quants (Q4_K, Q5_K, Q6_K) are
listed in the enum but `quantize_blocks` raises `NotImplementedError`.

Additionally, the Python library's string array handling in GGUFReader/GGUFWriter
can corrupt metadata when copying GGUF files — we observed Q8_0 models from the
Python quantizer producing cos=0.78 vs the same model's F32, while the C++ quantizer
produces cos=0.9997.

**Use the C++ quantizer for all quantization.** It calls ggml's native
`ggml_quantize_chunk` which supports all types including K-quants.

### Embedding tables and aggressive quantization

Token embedding tables (`token_embd.weight`) are very sensitive to quantization.
Quantizing them to Q4_K degrades output quality significantly (cos drops from
0.999 to 0.71 for some models). The CrispEmbed quantizer keeps embedding tables
at F32 for Q4_K/Q5_K; only Q8_0 and F16 are allowed to touch them.

### K-quant fallback chain

K-quants (Q4_K/Q5_K/Q6_K) require row widths divisible by 256. Many embedding
model tensors have rows of 384 or 768 which aren't 256-aligned. The quantizer
falls back: Q4_K→Q4_0, Q5_K→Q5_0, Q6_K→Q8_0. This means small-dim models
get Q4_0 instead of Q4_K for most tensors.

### ggml_get_rows for quantized embeddings

The BERT encoder must use `ggml_get_rows` (ggml graph op) for embedding table
lookup, not manual `ggml_backend_tensor_get` with float pointer arithmetic.
`ggml_get_rows` handles dequantization internally and works with any tensor type.
Manual CPU-side extraction assumes F32 layout and crashes on quantized models.

## Server performance: buffer reuse

The biggest server-mode optimization is reusing `graph_buf` and `work_buf` across
encode calls. Without this, every request allocates ~50-200MB (graph context +
compute workspace), causing 3x overhead from malloc/free.

With buffer reuse: gte-small goes from 8.8 to 27.8 texts/sec (3.2x improvement).

## BLAS/MKL for embedding models

BLAS (OpenBLAS/MKL) provides minimal benefit for embedding inference because:
- Quantized kernels (Q8_0/Q4_K) use ggml's SIMD paths, not BLAS
- BERT encoder matrices are moderate-sized (384x384 to 1024x4096)
- BLAS overhead dominates for small matrices

For CPU speed: use Q8_0 quantization. For GPU: build with `-DGGML_CUDA=ON` or
`-DGGML_VULKAN=ON` — the `ggml_backend_sched` dispatcher handles offloading.

## ggml_backend_sched with CPU-only

When using `ggml_backend_sched` in CPU-only mode, calling it repeatedly with
different graphs causes segfaults because the scheduler holds stale tensor
references from freed graph contexts. Solution: only create the scheduler when
a GPU backend is detected (`!ggml_backend_is_cpu(backend)`). For CPU-only,
direct `ggml_graph_compute` with a persistent work buffer is faster anyway.

## Matmul optimization — what we use, what's available

### Current state (as of April 2026)

Our embedding models have small matrices: 384×384 (MiniLM/GTE) to 1024×4096
(Qwen3 FFN). For these sizes, overhead per matmul call matters more than
raw FLOP throughput.

### CPU matmul options (ggml-cpu)

| Option | Default | Effect | Our impact |
|--------|---------|--------|-----------|
| `GGML_LLAMAFILE` | OFF | Custom SGEMM kernels optimized for small F32 matmul | **HIGH** for F32 models |
| `GGML_AVX512` | OFF | 512-bit SIMD (2x wider than AVX2) | **HIGH** if CPU supports |
| `GGML_AVX512_VNNI` | OFF | Hardware int8 dot products | Medium for Q8_0 models |
| `GGML_AMX_TILE` | OFF | Intel AMX for int8/BF16 (Sapphire Rapids+) | None (needs new CPU) |
| `GGML_OPENMP` | ON | Thread parallelism | Already enabled |

**Enable for best CPU performance (LOCAL builds only):**
```bash
cmake -S . -B build -DGGML_LLAMAFILE=ON   # custom SGEMM
cmake -S . -B build -DGGML_AVX512=ON      # if CPU supports (check /proc/cpuinfo)
```

> Never do this for a binary you ship — see the next section.

## Exit 127 with no output = the dynamic loader, not your code

SubtitleEdit#13205: `crispembed-server` from the Linux tarball exited **127**
and printed nothing, on EndeavourOS and on Linux Lite. 127 from a
dynamically-linked binary means the loader could not resolve a `DT_NEEDED`
library, so the process dies before `main()` — no log line of ours can ever
appear. The chain was:

```
crispembed-server -> libggml.so.0      (bundled, RUNPATH $ORIGIN)
libggml.so.0      -> libggml-blas.so.0 (bundled)
libggml-blas.so.0 -> libopenblas.so.0  <-- not bundled, not installed by default
```

`-DGGML_BLAS=ON` on the release legs put a hard link-time dependency on
OpenBLAS into every Linux artifact, x86_64 and arm64 alike. The workflow
apt-installed `libopenblas-dev` so the CMake BLAS probe would succeed, which
means **the runner always had it and the artifact never did** — the failure was
structurally unreachable from CI. Same shape as [#41]: a property of the build
environment leaking into the shipped artifact.

Three things worth carrying:

1. **It bought nothing.** `PERFORMANCE.md` "BLAS Acceleration" measures OpenBLAS
   at 0.9–1.0x on these models, and the section above already says BLAS is
   minimal-benefit because quantized kernels use ggml's SIMD paths. The release
   comment justifying it ("big matmuls of large encoders") was never measured.
   A dependency that costs a whole platform must at least be earning something.
2. **Verify the package, not the build.** `ldd`-equivalent inspection of the
   staged `pkg/` is the only check that can see this;
   `scripts/check-bundled-deps.py` parses each ELF's dynamic section and fails
   if a `DT_NEEDED` is neither bundled nor part of a base glibc system. It
   deliberately does not treat `libgomp.so.1` as base-system.
3. **A release you cannot dry-run is a release you cannot check.** `release.yml`
   only triggered on tag pushes, so the only way to see a packaged artifact was
   to publish one. It now also takes `workflow_dispatch`, with every publish
   step guarded on `refs/tags/`.

Related but distinct, and still open: the tarballs are built on Ubuntu 24.04 so
they need **glibc ≥ 2.38 / GLIBCXX ≥ 3.4.32** and will not start on Ubuntu
22.04 or Debian 12 even with OpenBLAS present. Already recorded for the wheels
in `python/pyproject.toml`; the fix for both is to build inside a manylinux
container.

## The build environment is part of the artifact

Three separate outages this session traced to the same root: a property of the
machine that did the build silently became a requirement of the thing shipped,
and CI could not see it because CI *was* that machine.

| what leaked in | symptom for the user | why CI was blind |
|---|---|---|
| runner's CPU had AVX-512 | `Illegal instruction` (#41) | no runner lacks it |
| runner had `libopenblas-dev` | `exit 127`, no output (SubtitleEdit#13205) | the workflow installed it so the probe would pass |
| runner's glibc was 2.38 | won't start on Ubuntu 22.04 (#42) | every runner is newer |

The shape is always the same, so the defence is too: **inspect the package, not
the build.** `scripts/check-cpu-baseline.py` reads the configured ISA and the
generated compile lines; `scripts/check-bundled-deps.py` parses each ELF's
dynamic section for unbundled `DT_NEEDED` entries and the glibc floor. Both run
at packaging time and fail the release. Neither is a test — a test would have
to run on hardware we do not have.

Corollaries worth keeping:

- **A dependency that costs you a platform must be earning something.** The
  OpenBLAS that made every Linux archive unlaunchable measured 0.9–1.0x here.
- **`|| true` on a copy is how a licence file silently stops shipping.** The
  bundled CUDA archive claimed an `NVIDIA-EULA.txt` it did not contain, because
  `cp "$CUDA_PATH/EULA.txt" … || true` swallowed the miss for a file the
  sub-package install never lays down.
- **Verify the real payload, not a copy you staged to verify with.** A check of
  mine failed on its own `find -type f` staging (which drops SONAME symlinks)
  while the artifact was fine.
- **`DT_RUNPATH` is not transitive.** Bundled libs each need their own
  `$ORIGIN`; a sibling does not inherit the caller's.
- **A release you cannot dry-run is a release you cannot check.** `release.yml`
  only triggered on tags, so the only way to see a packaged artifact was to
  publish one. It now takes `workflow_dispatch` with every publish step guarded
  on `refs/tags/`.

## GGML_NATIVE probes the BUILD machine — never ship a native build

`GGML_NATIVE` defaults to ON and is *not* a compile-time-only heuristic: it
**executes probe programs on the build machine**.

- MSVC: ggml includes `ggml-cpu/cmake/FindSIMD.cmake`, which uses
  `check_c_source_runs` to run an AVX-512 test binary. If it exits 0, the whole
  CPU backend is compiled `/arch:AVX512`.
- GCC/Clang: `-march=native`.
- ARM: `-mcpu=native` plus `check_cxx_source_runs` probes for
  `dotprod` / `i8mm` / `sve` / `sme`.

So the ISA of the artifact is a property of **whichever machine built it**. On a
heterogeneous CI fleet that is a coin flip. Issue #41: GitHub's `windows-latest`
pool mixes AVX-512-capable Intel hosts with AVX2-only AMD hosts, so v0.16.1's
`crispembed-windows-x86_64.zip` (cpu) shipped `/arch:AVX512` and died with
`Illegal instruction` on an i9-14900KF right after tokenizer load — while the
previous release, built on an AMD runner, was fine on the same machine, and so
was the cuda zip, the one leg that already pinned `-DGGML_NATIVE=OFF`.

Two traps that made this hard to see:

1. **The failure never reproduces in CI** — no runner lacks the extension the
   runner had. Only an end user finds it.
2. **`CMakeCache.txt` lies when NATIVE is ON.** `FindSIMD.cmake` sets
   `GGML_AVX512` as a *normal* variable, which shadows the cache entry. The
   cache can read `GGML_AVX512:BOOL=OFF` while the compile line says
   `/arch:AVX512`. Verify against the generated build files (`build.ninja`,
   `*.vcxproj`, `flags.make`), not the cache.

**The rule:** every workflow producing a redistributable artifact passes
`-DGGML_NATIVE=OFF` and then runs `scripts/check-cpu-baseline.py build`, which
checks both the cache options and the generated compile lines.
`CRISPEMBED_NATIVE` (which drives `-march=native` on CrispEmbed's own targets,
for the `cpu_ops.h` intrinsics) defaults to `GGML_NATIVE`, so one flag makes the
whole tree portable — setting only one of the two leaves half the binary tuned
for the builder's CPU, which is precisely how this shipped.

Shipped baselines are documented in the README ("CPU requirements &
redistributable builds"). Longer term, `GGML_CPU_ALL_VARIANTS` +
`GGML_BACKEND_DL` would give true runtime dispatch, but it requires the CPU
backend to be loaded as a DLL, and CrispEmbed calls `ggml_backend_cpu_*`
directly in ~200 places — a real refactor, not a flag flip.

### CUDA matmul options

| Option | Default | Effect |
|--------|---------|--------|
| `GGML_CUDA_FA` | ON | Flash attention CUDA kernel |
| `GGML_CUDA_GRAPHS` | OFF | Multi-op fusion via CUDA graph capture |
| `GGML_CUDA_FORCE_MMQ` | OFF | Force quantized matmul kernels (vs cuBLAS) |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF | Flash attn for all quant types |

CUDA auto-selects between MMQ (quantized matmul) and cuBLAS (F32) based
on matrix size and GPU compute capability. For our 384×384 Q8_0 matrices,
MMQ is usually selected (faster than cuBLAS for small quantized matmul).

### Why HF PyTorch is still competitive on CUDA

HF PyTorch uses cuBLAS with operator fusion via torch.compile/TorchScript.
For a 22M-param model (MiniLM), the GPU is underutilized — compute time
is dominated by kernel launch overhead and memory transfers, not FLOP
throughput. Both HF and CrispEmbed run at ~10ms, limited by the GPU's
minimum latency per kernel launch (~5μs × ~200 kernels = ~1ms overhead).

### Batched matmul on GPU

Single matmul `W[H,H] × X[H, T*B]` is much faster than B separate
`W[H,H] × X[H, T]` calls because:
1. One cuBLAS/MMQ launch vs B launches
2. Better GPU occupancy (more work per SM)
3. Memory access amortization

Our true batched graph concatenates all texts and uses 4D flash attention
with batch dimension. The matmuls naturally batch via the flattened T*B dim.

### QKV weight fusion

Pre-merging Q/K/V weight matrices into `[H, 3H]` reduces 3 matmul calls
to 1 per layer. The merged tensor must live in the same backend buffer as
the model weights (ggml_backend_alloc_ctx_tensors) so it works on GPU.

On CPU: ~0.5ms savings (15.3ms vs 16.8ms for MiniLM).
On GPU: minor savings (kernel launch overhead reduction).

## Optimization experiment results (April 2026)

| Optimization | CPU Impact | GPU Impact | Verdict |
|---|---|---|---|
| QKV weight fusion (1 matmul vs 3) | 15.3ms vs 17.0ms (**+11%**) | minor | **Keep** — matmul reduction wins |
| Flash attention (fused QKV attn) | 16.8→15.3ms | significant | **Keep** |
| Scheduler reservation (bucket T) | no change | may help | Keep (no cost) |
| GGML_LLAMAFILE | 15.3→14.7ms (**+4%**) | N/A | **Enable by default** |
| AVX512 (if CPU supports) | 15.3→14.4ms (**+6%**) | N/A | Enable if available |
| F16 model weights | 15.3→17.7ms (**-14%**) | may help (tensor cores) | **Skip on CPU** |
| Removing ggml_cont (no QKV fusion) | 15.3→17.0ms (**-10%**) | N/A | Don't remove |
| True batched graph (4D flash attn) | slower on CPU | should help | GPU only |

### Why we can't easily match HF PyTorch

1. **Graph rebuild cost**: ggml rebuilds the graph from scratch every call (~1ms).
   PyTorch JIT-compiles and caches the execution plan.
2. **No CPU operator fusion**: ggml CPU executes each op separately (separate memory pass
   for norm, mul, add). ORT/PyTorch fuse these into single kernels.
3. **No persistent CUDA graphs**: PyTorch can capture and replay GPU command streams.
   ggml has `GGML_CUDA_GRAPHS` but it's designed for llama.cpp's specific graph topology.
4. **Batch matmul**: PyTorch's cuBLAS wrapper handles batched matmul natively.
   Our 4D reshape + flash attention adds overhead vs native batch support.

### Practical CPU performance ceiling

For MiniLM (22M params, 6 layers, 384d) on 4-thread CPU:
- **15.3ms** with all optimizations (QKV fusion + flash attn + llamafile)
- **~14ms** theoretical minimum (pure matmul compute time)
- **~1ms** graph rebuild overhead we can't eliminate
- HF PyTorch on same CPU: **54ms** (CrispEmbed is **3.5x faster on CPU**)

### Practical GPU performance ceiling

For MiniLM on RTX A1000 (budget laptop GPU):
- **10.6ms** current (with all optimizations)
- **~5ms** theoretical minimum (kernel launch overhead + small matrix underutilization)
- HF PyTorch: **9.5ms** (they have better GPU batching)
- Gap is ~1ms — likely kernel launch overhead from ggml's per-op dispatch

## Windows build

Windows users often forget `--recursive` when cloning. The CMakeLists.txt now
checks for `ggml/CMakeLists.txt` existence and prints a helpful error message.
Build scripts (`build-windows.bat`, `build-vulkan.bat`, `build-cuda.bat`) auto-
detect VS2022 and Vulkan/CUDA SDKs.

## ggml operator fusion — what exists, what doesn't

### Existing fused ops (backend-specific)

**CUDA** (automatic when graph patterns match):
- RMSNorm + Mul (`ggml_cuda_op_rms_norm_fused`)
- RMSNorm + Mul + Add (`ggml_cuda_op_rms_norm_fused_add`)
- Multi-Add (up to 8 chained adds → 1 kernel)
- FFN gate: MUL_MAT + ADD + MUL_MAT + ADD + GLU → 1 kernel
- RoPE + SetRows fused
- Unary + Mul (SILU/Sigmoid/Softplus)

**Vulkan**: Add + RMSNorm (controlled by `GGML_VK_DISABLE_FUSION`)
**Metal**: Generic fusion framework with `use_fusion` flag
**CPU**: **No fusion at all** — every op executes individually

### What this means for performance

On **CPU**, there's a fundamental ~3x gap vs ONNX Runtime because:
1. ORT does Level3 graph JIT compilation: constant folding, op fusion, layout
   optimization, kernel selection — all at graph compile time
2. ggml has no graph optimization pass; fusion only happens in GPU backends
   during compute, not at graph construction time
3. Each ggml CPU op does a separate memory pass (read+write). Fusing
   LayerNorm (norm+mul+add = 3 passes) into 1 pass saves bandwidth

On **GPU (CUDA)**, the gap should be much smaller because:
1. CUDA backend automatically fuses RMSNorm+Mul, FFN gates, multi-add
2. `ggml_flash_attn_ext` runs as a single fused CUDA kernel
3. Matmul uses cuBLAS (same as PyTorch/ONNX)
4. Memory bandwidth is 10-20x higher on GPU, so fusion matters less

### What we optimized (practical CPU-side)

1. **Pre-merged QKV weights**: concatenate Q/K/V weight matrices into one
   [H, 3H] tensor at load time. One matmul instead of three per layer.
   Saves ~0.5ms for 6-layer 384d model.

2. **Flash attention**: `ggml_flash_attn_ext` replaces 8 separate ops
   (permute, cont, mul_mat, scale, softmax, mul_mat, permute, reshape)

3. **Graph caching**: build ggml graph once per sequence length, reuse
   across calls. Eliminates ~3ms of ggml_init + graph construction.

4. **Buffer reuse**: graph_buf and work_buf persist across calls.

### Why not modify ggml for CPU fusion?

Considered but impractical because:
- ggml's CPU backend is designed for portability (pure C + SIMD intrinsics)
- Adding a graph optimization pass would affect all ggml users
- The `ggml_map_custom` API allows custom kernels but doesn't help with
  matmul (the expensive op) — ggml's SIMD matmul is already well-optimized
- Fusing norm+mul+add saves < 0.1ms per text (memory-bound, not compute-bound)
- The 3x gap to ONNX is dominated by ORT's matmul scheduling and cache
  optimization, not by op fusion per se

### GPU prediction

On CUDA, CrispEmbed should match or beat ONNX because:
- cuBLAS matmul is the same engine ORT uses
- ggml's CUDA fusion handles the same patterns ORT fuses
- Flash attention is implemented as a single CUDA kernel
- No Python/ONNX overhead in our C++ server

Estimated GPU performance for MiniLM (RTX 3060):
- CrispEmbed CUDA: ~2-4ms (model fits entirely in GPU memory)
- fastembed ONNX+CUDA: ~2-4ms (cuBLAS + graph optimization)
- Likely on par, with CrispEmbed winning on server overhead

## Prompt prefix system for RAG models

Many embedding models require query/passage prefixes for optimal retrieval:
- BGE: `"Represent this sentence for searching relevant passages: "`
- E5: `"query: "` / `"passage: "`
- Nomic: `"search_query: "` / `"search_document: "`
- Jina v5: `"Query: "` / `"Document: "`

Implementation: prefix is stored in `crispembed_context::prefix` and prepended
to the raw text before tokenization in both `crispembed_encode()` and
`crispembed_encode_batch()`. This is correct because:
1. The prefix is part of the semantic input (not a tokenizer-level construct)
2. All tokenizer types (WordPiece/SentencePiece/BPE) handle it naturally
3. fastembed-rs does the same (injects prefix before tokenizer.encode)

**Not applied to sparse/colbert/reranker**: These have different input semantics.
Sparse retrieval operates on raw terms. Rerankers take (query, document) pairs
where the model handles the joint encoding.

## Bi-encoder vs cross-encoder reranking

Both approaches are valuable for RAG and complement each other:

**Bi-encoder** (embed query + docs independently, cosine similarity):
- Fast: encode once, compare N documents with dot products
- Same model used for initial retrieval AND reranking
- Quality limited by the embedding space
- CrispEmbed: `rerank_biencoder()` in Python/Rust, uses `encode_batch()` + dot product

**Cross-encoder** (encode query-document pairs jointly):
- Slow: each (query, doc) pair requires a full forward pass
- Much higher quality (joint attention between query and document tokens)
- Typically used as second-stage reranker after bi-encoder retrieval
- CrispEmbed: `rerank()` in Python/Rust, uses `crispembed_rerank()` C API

**RAG pipeline pattern**: bi-encoder retrieval (top-100) → cross-encoder reranking (top-10)

## Model registry for RAG feature parity

When adding new models to the registry (`model_mgr.cpp`), the key metadata is:
- **name**: short name for CLI/auto-download
- **filename**: GGUF filename (may include `-q8_0` suffix for default quant)
- **url**: HuggingFace direct download URL under `cstr/` namespace
- **desc**: architecture, dimension, language, parameter count

Models that are encoder-only (BERT/XLM-R) use the existing convert-bert-to-gguf.py.
Models that are decoder-based (Qwen3/Gemma3) use convert-decoder-embed-to-gguf.py.
Rerankers are encoder models with a classifier head — use `--crisp` flag to include
the classifier weights in the GGUF.

## MPNet relative position bias

MPNet uses T5-style relative position bias instead of absolute position embeddings.
The bias is a learned `Embedding(32, 12)` — 32 logarithmic distance buckets × 12
attention heads. For each (query_pos, key_pos) pair, a bucket index is computed
via logarithmic distance binning, then the bias is looked up and added to
attention scores before softmax.

**Our implementation** (CrispEmbed):
- Precompute the full `[T, T, n_heads]` bias matrix in C++ at encode time
- Pass it as the F16 mask parameter to `ggml_flash_attn_ext`
- Flash attention adds it to scores natively — no manual attention needed
- Result: cos=0.999997 vs HuggingFace

**llama.cpp approach** (PR #21880):
- Compute bucket indices in the ggml graph via `build_inp_pos_bucket_enc()`
- Look up bias weights with `build_pos_bias()` (ggml graph ops)
- Pass as `kq_b` to `build_attn()` which adds it to attention scores
- Tensor stored transposed `[n_heads, n_buckets]` on layer 0

**Key difference**: We precompute in C++ (simpler, works on CPU), they compute in
the ggml graph (GPU-accelerable, more modular). Both produce identical results.
Our approach is ~10 lines of C++ vs their ~50 lines of graph builder code.

**Bugs found during MPNet implementation**:
- Python `or` operator treats `cls_token_id=0` as falsy → falls through to
  default 101. Fix: use `is not None` check
- MPNet needs position offset = 2 (same as RoBERTa), but `model_type="mpnet"`
  was not included in the offset detection

## Reranker model conversion notes

Cross-encoder rerankers (bge-reranker, ms-marco-MiniLM, mxbai-rerank) have a
classifier head on top of the encoder:
- **1-layer**: `classifier.dense.weight [H,1]` + `classifier.dense.bias [1]`
  → CLS hidden → Linear → scalar score
- **2-layer** (RobertaClassificationHead): `classifier.dense.weight [H,H]` +
  `classifier.out_proj.weight [1,H]` + biases
  → CLS hidden → Linear → tanh → Linear → scalar score

The converter must include these weights. Detection: `crispembed.is_reranker`
is set based on presence of `classifier.dense.weight` in the GGUF.

Some rerankers (ms-marco-MiniLM) use `num_labels=1` with no activation,
while others (bge-reranker) use sigmoid/softmax. CrispEmbed returns the raw
logit — the caller decides on thresholding.

## ModernBERT architecture (pre-LN)

ModernBERT (gte-modernbert-base, modernbert-embed-large) uses **pre-LayerNorm**
ordering, which differs from standard BERT's post-LN:

**Post-LN (BERT/XLM-R/MPNet):**
```
attn(input) → residual_add(input) → LN → FFN → residual_add → LN
```

**Pre-LN (ModernBERT):**
```
LN(input) → attn → residual_add(input) → LN → FFN → residual_add
```

Pre-LN has the LayerNorm *before* each sub-layer, with the residual connection
bypassing the norm. This is the same as GPT-2/LLaMA-style normalization.

Detection: `bert.pre_ln` GGUF metadata flag. Combined with:
- GeGLU activation (GELU-gated FFN instead of SwiGLU)
- RoPE (no position embeddings)
- No biases on attention or FFN
- Fused QKV weights

ModernBERT is essentially a bidirectional LLaMA with GELU instead of SiLU.
CrispASR has a reference implementation in `examples/talk-llama/models/modern-bert.cpp`.

### ModernBERT debugging: cos 0.69 → 0.97

Two bugs caused cos=0.69 across 22 layers (1-layer was 0.999):

**Bug 1: Wrong SEP token.** The BPE merge re-loading after tensor init
was calling `load(vocab, merges, eos_id=sep_id, pad_id, suffix_id=unk_id=3, ...)`
instead of `suffix_id=-1`. This made the tokenizer append token 3 (unk)
instead of 50282 (SEP). The wrong token propagated through all 22 layers
of the transformer, compounding the error.

Lesson: when re-initializing a tokenizer after loading merges, preserve
ALL original parameters — don't substitute defaults for parameters that
were carefully set during the first init.

**Bug 2: Separate GELU+MUL vs fused ggml_geglu.** Our code used:
```cpp
up = matmul(fc1_w, cur);     // [inter, T]
gate = matmul(ffn_gate_w, cur); // [inter, T]
up = gelu(up);
ffn = mul(up, gate);
```

llama.cpp uses:
```cpp
up_gate = matmul(ffn_up_gate_w, cur); // [2*inter, T]
ffn = ggml_geglu(up_gate);           // fused: gelu(first_half) * second_half
```

The fused `ggml_geglu` is a single ggml operation that avoids intermediate
rounding between the GELU and multiply. With 22 layers × ~1000 intermediate
dimensions, the accumulated rounding difference is significant for pre-LN
models (where residual connections pass raw values without normalization reset).

Fix: store the original fused `Wi` / `up_gate_proj` weight in the GGUF
and use `ggml_geglu` instead of separate ops. Also use `ggml_swiglu` for
NomicBERT-style SwiGLU.

**Why post-LN models don't have this problem:** In post-LN models (BERT),
LayerNorm after each residual add normalizes the hidden state to unit
variance. This effectively "resets" any accumulated floating-point drift.
In pre-LN models, the raw residual passes directly to the next layer,
allowing small per-layer errors (~0.001) to compound nonlinearly.

**Per-layer theta:** ModernBERT alternates sliding (theta=10000) and global
(theta=160000) attention. For encoding (not generation), sliding window
masking is NOT applied — confirmed by llama.cpp's `build_attn_inp_no_cache()`.

## Head-to-head benchmark: CrispEmbed vs FastEmbed

**MiniLM-L6 (6 layers, 384d)**: CrispEmbed is **9.5x faster** on single text
and **10.8x faster** on batch. This is our best-optimized model: QKV fusion
reduces 3 matmuls to 1 per layer, flash attention replaces 8 separate ops,
and graph caching eliminates rebuild overhead.

**BGE-small (12 layers, 384d)**: FastEmbed is **1.7x faster**. ONNX Runtime's
Level3 graph JIT compilation (operator fusion, layout optimization, cache-aware
scheduling) gives it an edge on 12-layer models. Our per-op execution on CPU
has higher overhead per layer.

**Arctic-M (12 layers, 768d)**: Tied on batch (126 vs 127ms). As hidden size
grows, matmul compute dominates over per-op overhead, equalizing performance.

**Conclusion**: CrispEmbed wins decisively on small models (6 layers) where
per-op overhead matters most. On larger models, ONNX Runtime's graph optimization
closes the gap. GPU (CUDA/Metal) should favor CrispEmbed across all sizes due
to ggml's fused CUDA kernels and flash attention.

## DeBERTa-v2 disentangled attention (full parity)

DeBERTa-v2's attention computes three components, all now implemented:
1. **c2c** (content-to-content): standard Q×K^T
2. **c2p** (content-to-position): Q × K_proj(rel_embd)^T
3. **p2c** (position-to-content): K × Q_proj(rel_embd)^T

### Key implementation details

**Pre-expansion approach**: Rather than gather+matmul at runtime, we pre-expand
the position embeddings on CPU: `P[H, T*T]` where `P[:, i*T+j] = LN(rel_emb[bucket(i-j)+256])`.
Then project through K/Q weights and use batched matmul to compute all scores.

**Critical: HF uses bucket(query-key) for BOTH c2p AND p2c**. This is
counter-intuitive — you'd expect p2c to use bucket(key-query). But HF's
`disentangled_attention_bias` gathers p2c using the same relative position
index, then transposes the result. To achieve this with pre-expansion, we
transpose the T×T grid for p2c: `P_p2c = P.reshape(H,T,T).permute(0,2,1)`.

**Encoder LayerNorm on position embeddings**: HF applies `encoder.LayerNorm`
to `rel_embeddings.weight` BEFORE using them in attention (`get_rel_embedding()`).
This is separate from the post-encoder LayerNorm. Missing this causes ~15%
error in position scores.

**Position projection biases**: HF's `key_proj`/`query_proj` are `nn.Linear`
which include bias. Must add `k_bias` to Pk and `q_bias` to Pq.

**Log-bucket formula** (`make_log_bucket_position`): Uses signed bucket values
centered at `att_span` (= position_buckets = 256). The log denominator is
`log((max_relative_positions - 1) / mid)`, NOT `log((max_pos/2 - 1) / mid)`.

**Attention output reshape**: After V-weighted sum `[hd, T_q, nh]`, must permute
to `[hd, nh, T_q]` BEFORE reshaping to `[H, T]`. Without this permute, head
dimensions get incorrectly interleaved.

**Score scaling**: `1/sqrt(3 * head_dim)` when both c2p and p2c are present
(the 3 = 1 + num_position_attention_types).

### ggml_permute semantics (output-position convention)

`ggml_permute(a, ax0, ax1, ax2, ax3)`: `axes[k]` means "source dimension k
goes to result dimension `axes[k]`". So `permute(a, 0, 2, 1, 3)` on
`[hd, nh, T, B]` gives `[hd, T, nh, B]` (dims 1 and 2 swap).

This is the OPPOSITE of numpy's `transpose` where you specify source→result.

## Rust crate verification

The CrispEmbed Rust crate (`crispembed/`) wraps the C API via `crispembed-sys`
(cmake build.rs). Verified features:
- Dense encode (384d, correct values match Python)
- Batch encode (3 vectors, correct)
- Prefix set/get
- Matryoshka truncation (128d from 384d)
- Bi-encoder reranking (correct ordering)
- Capability queries (has_sparse, has_colbert, is_reranker)

The crate links dynamically (`dylib=crispembed`). Set `LD_LIBRARY_PATH` to the
build output directory. Static linking would avoid this but requires listing
all ggml dependencies in build.rs.

## BidirLM-Omni: 3D interleaved MRoPE via ggml IMROPE

HF `BidirLMOmniTextRotaryEmbedding.apply_interleaved_mrope` builds a per-token
`freqs_t` of length `head_dim/2` from three position channels `(t, h, w)` and
the configured `mrope_section = [s_t, s_h, s_w]` (default `[24, 20, 20]`):

- Start with `freqs_t = freqs[t]` (channel 0 across the entire vector).
- Replace indices `slice(1, 3*s_h, 3)` with `freqs[h]` at those indices.
- Replace indices `slice(2, 3*s_w, 3)` with `freqs[w]` at those indices.
- Anything beyond `3*s_h` (resp. `3*s_w`) stays in the t-channel.

For `[24, 20, 20]` and `head_dim=128` (so 64 cos/sin pairs), this produces:
T at positions 0, 3, …, 60, 63; H at 1, 4, …, 58; W at 2, 5, …, 59; T at 61–63
beyond the H/W slice ends.

ggml's `GGML_ROPE_TYPE_IMROPE` takes 4-channel positions `(t, h, w, e)` and
sections `[s_t, s_h, s_w, s_e]`. Its sector check is:

- `sector%3==0 && sector < 3*s_t` → `theta_t`
- `sector%3==1 && sector < 3*s_h` → `theta_h`
- `sector%3==2 && sector < 3*s_w` → `theta_w`
- otherwise → `theta_e`

For sections `[24, 20, 20, 0]` ggml routes sectors 61 and 62 to `theta_e`,
whereas HF leaves them on the T channel. The fix is to **pin `pos_e = pos_t`
per-token**: with that, `theta_e == theta_t` numerically at every sector and
the ggml IMROPE output matches HF byte-for-byte. The position tensor passed
to `ggml_rope_multi` therefore has shape `(4*T,)` laid out as
`[pos_t, pos_h, pos_w, pos_t]` (the tail mirrors the head).

For text-only inputs the three channels are all equal, so MRoPE collapses to
plain NEOX RoPE — `decoder_embed.cpp` keeps using `ggml_rope_ext` on the
text-only path to stay bit-identical with the pre-Phase-3 baseline tests.

## BidirLM-Omni: decoder scheduler init was missing

Before Phase 3 the decoder branch in `crispembed_init` never created a
`ggml_backend_sched` or sized `compute_meta` — those were only set up by
`load_model()` on the encoder branch. `decoder_encode_tokens` checks
`(sched != nullptr && compute_meta != nullptr)` and falls back to direct
`ggml_graph_compute` when either is null, so BidirLM-Omni text and audio
were silently running CPU-only on Metal builds.

Fix: in the decoder branch, after `load_decoder_model`, allocate

```cpp
const int graph_nodes = std::max(4096, ctx->dec->n_layer * 50 + 256);
ctx->sched = ggml_backend_sched_new(...);
ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes
                       + ggml_graph_overhead_custom(graph_nodes, false));
```

The `4096` floor is important: with image-conditioned text the graph adds an
input mask + patch (2 ops), per-layer DeepStack adds (n_ds ops), and
`ggml_rope_multi` instead of `ggml_rope_ext` (no node-count delta but extra
per-tensor metadata). 28 layers × ~50 ops ≈ 1400 still fits, but the floor
keeps headroom for future architectural growth and avoids surprising
allocation failures. Verify with `--save-baseline` / `--compare-baseline` in
`tests/benchmark_bidirlm.py` — text-only cosine should remain ≥ 0.99999
against the baseline taken before this change.

## BidirLM-Omni: parity reference dtype matters

When validating a quantized GGUF against a HuggingFace reference, the
**reference dtype** is part of the comparison and silently shifts the
upper bound. BidirLM-Omni-2.5B-Embedding ships its `model.safetensors`
in bf16 — that's the dtype the model was trained in. Loading it into
torch and calling `.to(torch.float32)` doesn't reconstruct any
pre-bf16 information; it just zero-pads the mantissa. So a cosine of
~0.94 vs HF fp32 is two distinct quantization steps stacked (bf16
trained → q4_k storage, then bf16 → fp32 upcast for the reference),
not "the q4_k is broken."

The fix in `tests/test_bidirlm_image_text.py`: the reference dtype is
a `--ref-dtype` flag, defaulting to bf16. Match the trained dtype.

## BidirLM-Omni: q4_k quantization cosine ceiling

Empirically, **q4_k vs HF bf16 settles at ~0.94 cosine** for the 2.5B
embedding variant, on both text-only (`tests/test_bidirlm_text.py`)
and image+text (`tests/test_bidirlm_image_text.py` /
`tests/test_bidirlm_image_text_lite.py`). That's the q4_k *intrinsic*
cosine — not a Phase 3 multimodal-injection bug.

The README's "cosine ≥ 0.99999" gate is for **graph regressions**
(CrispEmbed-q4_k vs a saved CrispEmbed-q4_k baseline from before a
code change); it doesn't measure CrispEmbed-vs-HF. To get ≥ 0.99
cosine vs HF bf16 you need q8_0 or higher precision.

Concretely measured (April 2026, q4_k against HF bf16 on /tmp/cat.jpg):

| path | cosine |
|---|---|
| text-only (`encode("Hello world")`) | 0.93–0.95 |
| multimodal (`encode_with_image_ids`) | 0.94 |

When debugging Phase 3 parity, run *both* test paths against the same
quant — if the multimodal cosine matches the text-only cosine for the
same prompts (modulo image content), the multimodal graph is fine and
the gap is the quant's intrinsic precision floor. If multimodal is
lower than text-only, that's a Phase 3 bug.

## BidirLM-Omni: image preprocessor parity is governed by mean/std, not the JPEG decoder

When porting HF Qwen2VLImageProcessorFast to C++ for `image_preprocess.cpp`,
the initial cosine vs HF was 0.97 — well below the ≥0.99 target. The
intuition was "stb_image's JPEG decoder differs from PIL/libjpeg-turbo by
a few LSBs, that propagates through the bicubic resize." That was wrong:

- Adding a PIL-decoded-RGB pass-through (`crispembed_preprocess_image_rgb`,
  skipping stb's JPEG decode entirely) moved cosine from 0.987 to 0.987.
- Switching `bicubic_resize_u8_to_f32` to round-clamp to integer (mimicking
  torchvision's uint8 round-trip on `tvF.resize(uint8, antialias=True)`)
  also moved cosine from 0.987 to 0.987.

The actual cause was the `image_preproc::config` defaults using OpenAI CLIP
mean/std `[0.481, 0.458, 0.408]` / `[0.269, 0.261, 0.276]`, while
**BidirLM-Omni's `preprocessor_config.json` specifies `mean = std = [0.5,
0.5, 0.5]`** (the SimVL / Qwen2-VL convention that maps `[0,1]` → `[-1,1]`).
Every normalized pixel value was off by a roughly-constant linear transform,
which has *high* flat cosine (0.987) but huge max-abs-diff (1.19 in
normalized space). The numbers had a strong mean-shift, which cosine
similarity is largely insensitive to until rescaled by std.

After fixing the defaults: pixel_values cosine 0.987 → 0.999989,
encode_image embedding cosine 0.970 → 0.999984. Sub-1e-5 residual is
sub-pixel torchvision-uint8 bicubic kernel weight quantization (PyTorch
uses int16 weights for the uint8 AA path; we use float weights).

`min_pixels` and `max_pixels` were also wrong for BidirLM-Omni (the
defaults from Qwen2-VL: 56² and 14²·4·1280; BidirLM uses 256² and 1024²
per the preprocessor config). For our test image these happened to land
on the same `smart_resize` output, but a different aspect ratio could
have produced a different grid_thw.

Lesson: when matching a model's preprocessor, read the actual
`preprocessor_config.json` from the HF repo. Don't assume CLIP defaults.
The converter (`models/convert-decoder-embed-to-gguf.py`) now writes
`bidirlm.vision.image_mean / image_std / min_pixels / max_pixels` into
the GGUF so future model variants can be picked up without guessing.

## BidirLM-Omni: image-embed splice via mask + add

HF does `inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)`
to replace token-embed rows at every `image_token_id` placeholder with vision
tower output. ggml has no native `masked_scatter`, so `decoder_embed.cpp`
emulates it with two host-prepared inputs:

- `in_keep_mask` shape `(1, T)` — 1.0 at text positions, 0.0 at image positions.
- `in_patch` shape `(H, T)` — `image_embeds[k]` row at the k-th image position,
  zeros at text positions.

```
cur = ggml_get_rows(token_embd, ids_t)
cur = ggml_mul(cur, in_keep_mask)   // zero out image-position rows
cur = ggml_add(cur, in_patch)       // splice image_embeds in at those rows
```

The `(1, T) * (H, T)` mul broadcasts the leading dim over H — same trick the
vision tower uses for the 4-corner pos-embed gather. DeepStack adds use the
same `(H, T)` patch shape, one per layer for the first `n_deepstack` layers,
zero everywhere except at image positions; `cur = ggml_add(cur, ds_patches[il])`
after each layer's residual+ffn output mirrors HF's `_deepstack_process`.

## Distribution: install(EXPORT) + ggml SHARED don't compose

When `crispembed-shared` is a SHARED library that PRIVATEly links a SHARED
ggml backend (`ggml-cpu`, `ggml-base`, …), `install(TARGETS crispembed-shared
EXPORT crispembed-targets)` errors with:

> install(EXPORT "crispembed-targets" …) includes target "crispembed-shared"
> which requires target "ggml-cpu" that is not in any export set.

The reason: even for PRIVATE link deps of a SHARED lib, CMake records them
as `IMPORTED_LINK_DEPENDENT_LIBRARIES` so downstream consumers know what
runtime SO names the .so will dlopen. install(EXPORT) demands those deps
either be in some export set or be system-IMPORTED.

Two viable workarounds:

1. **Hand-rolled IMPORTED target** (what CrispEmbed does, mirroring
   CrispASR): skip `install(EXPORT)` entirely. The `crispembed-config.cmake.in`
   uses `find_library(crispembed_LIBRARY crispembed HINTS …)` plus
   `add_library(crispembed::crispembed UNKNOWN IMPORTED)` to manufacture
   the IMPORTED target at config time. Runtime resolution of `libggml*.so`
   siblings is handled entirely by the .so's RPATH (`$ORIGIN` /
   `@loader_path`), not by the consumer's link line.
2. **Add ggml to the same EXPORT** via `set_target_properties(ggml*
   PROPERTIES EXPORT_NAME …)` and put ggml's install in your export.
   More invasive and requires patching the ggml submodule.

(1) is the right choice when the .so is the only thing the user sees and
ggml is implementation detail; (2) is right when you want consumers to be
able to `find_package(ggml)` separately.

## Distribution: relocatable pkg-config via ${pcfiledir}

`@CMAKE_INSTALL_PREFIX@` in a `.pc.in` is bound at configure time, not
install time. A user who runs `cmake --install build --prefix /opt/foo`
gets a `.pc` file with `prefix=/usr/local` (the configure default), and
`pkg-config --libs crispembed` returns wrong paths.

The fix is the **relocatable** pattern — set `prefix` from the .pc file's
own location:

```pc
prefix=${pcfiledir}/../..
libdir=${prefix}/lib
```

Since the .pc lives at `<prefix>/lib/pkgconfig/crispembed.pc`, going
`../..` from there is the prefix dir, no matter where the user dropped it.
Verified across `cmake --install --prefix /tmp/...`, tarball extraction
into `/opt/foo`, and the standard `/usr/local`.

## Distribution: forward-declared structs need typedefs for C consumers

`crispembed.h` had `struct crispembed_context;` plus function signatures
like `crispembed_context * ctx`. In C++ the struct name lives in the type
namespace so this compiles; in **C** the caller has to write
`struct crispembed_context *` everywhere. Adding

```c
typedef struct crispembed_context crispembed_context;
typedef struct crispembed_hparams { … } crispembed_hparams;
```

(forward-decl style for opaque types, full definition for value types) was
caught by the install verification test — a plain-C consumer of the
freshly `cmake --install`-ed header. The build directory consumers
(crispembed-cli, crispembed-server) didn't catch it because they're
compiled as C++.

## CNN forward path for face models (Phase 8)

### Available ggml ops for CNN
- `ggml_conv_2d(a, b, s0, s1, p0, p1, d0, d1)` — standard 2D conv
- `ggml_conv_2d_dw(a, b, ...)` — depthwise 2D conv
- `ggml_pool_2d(a, op, k0, k1, s0, s1, p0, p1)` — average/max pool
- `ggml_relu`, `ggml_leaky_relu(a, slope, inplace)` — activations
- No `ggml_prelu` — implement as: `relu(x) + slope * (x - relu(x))`
  where slope is a learned [C, 1, 1] tensor per channel

### BatchNorm folding
At inference time, BN is folded into the preceding Conv:
```
w_new = w * gamma / sqrt(var + eps)
b_new = (b - mean) * gamma / sqrt(var + eps) + beta
```
This eliminates all BN tensors from the forward pass.

### Conv2d output layout in ggml
`ggml_conv_2d` output: `[OW, OH, OC]` — width-first (ne[0]=OW).
To match HF's `[OC, OH, OW]` (channel-first): `permute(2, 1, 0)`.
This matters for position embeddings in ViT but NOT for CNNs
(CNNs are translation-equivariant — spatial order preserved naturally).

### SFace architecture (MobileFaceNet)
27 Conv layers (14 depthwise separable blocks), PReLU activation,
final GDC pool → FC(50176→128). 128-D L2-normalized embedding.
Input: 112×112 aligned face crop.

### SCRFD architecture (ResNet-50 + FPN)
58 Conv layers, ReLU activation, FPN with 3 scales (stride 8/16/32).
9 output heads: 3 × (confidence [N,1], bbox [N,4], landmarks [N,10]).
Dynamic input size (typically 640×640).
Needs NMS post-processing.

### AuraFace architecture (ResNet-100)
103 Conv layers, PReLU, 49 residual Add connections.
512-D ArcFace-compatible embedding. Apache 2.0.

### CrispASR CNN reference
CrispASR has CNN forward paths for marblenet (depthwise 1D conv),
wav2vec2 (grouped conv), and others. Same ggml ops, similar patterns.
Patches at tools/upstream-prs/ may be needed for CUDA conv2d.

### YuNet ggml Transpose behavior (2D vs 3D tensors)
The `replay_graph()` Transpose op does a real 2D transpose for tensors
where `ggml_n_dims == 2` (i.e., last dimension is 1). YuNet's cls/obj
outputs have 1 channel and thus get physically transposed, while bbox/kps
with 4/10 channels remain in the original ggml layout. This requires
different spatial indexing for each:
- cls/obj (1 channel, transposed): `data[row + col * grid_h]`
- bbox/kps (multi-channel, passthrough): `data[col + row * grid_w + chan * plane]`

## ViT / CLIP parity: patch ordering bug (FIXED — cos 0.8 → 1.0)

**Previously**: CLIP and SigLIP vision achieved cos ≈ 0.8 vs HuggingFace.
This was incorrectly attributed to FP32 matmul accumulation order differences.

**Actual root cause (fixed 2026-06-06)**: The `ggml_permute(2,1,0)` used to
reshape `[OW, OH, D]` → `[D, OH, OW]` produced column-major spatial ordering
when flattened to `[D, T]`: `t = oh + ow*OH`. But HuggingFace's `flatten(2)`
gives row-major: `t = oh*OW + ow`. Every patch beyond (0,0) got the wrong
position embedding, causing systematic error at the very first layer that
compounded through all 12 layers.

**Fix**: `ggml_permute(1,2,0,3)` produces `[D, OW, OH]` with `ne[0]=D,
ne[1]=OW, ne[2]=OH`. When flattened to `[D, T]`, patches follow row-major
`t = ow + oh*OW = oh*OW + ow`, matching HF.

**Result**: Per-layer cos = 1.000000 across all 12 layers. Final embedding
cos = 0.9998 vs HuggingFace (SigLIP-base-384).

**Lesson**: Always verify data layout empirically, especially with
`ggml_permute` where the axis semantics ("old dim N goes to new position
axN") differ from numpy/PyTorch conventions. The first few values matched
(patch 0 is at position 0 in both orderings) which masked the bug.

### SigLIP attention pooling head: missing residual

HF's `SiglipMultiheadAttentionPoolingHead` computes:
```
residual = probe + attention(probe, x_cat, x_cat)
output = residual + MLP(LayerNorm(residual))
```

The final `residual +` was initially missing in our implementation,
producing cos=0.17 vs HF. After fix: cos=0.74 (same precision ceiling
as other ViT models).

## Handwritten Math OCR (HMER + BTTR)

### Image polarity auto-detection

Both HMER and BTTR expect white-on-black input (ink = 1.0, background = 0.0).
Real-world images are typically black-on-white. Both implementations auto-detect
by checking the mean pixel value: if mean > 0.5, the image is inverted
(`pixel = 1.0 - pixel`). This avoids requiring the user to preprocess images.

### BTTR architecture (DenseNet + Transformer decoder)

BTTR (Bidirectionally Trained Transformer, ICDAR 2021) uses:
- DenseNet encoder (growth=24, 16 layers × 3 blocks, 1-channel grayscale)
- Conv 1×1 projection to d=256
- 2D sinusoidal position encoding (added to encoder features)
- Standard nn.TransformerDecoder (3 layers, 8 heads, d=256, FFN=1024)
- Post-LayerNorm, fused QKV weights preserved from PyTorch
- 113 LaTeX tokens, 6.5M params

Key implementation details:
- BN is folded into Conv at convert time (same as face models)
- Fused QKV weights: kept as-is, split via ggml_view_2d in the decoder
- Decoder uses causal mask for autoregressive generation
- Cross-attention: Q from decoder, K/V from encoder features

### HMER architecture (DenseNet-121 + GRU attention)

HMER uses a coverage-based GRU attention decoder (not Transformer):
- DenseNet-121 encoder (growth=32, 3 blocks of [6, 12, 24] layers)
- 2-channel input: grayscale + mask (coverage mechanism)
- GRU decoder with attention (not self-attention — attends to encoder features)
- Coverage vector prevents the decoder from re-attending to the same regions
- 112 LaTeX tokens, 6.8M params

### Dequantization for CNN inference

When running quantized HMER/BTTR models (Q4_K/Q8_0), the DenseNet Conv2D
kernels need dequantization because `ggml_conv_2d` only supports F32/F16
weights. Both implementations call `ggml_backend_tensor_get` to read
quantized data into a CPU buffer, then use `ggml_quantize.h` functions
to dequantize to F32 before building the conv2d graph node.

**Important**: ggml only supports `quantized → F32` cast (in `ggml_compute_forward_dup`).
Direct `Q8_0 → F16` cast triggers a fatal error. Always dequant to F32 first,
then cast F32 → F16 as a separate step.

### Conv weight reshape for GGUF

PyTorch Conv2D stores weights as [out_ch, in_ch, kh, kw] (4D). GGUF
requires 2D tensors for quantization. The converter flattens to
[out_ch, in_ch * kh * kw] for storage. At load time, the C++ code
reshapes back to the 4D layout expected by `ggml_conv_2d`.

**Pitfalls in the 2D→4D reshape** (resolved 2026-06-06):

1. **`ggml_n_dims()` collapses trailing 1s**: A 4D weight `[3,3,1,1]`
   (OC=1, IC=1) reports `ndims=2`, same as a genuinely flattened 2D weight.
   Fix: validate `KW*KH*IC*OC == nelements` before applying reshape.

2. **Depthwise conv IC detection**: DW weights are `[OC, 1*KH*KW]` when
   flattened. Using input channels as IC gives `kernel_area = 9/16 = 0`.
   Fix: parse the group attr from the graph node `[s1p1g16]` BEFORE
   the reshape. When `group > 1`, set IC=1.

3. **OC=1 weights report ndims=1**: Flattened `[IC*KH*KW, 1]` has
   `ne[1]=1`, so `ggml_n_dims = 1`. Use `ndims <= 2` to catch these.

### YuNet raw tensor cos vs ONNX — layout difference, not a bug

Raw tensor cos between C++ replay_graph and ONNX reference is 0.35-0.85
for bbox/kps outputs. This is NOT a parity issue — the Transpose and
Reshape handlers in replay_graph don't rearrange memory for 3D+ tensors
(passthrough). The result is planar `[C, H, W]` layout in ggml vs
interleaved `[H*W, C]` in ONNX. The YuNet decode loop uses matching
indexing: `col + row*grid_w + chan*plane` for the planar layout.

Verified: decoded detection coordinates match OpenCV FaceDetectorYN to
sub-pixel accuracy (< 0.5px diff) on both single-face and multi-face
images. The cls tensors (1 channel) show cos=0.985-0.992 because layout
is irrelevant for single-channel data.

## PosFormer port — encoder/decoder debugging

### 2D sinusoidal positional encoding: sin/cos MUST share frequency

PyTorch `ImgPosEnc` computes inv_freq with `arange(0, half_d, 2)` → 64 values.
Each sin/cos pair uses the SAME frequency: `sin(x * f_i), cos(x * f_i)`.

The initial C++ used different freq indices for sin vs cos:
```cpp
enc[2*i]     += sinf(x_norm * inv_freq[2*i]);     // freq 2i
enc[2*i + 1] += cosf(x_norm * inv_freq[2*i + 1]); // freq 2i+1 ← WRONG
```
Fix: both must use `inv_freq[i]` (or `inv_freq[2*i]` from a 128-element array).
Symptom: encoder cosine dropped to 0.58.

### Operation ordering: pos_enc THEN LayerNorm (not reversed)

PyTorch encoder does: `feature_proj → rearrange → pos_enc_2d → LayerNorm`.
The C++ initially did: `feature_proj → rearrange → LayerNorm → pos_enc_2d`.
LayerNorm normalizes the combined feature+pos signal; applying it before
pos encoding means the positional encoding is un-normalized.

### No ReLU after feature projection

PyTorch's `self.feature_proj = nn.Conv2d(...)` has no activation. The C++
had a spurious `relu_ip()` that clipped half the signal.

### Missing decoder input LayerNorm (the biggest bug)

PyTorch decoder does:
```python
tgt = self.word_embed(tgt)  # nn.Sequential(Embedding, LayerNorm)
tgt = self.pos_enc(tgt)     # sinusoidal pos encoding
tgt = self.norm(tgt)        # ← SECOND LayerNorm, was missing in C++
```

This `decoder.norm` was not in the GGUF converter OR the C++ inference.
Symptom: layer 0 self-attention output had cos=0.868 at step 0 (should
be 1.0). After adding `dec.input_norm` to converter and C++ decoder:
cos=1.000000 at every step, max_diff < 0.00001.

**Lesson**: never attribute divergence to "FP accumulation." If cosine is
below 0.999 at step 0, there is a real bug. Trace layer-by-layer with
intermediate dumps (after SA, after CA, after FFN) to find it.

### ARM (Attention Refinement Module) incremental mode is correct

The incremental ARM with per-ARM-instance accumulators matches the PyTorch
batch cumsum exactly, IF the encoder and decoder embedding are correct.
The ARM was never the bug — the divergence came entirely from the four
encoder/decoder bugs listed above.

### Bi-directional beam search vs greedy

PosFormer's published 62.7% uses bi-directional beam search (L2R + R2L
decode, cross-rate, pick best). The C++ implements L2R greedy only. Direct
comparison must use the PyTorch decoder.forward() in a manual greedy loop,
NOT the model.beam_search() which includes the bi-directional scoring.

### Kaggle kernel patterns — MUST follow established conventions

1. **Always clone CrispASR and import kaggle_harness** — never reimplement
   token resolution, progress logging, or GPU detection. The harness has
   been debugged across 15+ kernels.
2. **kernel-metadata.json uses string "true"** not boolean true.
3. **P100 (sm_60) + PyTorch**: Kaggle's pre-installed PyTorch (CUDA 12.x)
   dropped sm_60 support. Fix: `pip install torch --index-url .../cu118`
   which still supports P100 GPU. Do NOT fall back to CPU.
4. **Dataset mount path**: Kaggle mounts `chr1str/crispasr-hf-token` at
   `/kaggle/input/datasets/chr1str/crispasr-hf-token/`, NOT at
   `/kaggle/input/crispasr-hf-token/`. The harness was patched to scan
   both paths.
5. **Kaggle Secrets API**: intermittently returns ConnectionError. The
   dataset file fallback is the reliable path.
6. **Validation speed**: PosFormer's `approximate_joint_search` uses
   bi-directional beam search (beam_size=10) on all 986 test images.
   This takes 30-60 min per validation step. Override with greedy
   beam_size=1 for ~10x faster validation during training.
7. **Heartbeat**: wrap `trainer.fit()` in `kh.build_heartbeat("train")`
   so Kaggle logs show the run is alive during long operations.
8. **W&B run resume**: using a fixed `id=` with `resume="allow"` lets
   multi-session training continue the same W&B run. But if you kill
   and restart, the charts mix old+new data. Change the run ID for
   a clean restart.
9. **Vocabulary ordering is critical**: PosFormer uses an alphabetical
   dictionary (!, (, ), +, ...). Building vocab from `Counter.most_common()`
   sorts by frequency ({, }, 1, 2, ...), scrambling 110/113 token indices.
   The model trains "successfully" (internal metrics look fine) but the
   checkpoint is completely unusable with the original dictionary, GGUF
   converter, or C++ inference. ALWAYS use the canonical dictionary.txt.
10. **OOV tokens**: 14 CROHME captions contain `'` (apostrophe) which is
    not in PosFormer's 110-token dictionary. Filter these before training
    or the DataLoader crashes with KeyError.
11. **Cosine warm restarts are dangerous**: CosineAnnealingWarmRestarts
    (T_0=30) reset LR from 0.008→0.08 at epoch 94, crashing val_ExpRate
    from 57% to 38%. The model briefly recovered to 60.1% then fell
    again. Plain CosineAnnealingLR (no restarts) is safer. The 60.1%
    peak was lost because the checkpoint was overwritten.
12. **Never delete HF checkpoints hastily**: HuggingFace has git history
    — deleted files can be recovered via `hf_hub_download(revision=SHA)`.
    But always back up to /mnt/storage first before deleting.
13. **Dataset license verification**: figshare uploads can have wrong
    licenses (user picks any license, no verification). CROHME+HME100K
    on figshare claims CC BY 4.0 but the original datasets are NC/
    proprietary. Always check the original source, not re-uploads.
14. **UniMER dataset (Apache 2.0)**: wanderkid/UniMER_Dataset on HF has
    978K printed math images (ArXiv+Pix2tex) under Apache 2.0. The
    CROHME and HME100K subsets are excluded from this license ("requires
    manual download for copyright"). Best commercial data source found.
15. **MathWriting augmentation works**: Adding 2000 MathWriting samples
    (filtered to v1 110-token vocab from deepcopy/MathWriting-human on HF)
    to CROHME training broke the 59.3% ceiling → 60.5% verified.
    47% of MathWriting is compatible with v1 vocab (~109K out of 230K).
16. **Beam=10 bi-directional doesn't help our model**: 60.3% beam=10 vs
    60.5% beam=1 — beam search actually hurts by 0.2%. The R2L path
    sometimes picks worse hypotheses that beat correct L2R in cross-scoring.
    This differs from SJTU's published model where beam=10 added ~6 points.
17. **ReduceLROnPlateau is the key to peaks**: The best val_ExpRate always
    came right after an LR drop (0.08→0.005 gave 57%, 0.005→0.00125 gave
    62%). Manual LR patching in checkpoint files works when callbacks fail.
18. **Use deepcopy/MathWriting-human for MathWriting data**: Pre-rasterized
    JPG images + LaTeX strings on HuggingFace. Much faster than downloading
    and parsing 230K InkML files from Google Storage.

## NomicBERT v2-moe: hidden biases and GPT2 config

NomicBERT extends `GPT2Config`, so standard attribute names are missing:
`intermediate_size` → `n_inner`, `hidden_act` → missing (default GELU).
Patch onto config before accessing.

**Critical**: NomicBERT v1.5 has NO Wqkv/out_proj biases, but v2-moe
DOES have them (`Wqkv.bias [2304]`, `out_proj.bias [768]`). The original
converter assumed "no bias" based on v1.5 — this caused cos ≈ 0.955 parity
(consistent across all texts, easily mistaken for a precision issue rather
than a missing-data bug). Always check `bias is not None` dynamically
rather than hardcoding assumptions from one model variant.

Diagnosis approach: tensor diff showed all 148 weights bit-exact (0.0),
proving the bug was runtime-only. Layer-by-layer dump (`CRISPEMBED_DUMP_LAYERS`)
showed divergence starting at the attention output (before residual/LN),
which pointed to QKV projection. Manual `x @ W.T` matched HF weights
but not `Wqkv(x)` — the missing bias term.

## MoE encoder: ggml_mul_mat_id layout

For `ggml_mul_mat_id(A, B, ids)`:
- A shape `[ne0, ne1, n_experts]`, B shape `[ne0, K, T]`, ids `[K, T]`
- Result: `[ne1, K, T]` — transposes A along ne0/ne1 (same as mul_mat)
- For expert fc2 (down projection): HF stores `w2 [n_exp*inter, hidden]`,
  used as `act_out @ w2` (NO transpose). For ggml we need ne0=inter,
  ne1=hidden → numpy `[n_exp, hidden, inter]` → converter does
  `.permute(0, 2, 1)` on the `[n_exp, inter, hidden]` reshape.

## GELU variants matter for NomicBERT

NomicBERT uses `nn.GELU(approximate='none')` (exact erf-based), not the
tanh approximation. ggml provides both: `ggml_gelu()` (tanh approx) and
`ggml_gelu_erf()` (exact). Per-element error is ~1e-4 but compounds over
12 layers. Use `ggml_gelu_erf` for NomicBERT (both MoE expert and dense
FFN layers). Standard BERT typically uses `gelu_new` (tanh approx).

## General OCR: DBNet + TrOCR

### ConvTranspose2d weight layout differs from Conv2d
PyTorch Conv2d: `(OC, IC, KH, KW)` → flattened `(OC, IC*KH*KW)`.
PyTorch ConvTranspose2d: `(IC, OC, KH, KW)` → flattened `(IC, OC*KH*KW)`.

ggml `conv_transpose_2d_p0` expects kernel `[KW, KH, OC, IC]` — note IC
and OC are swapped vs regular `conv_2d` kernel `[KW, KH, IC, OC]`.
Needed a separate `prep_deconv_weight()` that reshapes to `(KW, KH, OC, IC)`.

### ODR violations with common struct names
`struct dec_layer` was defined in both `math_ocr.cpp` (30 pointer fields,
240 bytes) and `decoder_embed_internal.h` (18 pointer fields, 144 bytes).
In the test binary (linking only math_ocr), the correct 240-byte version
was used. In the CLI binary (linking everything), the 144-byte version won,
causing heap-buffer-overflow when math_ocr tried to write 30 fields into
18-field-sized allocations.

Fix: namespace-prefix struct names (`math_ocr_dec_layer`). ASAN caught this
immediately — always test with the full binary, not just individual TU tests.

### XLM-R / SentencePiece fairseq vocab offset
TrOCR uses XLMRobertaTokenizer which adds a fairseq offset to SentencePiece
token IDs. Raw `SentencePiece.id_to_piece(43778)` returns the wrong string.
Must use HF `AutoTokenizer.convert_ids_to_tokens(43778)` to get correct
mapping. Also: use `convert_ids_to_tokens()` (not `decode()`) to preserve
the `▁` word boundary markers for proper space reconstruction.

### DBNet FPNC (FPN-Cat) architecture
MMOCR's FPNC is NOT standard FPN. Standard FPN: lateral (1×1) → top-down →
smooth (3×3), all at 256ch. FPNC: lateral (1×1, 256ch) → top-down → smooth
(3×3, **64ch**), then concatenate all 4 levels (4×64=256ch). No output conv.
The smooth conv reduces channels, not the lateral.

### ggml_interpolate replaces ggml_upscale_ext
`ggml_upscale_ext` is deprecated. Use `ggml_interpolate(ctx, a, ne0, ne1,
ne2, ne3, mode)` with `GGML_SCALE_MODE_BILINEAR` for FPN upsampling.
Nearest-neighbor vs bilinear makes a visible difference in detection parity
(cos_min drops from 1.0 to 0.0 with nearest on some rows).

## Quantizer skips 3D tensors

`tools/quantize.cpp` line 172 skips tensors with ndims > 2 ("skipping N-D
tensor (conv2d)"). This was added for face model conv kernels (4D) but
also catches MoE expert weights (3D: `[n_exp, dim1, dim2]`). For
nomic-v2-moe, this means expert weights stay F32 in all quants, limiting
Q8_0 compression to 1.6x instead of potential ~3x. Fix: quantize 3D
tensors by iterating over the outermost dimension.

## Qwen2.5-VL: KV cache for VLM generation

### Prefill K/V extraction pattern

The prefill forward pass computes all prompt tokens at once. To extract
per-layer K/V for caching, add output tensors **after mRoPE but before
GQA repeat**: the K/V at shape (head_dim, n_kv_heads, n_tokens) is what
goes into the cache. GQA repeat is reapplied in each decode step.

```cpp
// In prefill graph, after RoPE:
K_flat = ggml_reshape_2d(g, ggml_cont(g, K), kv_dim, n_tokens);
ggml_set_name(K_flat, "k_out_0");
ggml_set_output(K_flat);
```

### Decode step graph: single token + cache concat

The decode step takes one token embedding + cached K/V as inputs.
K/V cache tensors are 2D (kv_dim, n_kv) passed as graph inputs,
reshaped to 3D, concatenated with the new single-token K/V on dim 2,
then GQA-repeated for attention.

No causal mask needed — a single query token always attends to all
cached KV tokens (it's always the latest position).

### Token embedding lookup for quantized models

During decode, embed_tokens may be quantized (Q8_0/Q4_K). Can't just
index into the data directly. Solution: build a mini ggml graph with
`ggml_get_rows(embed_tokens, [token_id])` to handle dequantization.

### KV cache memory budget

36 layers × 2 (K+V) × kv_dim(256) × n_tokens × 4 bytes.
For 500 prompt tokens: 36 × 2 × 256 × 500 × 4 = 36 MB.
For 2000 tokens: 144 MB. Well within budget.

## Qwen2.5-VL: BPE tokenizer from GGUF

### Standard ggml tokenizer keys

Write to GGUF: `tokenizer.ggml.tokens` (string array), 
`tokenizer.ggml.merges` (string array), `tokenizer.ggml.model` = "gpt2",
`tokenizer.ggml.type` = 1 (BPE).

Load in C++: read arrays from GGUF metadata, pass to `BPETokenizer.load()`.

### GPT-2 byte-level decode

BPE tokens are unicode codepoints, not raw bytes. Decode: concatenate
token strings, then reverse the `bytes_to_unicode()` mapping. The table
maps printable ASCII + Latin-1 to themselves, and remaining bytes to
codepoints 256+. Build the inverse table once at init.

### Chat template in C++

Hardcode special token IDs (im_start=151644, system=8948, user=872,
assistant=77091, etc.) and use the BPE tokenizer for the user prompt
text only. This avoids needing a Jinja template engine in C++.

## Qwen2.5-VL: ggml_set_output memory impact

Marking N intermediate tensors as output prevents ggml's graph allocator
from reusing their memory. For 32 ViT + 36 LLM layers, this adds ~500 MB
of pinned memory — enough to OOM on 8 GB machines.

Fix: only set_output when diff comparison is active (`ctx.diff_ref_path`
is non-empty). Logits tensor always needs set_output.

## Kaggle: always use the full harness

Never simplify or inline the CrispASR kaggle_harness.py. It has:
- `kh.build_heartbeat()` — prevents Kaggle killing long ops (uploads)
- `kh.resolve_hf_token()` — 3-tier auth (env → Secret → dataset file)
- `kh.step()` — JSONL progress to /kaggle/working + HF mirror
- `kh.install_build_toolchain()` — ninja + ccache + mold

Bundle `kaggle_harness.py` in the push directory as fallback.
Use `chr1s4/crispasr-hf-token` dataset (chr1s4's own, not chr1str's).
Don't `pip install torch` (pre-installed, 2 GB download wastes time/OOMs).
## LFM2 backbone: causal → bidirectional (GLiNER NER port)

Porting the LFM2.5 backbone from CrispASR (causal audio model) to
CrispEmbed (bidirectional NER encoder) required exactly two changes:

1. **Attention mask**: causal `(j <= i) ? 0 : -inf` → pass `nullptr`
   to `ggml_flash_attn_ext` for full bidirectional attention.
2. **Conv padding**: left-pad `pad=K-1` → center-pad `pad=(K-1)/2`
   for symmetric (bidirectional) convolutions.

Everything else (RMSNorm, SwiGLU FFN, RoPE, GQA, ShortConv gating)
is identical. The layer_types string `"ccaccaccacacacac"` is the same
pattern for both the 1.5B audio and 350M NER models.

## GLiNER layer fusion: sigmoid not softmax

GLiNER's `LayersFuser` uses **sigmoid** gates (independent per-layer),
NOT softmax (competing across layers). The squeeze-and-excitation
pattern is: squeeze(hidden→1) per layer → mean over tokens → W1→ReLU→W2
→ **sigmoid** → element-wise multiply each layer → sum → output_projection.

Using softmax instead produced cos=0.65 vs reference. Sigmoid gives
cos=1.000000.

## GLiNER pipeline: word-level pooling before BiLSTM

GLiNER's `subtoken_pooling="first"` means: after the backbone + layer
fusion, take the first BPE subtoken of each word to get word-level
representations, THEN run the BiLSTM on word-level only. The entity
type reps (at `<<ENT>>` positions) are extracted from the fused
token-level output BEFORE the BiLSTM.

Running the BiLSTM on the full token sequence (including label prefix
tokens) produces cos=-0.96 vs reference. Word-level gives cos=1.000000.

## GLiNER tokenization: regex word splitter

GLiNER's `WhitespaceTokenSplitter` uses regex `r"\w+(?:[-_]\w+)*|\S"`,
NOT simple whitespace splitting. This separates punctuation from words:
"Cupertino," → ["Cupertino", ","]. Simple whitespace splitting glues
punctuation to the word, causing entity span mismatches.

## GLiNER input format

The input sequence is: `BOS <<ENT>> label1 <<ENT>> label2 ... <<SEP>> text EOS`.
Note: `<<ENT>>` before each label (not `<<SEP>>` between), single
`<<SEP>>` after all labels, BOS at start, EOS at end.

## ggml_conv_1d_dw requires F16 kernel weights

`ggml_conv_1d_dw` internally uses `ggml_im2col` which asserts
`src0->type == GGML_TYPE_F16`. When model weights are F32, cast
the conv kernel to F16 before passing to `ggml_conv_1d_dw`:
`ggml_cast(ctx, w.conv_conv_w, GGML_TYPE_F16)`.

## ggml_gallocr works with model weight tensors

Model weight tensors that already have a backend buffer are skipped
by `ggml_gallocr_alloc_graph` — it only allocates compute tensors.
So model weights can be used directly as operands in graphs allocated
with gallocr. No need for `ggml_backend_sched` for this use case.

However: `ggml_add` with a 1D bias tensor (ne[0]=D) broadcasts
correctly over a 2D tensor (D, N) — no `ggml_repeat` needed.
Using `ggml_repeat` with a reshaped view of a weight tensor can
cause subtle issues.

## Dequantizing backend tensors to CPU

Model weight tensors in Q8_0/Q4_K backend buffers can't be read
with `ggml_backend_tensor_get(t, dst, 0, nelements*sizeof(float))`
— that reads raw quantized bytes. Use:
```cpp
std::vector<uint8_t> raw(ggml_nbytes(t));
ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
ggml_get_type_traits(t->type)->to_float(raw.data(), out, nelements);
```

## Cache dequantized weights for CPU-side ops

When CPU-side operations (BiLSTM, layer fusion) read quantized model
weights every call, the dequant overhead adds ~50-100ms per call.
Cache the F32 versions in the context struct at init time — they're
small (~52 MB for BiLSTM + fuser weights) and eliminate per-call cost.

## Batched span MLP via ggml graph: 2-3x speedup

GLiNER's span scoring evaluates hundreds of spans, each through a
2-layer MLP (3072→4096→1024). The naive approach runs each span as a
separate CPU scalar matmul. Batching all spans into a single ggml
`mul_mat` (3072, n_spans) × (3072, 4096) leverages BLAS and gives
2-3.5x speedup on the GLiNER head.

Two-pass approach works well: pass 1 computes proj_start/end/first +
prompt_rep (independent of spans), then CPU assembles span
concatenations, pass 2 computes batched out_project + scoring.

## Swin shifted-window attention: cyclic_shift vs torch.roll

When implementing `torch.roll(x, shifts=-s, dims)` as a C++ cyclic shift
function, the sign convention is inverted:
- `torch.roll(shifts=s)`: `out[y] = in[(y - s) % H]`
- `cyclic_shift(shift_h=s)`: `out[y] = in[(y + s) % H]`

So `torch.roll(shifts=-3)` = `cyclic_shift(shift_h=+3)`. Getting this
wrong produces cos=0.0 on the shifted data — completely scrambled.

Also: HF Swin pads to window-size multiples BEFORE the cyclic shift.
The mask is built on the padded dimensions. If you shift first then pad,
the data layout differs in the boundary/padding zone.

## Swin GELU: tanh approx vs exact erf

HF Swin uses `nn.GELU()` which is exact erf-based GELU:
`0.5 * x * (1 + erf(x / sqrt(2)))`. The commonly-used tanh
approximation `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
introduces small systematic errors that compound through multiple
layers. Always check which variant the upstream model uses.

## Per-step diff isolates bugs faster than E2E comparison

When block 1 had cos=0.995, comparing only the block output gave no
clue where the error originated. Adding per-step diff checkpoints
(LN1 → shift → windows → attention → merge → residual → FFN → output)
immediately showed that LN1 was perfect (cos=1.0) but shifted data
was cos=0.0 — pinpointing the cyclic shift function as the culprit.
Always instrument fine-grained checkpoints for shifted/masked operations.

## Swin PatchMerging: sub-patch concat order matters

HF Swin PatchMerging concatenates 2×2 sub-patches in the order
`[TL, BL, TR, BR]` (top-left, bottom-left, top-right, bottom-right):
```python
x0 = x[:, 0::2, 0::2, :]  # top-left
x1 = x[:, 1::2, 0::2, :]  # bottom-left
x2 = x[:, 0::2, 1::2, :]  # top-right
x3 = x[:, 1::2, 1::2, :]  # bottom-right
```
The natural C loop `(2y, 2x), (2y, 2x+1), (2y+1, 2x), (2y+1, 2x+1)`
gives `[TL, TR, BL, BR]` — positions 1 and 2 swapped. This feeds the
wrong channels into the LayerNorm+Linear reduction, corrupting ALL
subsequent stages (cos=-0.37 at encoder output).

## RoBERTa position embedding offset (+2)

RoBERTa position embeddings have `padding_idx=1`. Content positions
start at index `padding_idx + 1 = 2`. So decode step 0 uses
`pos_embed[2]`, step 1 uses `pos_embed[3]`, etc. Using index 0
(random values, norm=0.57) and index 1 (all zeros, padding) produces
wrong embeddings that cascade through the decoder.

## Tesseract `.traineddata` binary format

The `.traineddata` file is a custom archive: `int32 n_entries`, then
`n_entries × int64` offsets (-1 = absent). Component 17 = LSTM network.

The LSTM component has a recursive binary tree starting with
`Network::Serialize`: `int8 type_enum` (if 0 → read type string),
then `int8 training, int8 needs_bp, int32 flags, int32 ni, int32 no,
int32 num_weights, uint32 name_len + chars`.

Key gotcha: `kDoubleFlag = 128` (not 2!). Controls whether scales/arrays
use float64 vs float32. The kInt8Flag is 1, kAdamFlag is 4.

## Tesseract int8 weight dequantization

Stored scale = `runtime_scale * INT8_MAX`. At load:
`loaded_scale = stored / 127`. Runtime: `output = dot(int8_w, int8_input) * loaded_scale`.
For float dequant: `float_w = int8_w * stored_scale` (the RAW value,
NOT divided by 127). The factor of 127 cancels with the int8 input
quantization. Dividing by 127 gives weights 127x too small.

## Tesseract Convolve layer is NOT a learned conv

`Convolve` in Tesseract just stacks (im2col) the 3×3 neighborhood —
no trainable weights. The actual "convolution" is a `FullyConnected`
layer with tanh activation in a `Series` after the `Convolve`.

## XYTranspose wraps SummLSTM, not the other way

`Reversed(NT_XYTRANSPOSE)` extends `Plumbing` — it's a container that
reads child networks (including count + learning rates). The `Lfys`
VGSL layer is actually: `XYTranspose(SummLSTM)` where the XYTranspose
swaps axes before and after so the LSTM runs over the y-dimension.

## Vertical shear for deskew, not horizontal

Leptonica's `pixFindSkew` uses VERTICAL shear (shift columns up/down)
to score alignment, NOT horizontal (shift rows left/right). Horizontal
shear doesn't change row sums for horizontal text lines. Vertical
shear moves pixels between rows, which is what the differential
square-sum scoring measures.

## 1-bit DWA morphology: massive speedup from word-level ops

Packing 32 pixels per uint32 and using word-level OR (dilation) gives
21x speedup over float separable morphology and 32x less memory.
The cache efficiency from 1-bit packing dominates even over the
algorithmic improvement.

## PAN super-resolution for low-DPI OCR

Tested PAN 4x upscale on 75dpi text (150×9 px → 600×36 px):
- 75dpi raw → Tesseract: "C Melbe Wesld1" (garbage)
- 75dpi + PAN 4x → Tesseract: "Hello Werdd 123" (1 char error)
- 150dpi raw → Tesseract: "Hello World 123" (perfect)

Key findings:
1. PAN 4x rescues unreadable 75dpi text — garbage → mostly correct
2. Don't cleanup (binarize/deskew) before SR — destroys sub-10px text
3. Don't cleanup after SR either — upscaled text is clean enough
4. For 150dpi+, no SR needed — OCR works fine directly
5. Optimal pipeline: estimate DPI → if < 150, PAN 4x → OCR

The `estimate_dpi()` heuristic assumes longer edge ≈ 11 inches (A4/letter).
This is wrong for cropped regions but acceptable for full pages.

## SigLIP ViT ggml graph: tensor layout and permute pattern

SigLIP ViT (D=1152, n_heads=16, d_head=72, T=729 patches) ggml graph
matches the pattern established in `bidirlm_vision.cpp` (Qwen2VL ViT).
Key differences vs Qwen2VL:

1. **No RoPE**: SigLIP uses absolute position embeddings added before
   the transformer loop. No cos/sin tensors needed.

2. **Separate Q, K, V projections**: bidirlm uses fused QKV; SigLIP
   uses three independent `vis.layer.N.attn.{q,k,v}.weight` matrices.
   Three `ggml_mul_mat` calls per layer instead of one.

3. **GELU (tanh approx)** in FFN: `ggml_gelu`. Not `ggml_gelu_erf`.

**Tensor shapes through the attention block:**
- Input x: `[D, T]` ggml (ne[0]=D fast dim, ne[1]=T tokens)
- After QKV mul_mat + bias: `[D, T]`
- After `reshape_3d(Q, d_head, n_heads, T)`: `[d_head, n_heads, T]`
- After `permute(0, 2, 1, 3)` + `cont`: `[d_head, T, n_heads]` contiguous
- scores = `mul_mat(K, Q)`: K=[d_head,T,n_heads], Q=[d_head,T,n_heads]
  → `[T_k, T_q, n_heads]`. `ggml_mul_mat(a,b)` computes b@a^T so
  the inner dim (ne[0]) must match: both ne[0]=d_head ✓
- After `soft_max_ext(scores, null, 1/sqrt(d_head), 0)`: same shape,
  softmax over dim 0 (key axis) ✓
- V_perm = `permute(V, 1, 0, 2, 3)` + `cont`: `[T, d_head, n_heads]`
- attn = `mul_mat(V_perm, scores)`: [T,d_head,n_heads] × [T_k,T_q,n_heads]
  → `[d_head, T_q, n_heads]`
- After `permute(attn, 0, 2, 1, 3)` + `cont`: `[d_head, n_heads, T]`
- After `reshape_2d(attn, D, T)`: `[D, T]` — per-token D-vector with
  head-major interleaving: [h0_d0, h0_d1, ..., h1_d0, ...] ✓

**ggml_norm broadcasting**: `ggml_norm(g, x, eps)` normalizes along ne[0].
For x=[D,T], each of the T token vectors is independently normalized ✓.
`ggml_mul(g, normed, w)` where w=[D] broadcasts over T via `ggml_can_repeat`
(T % 1 == 0). No reshape needed for bias/scale vectors.

**Feature extraction**: `ggml_set_output(ggml_cont(g, x))` at each
`feature_layers[fi]` layer index. The ggml_cont ensures the feature tensor
is a separate node (not an alias of x); the scheduler keeps its buffer live
after the full graph runs. Four feature layers → four independent tensors
read back via `ggml_backend_tensor_get` after compute.

**compute_meta buffer**: Pre-allocated in `vis_compute_meta` (owned by
`granite_vision_context`). Passed to `ggml_init({size, data, no_alloc=true})`.
`ggml_free(g)` after compute frees only the `ggml_context` struct (small
malloc), NOT the buffer itself — so the graph nodes in compute_meta stay
valid while the scheduler holds the alloc. Buffer is overwritten on the
next inference call (graph rebuilt from scratch each call). This avoids
heap allocation per inference. Same pattern as `bidirlm_vision.cpp`.

**Scalar fallback**: `gv_run_vit_graph` sets `feat_outs.assign(n_feat, {})`
at entry. If the graph fails (alloc fails, compute fails, missing weights),
it returns early with all feat_outs empty. The caller checks
`n_feat > 0 && layer_outputs[0].empty()` to trigger the scalar path.

## granite_vision: converter writes weights transposed — reshape before ggml_mul_mat

`models/convert-granite-vision-to-gguf.py` writes 2D weights with their
PyTorch `[out, in]` shape **un-reversed**, so the GGUF `ne` is transposed
relative to ggml's convention (ne[0] = the fast/contraction = `in` dim). The
CPU-scalar path (`gv_linear` + DequantCache) is immune — it dequantizes the
raw bytes and indexes with explicit `id`/`od`. But **`ggml_mul_mat` asserts**
`GGML_ASSERT(ggml_can_mul_mat(a, b))` for every non-square weight (k/v:
512×2048, gate/up: 8192×2048, down: 2048×8192, projector linear_1:
2048×4608, vision fc1/fc2). Square weights (q/o, linear_2) happen to pass.

Fix: relabel the contiguous data with a reshape before the matmul —
`ggml_mul_mat(g, ggml_reshape_2d(g, w, w->ne[1], w->ne[0]), x)`. This is a
no-op for square weights and a pure view (no copy) otherwise. The projector and
LLM graphs (`gv_run_projector_graph`, `gv_run_llm_body`) were missing it and
crashed until `feat/granite-vision-ne-fix`.

**Caveat (this is a real footgun — see next entry):** the reshape is only valid
on a *quantized* tensor when the **target `ne[0]` is a multiple of the quant
block size**. It happens to hold for the Q4_K LLM weights (256-block; reshaped
`ne[0]` ∈ {2048, 8192}) and the Q8_0 projector (32-block; 4608/2048), but NOT
for the Q8_0 vision `ffn.down` (32-block; reshaped `ne[0]=4304`, 4304 % 32 ≠ 0).
There the reshaped view mis-strides the blocks and silently returns garbage.

Beware: a *systemic* load-time `ne` swap would double-correct the vision FFN
(which already reshapes per-site) — pick one approach, not both.

## granite_vision: the ggml-graph divergence was a Q8_0 reshape, not alloc reuse (2026-06-21)

`gv_run_vit_graph` on Metal produced cos ~0 from the first encoder layer with
`max_abs` exploding through the residual stream. A prior handover diagnosed an
"unpatched ggml-alloc in-place buffer-reuse defect" (claiming the input `x_t`
went NaN after compute). **That was wrong** — re-verifying with
`CRISPEMBED_GRANITE_VIS_DBG=1` showed `x_t` *unchanged* after compute, no NaN.

Real cause: `ggml_reshape_2d` on a **quantized** weight whose target `ne[0]`
isn't a multiple of the quant block size. The converter stores 2D weights with
transposed `ne` (ne[0]=out), so the FFN matmul relabels via reshape. For the
Q8_0 `vis.layer.*.ffn.down` weight, `ne[0]` goes 1152→4304 and 4304 % 32 ≠ 0,
so the reshaped view's 32-element Q8_0 blocks no longer align to rows — every
down-projection from layer 0 onward is garbage that compounds through the
residual adds (hence "explodes" and looks like an alloc smash). The square Q8_0
attention weights are used raw (ne[0]=in=1152 already carries the blocks) and
were always fine; the F16 `ffn.up` reshapes safely (per-element). This mirrors
the merged `restormer`/`nafnet` "transposed 2D conv weight" fix, which also
casts to F32 before reshaping a quantized weight.

Fix (`gv_run_vit_graph` FFN): dequantize quantized weights to F32 *before* the
reshape — `ggml_is_quantized(w->type) ? ggml_cast(g, w, F32) : w`, then
`ggml_reshape_2d`. `ggml_cast` reads the Q8_0 with its *original* (correct) `ne`
so the dequant is right; the post-cast F32 reshape is a plain element relabel.
Result: per-layer parity with the scalar ViT (cos 0.9996–0.99987 early; 0.96
late = the scalar's `F.layer_norm` eps 1e-5 vs model 1e-6 ref artifact, NOT a
bug), and ~3 s vs ~100 s scalar. Gated behind `CRISPEMBED_GRANITE_VIS_GRAPH=1`.
Two caveats keep it opt-in: the **ggml-CPU** backend still drifts at late layers
(cos ~0.84 — only Metal is validated), and the full Metal OCR path is still
blocked on the LLM Metal graph.

**Separately on the LLM Metal graph:** the same handover called it "broken on
Metal (emits EOS immediately)". The 7-token self-consistent diff
(`granite-llm-ref`) **passes on Metal at cos 0.9999 through logits**, so the LLM
math/weights/reshape are correct. But a full 784-token prefill with spliced image
features yielded **0 decode steps** (first token = EOS) on Metal while the same
prefill on ggml-CPU decoded correctly. Root cause was NOT attention/length — see
the dedicated "Metal mul_mm F16 activation overflow" entry below.

## Metal mul_mm F16 activation overflow on massive activations (2026-06-21)

The Granite LLM Metal graph (`gv_run_llm_body`) produced correct logits for text
tokens at any length, and correct results on the ggml-CPU backend, but cascaded
to **NaN from layer 9** during the real OCR prefill on Metal — yielding 0 decode
steps. Localized by dumping per-layer max-abs over all tokens: the residual
stream carries a **massive activation** (~1.1e4 — a few outlier dims, amplified
~×12 by the `embedding_multiplier` applied to the spliced image features), held
fine through layer 7, then exactly 2 elements overflow to `inf` at layer 8.

Root cause: Apple's batched matmul kernel `kernel_mul_mm_*` (selected when
`ne11 > ne11_mm_min`, hardcoded 8, i.e. the T=784 **prefill**; T=1 **decode** uses
`mul_mv`) casts activations to **F16** for the simdgroup matrix units. The SwiGLU
`silu(gate)*up` product for the massive dimension exceeds F16 range (65504) at
the first layer whose FFN weights are large enough → `inf` → NaN. This is the
same phenomenon `ggml_mul_mat_set_prec(GGML_PREC_F32)` targets "for phi-2" — but
on Metal that flag does NOT change `mul_mm` (the kernel name has no prec variant),
so it does not help here. F32 KV cache, disabling fusion/concurrency, and manual
attention all leave the overflow (it is in the FFN matmul, not attention/KV).

Fix (`gv_run_llm_body` FFN): scale the down-projection activation down before the
matmul and back up after — a **lossless exponent shift** (F16 is floating-point,
so scaling preserves relative precision) that keeps the F16 cast in range:
`down = scale(mul_mat(Wd, scale(silu(gate)*up, 1/256)), 256)`. 256 gives headroom
to ~1.6e7. Result: full Metal OCR (vision graph + LLM graph) returns the correct
text in ~22 s (vision ~3 s, 784-tok prefill ~12 s, decode ~5 s) vs the scalar
path's ~100 s vision + ~8 min prefill. Both graphs now DEFAULT ON for GPU
backends; scalar stays default on ggml-CPU (where the ViT still drifts).

Diagnostic lesson: the LLM diff originally exercised only the scalar
`gv_llm_decode_step`, NOT the ggml graph — so "LLM diff passes" said nothing
about `gv_run_llm_body`. Threading a dump_cb through `gv_run_llm_body` and running
it under `granite_vision_dump_llm` when `CRISPEMBED_GRANITE_LLM_GRAPH=1` is what
finally put the graph under the harness. Always confirm the diff drives the path
you think it does.

## ggml-CPU ViT precision: F16-table gelu + Q8_0-quantized activations (2026-06-21)

After the Metal Granite ViT graph reached scalar parity, the **ggml-CPU** backend
of the same graph still drifted to cos ~0.84 at late layers (vs Metal's 0.96),
accumulating over the 27 encoder layers. Two CPU-only precision losses, both
absent on Metal (which runs these in F32):

1. **`ggml_gelu` (tanh) on CPU is a precomputed F16 lookup table.**
   `ggml_vec_gelu_f32` (ggml-cpu/vec.h) does `ggml_table_gelu_f16[fp32_to_fp16(x)]`
   — the input is quantized to F16 and the result is an F16 table value. `silu`
   (the LLM FFN) is computed directly in F32, which is why the LLM CPU graph never
   showed this. `ggml_tanh`/`ggml_gelu_erf` on CPU are also direct F32. Fix:
   compute tanh-gelu from primitives —
   `0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))` with `ggml_tanh`/`ggml_mul`/
   `ggml_scale` (all F32). Closed about half the gap (0.84 → 0.90).

2. **CPU `mul_mat` against a Q8_0 weight quantizes the F32 activation to Q8_0**
   for the dot product (its `vec_dot_type`), coarser than Metal's `mul_mm` F16
   activation cast. The square attention q/k/v/o weights were used raw (Q8_0), so
   on CPU every attention projection ran an int8 activation dot → drift. Fix:
   `ggml_cast` the (square) attention weights — and the F16 FFN up — to F32 **on
   the CPU backend only** (`ggml_backend_is_cpu(ctx->backend)`), forcing an F32
   activation dot. No-op on GPU, where raw Q8_0 + `mul_mm` is already accurate and
   faster, and avoids the extra F32 weight copy. Closed the rest (0.90 → 0.958 =
   scalar parity); CPU end-to-end OCR then returns the correct text.

Both Granite graphs are now correct on Metal AND ggml-CPU and default ON for all
backends. Takeaway: ggml's CPU backend silently trades precision for speed
(F16-table activations, activation requantization to the weight type); when a
graph drifts on CPU but not GPU, suspect those before the math.

## Self-consistent crispembed-diff reference from the GGUF — no original weights needed (2026-06-21)

To numerically validate a backend you don't have the original framework checkpoint
for, build the diff reference from the **GGUF the runtime already loads**: reverse
the converter's name map (every `convert-*-to-gguf.py` has an explicit
`add(gguf_name, torch_name)` table), dequantize each gguf tensor, build the torch
`state_dict`, `load_state_dict(strict=False)` into the original arch, run the
forward, and dump input+stages+output as the `-ref.gguf`. This proves
**C++ runtime == framework on identical weights** (catches algorithm bugs; not
weight-conversion bugs — for those use the real checkpoint). It's the granite-llm
-ref pattern generalized; used to verify hat_sr's OCAB at output cos 0.999968 with
no HAT `.pth`. See `tools/dump_hat_reference_from_gguf.py`. Gotchas: computed
buffers (`relative_position_index`, `attn_mask`, `mean`) aren't in the gguf —
`strict=False` keeps the arch's `__init__`-computed ones; assert that no *weight*
keys are missing. Pick a small input size (e.g. 32) so the scalar C++ forward in
the diff harness finishes quickly. Refs get uploaded to the model's HF repo.

Corollary methodology trap: **a diff test that exists in `tests/` but isn't wired
into CMake was never actually run.** `tests/test_hat_diff.cpp` was present for
months but had no `add_executable` — so "there's a test for HAT" was false. When a
backend is "validated," confirm the harness target is built and the reference
exists, not just that the `.cpp` is in the tree.

## VLM/OCR decoder perf: keep the LM head on-GPU; don't re-copy KV history (2026-06-21)

A cross-backend audit of the OCR decoders (granite_vision vs qwen2vl_ocr,
internvl2_ocr, smoldocling_ocr, deepseek_ocr2) surfaced two cheap decode wins
granite was missing — both worth checking on any ggml decoder:

1. **Run the LM head (vocab projection) IN-GRAPH, not on CPU.** Granite computed
   the ~`vocab×dim` logits with `core_cpu::linear_cpu` every token, which also
   forces a hidden-state GPU→CPU readback. qwen2vl/internvl2/deepseek append
   `ggml_mul_mat` for the last token to the graph and read back `[vocab]`. Moving
   it on-GPU (`gv_run_llm_body`'s optional `logits_out`) cut granite decode
   270→165 ms/tok (~1.6×). Gotchas: the last-token hidden is post-final-norm so
   it's O(1) → the T=1 `mul_mv` is F32-safe (no massive-activation overflow); and
   Granite's tied embedding weight has the transposed `ne` (`ne[0]=vocab`), so
   relabel it to `[in=dim, out=vocab]` (cast quantized first) before `mul_mat` —
   same pattern as the FFN weights.

2. **Don't `ggml_cont` the KV history every step.** `flash_attn_ext` reads K/V via
   `nb` strides, so pass the `[d_head, Lk, n_kv]` cache *views* straight in. The
   `cont` was copying the entire history each layer each token (~64 MB/step at
   len ~800). Writing new K/V into the cache still uses `ggml_cpy` into the
   persistent buffer (correct); only the read-side full-history copy was waste.

**Cross-backend reference** (who does what, as of 2026-06): LM head on-GPU —
qwen2vl, internvl2, deepseek (granite now too); LM head on CPU — smoldocling.
**Decode-graph reuse**: only `deepseek_ocr2` builds a *persistent* T=1 decode
graph once (`build_persistent_decode_graph`) and just sets inputs + recomputes
per step; everyone else (incl. granite) rebuilds the graph per token. qwen2vl's
cheaper half-measure: keep KV-read views + mask at constant `[0..max_seq]` shape
so `ggml_backend_sched_alloc_graph` takes the no-realloc fast path. KV cache is
F16/Metal-resident across all. No OCR decoder uses `sched_reserve` yet.

**Profile before chasing graph reuse.** Env-gated timers around granite's
`gv_run_llm_body` showed a T=1 decode token is **~95 % GPU `graph_compute`**
(~135 ms); `ggml_init`+graph build ≈ 0.8 ms and `sched_reset`+`sched_alloc_graph`
≈ 5 ms are noise. So a persistent decode graph / `sched_reserve` would save ~5 ms
of ~140 ms — **not worth the refactor here** (contrast the LEARNINGS note above
where alloc *was* the bottleneck for a 16-layer 350 M model; for a 2 B/40-layer
model the GPU compute dominates). Decode is **dispatch-bound on ~800 tiny T=1
kernels** (each `mul_mv` etc. underutilizes the GPU), so the lever is **fewer
kernels per token**, not graph management. Concrete win: the SwiGLU down-proj
F16-overflow guard (÷256/×256 exponent shift) only matters for the prefill
`mul_mm` F16 cast — skip it for T ≤ 8 (decode `mul_mv` is F32-safe), cutting 2
dispatches/layer → decode 165 → 139 ms/tok. Cumulative 270 → 139 (~1.9×).

The one-shot *total* is still dominated by the 784-token prefill (~12 s, ~100 %
GPU compute) + Metal **pipeline compilation** (compiled lazily on first kernel
use, so it lands inside the first vision/prefill `graph_compute` and inflates
one-shot timings by seconds; a persistent server pays it once). The decode wins
compound for long (multi-hundred-token) document OCR, which is the real workload.

## granite_vision OCR: the two image-path bugs (vision parity passed but OCR hallucinated)

The vision tower passed crispembed-diff at cos≈0.99999, yet end-to-end OCR
produced a fluent but wrong document (`<doc>…indd…</doc>`). Two bugs, both in
the *inference* path the parity test never exercised (it feeds an already-
ranged reference image and skips the LLM):

1. **SigLIP normalization**: feed `(pixel/255 - 0.5)/0.5` → `[-1, 1]`
   (preprocessor_config: mean=std=0.5), not `[0,1]`. Wrong range → garbage
   features → hallucination.
2. **Image features ×embedding_multiplier**: HF `LlavaNextForConditionalGeneration`
   scatters raw projector features into `inputs_embeds`, then the Granite LM
   multiplies the *whole* tensor (text + image) by `embedding_multiplier`
   (12.0). So spliced vision rows must be ×12 too — otherwise they are 12×
   weaker than text and the LM ignores the image. The one-space difference
   between two runs (with vs without the normalization change) was the tell:
   the image *was* reaching the LM but far too weakly to matter.

The Granite LLM decode itself (RoPE = HF `rotate_half` / ggml NEOX, attention
scaled by `attention_multiplier` = 1/64 not 1/√d, embedding/residual/logits
multipliers, tied lm_head) is validated layer-by-layer by
`tools/dump_granite_llm_reference.py` (builds the reference straight from the
dequantized GGUF — no 5 GB HF checkout needed) at cos=1.0.

## granite_vision: a self-consistent diff can hide a broken graph (2026-06)

CRITICAL methodology lesson. There are TWO references and they prove different
things:
- `tools/dump_granite_llm_reference.py` → `granite-llm-ref.gguf` is built from
  the **same dequantized GGUF** the C++ uses. Passing it (cos 1.0) only proves
  the C++ math matches a numpy reimplementation on identical weights. It CANNOT
  catch weight corruption or a wrong runtime backend.
- `tools/kaggle/granite-vision-parity/granite_vision_parity.py` →
  `granite-vision-ref.gguf` is built from the **real HF safetensors** (true
  parity). It catches weight AND code bugs. Download from
  `cstr/granite-vision-crispembed-GGUF` (37 MB) — no need to rerun on Kaggle.

A prior handover trusted the self-consistent LLM cos 1.0 + a stale "vision cos
0.99998" and concluded OCR garbage was a prompt/template bug. The HF-blueprint
diff instead showed the **Metal/ggml SigLIP ViT graph (`gv_run_vit_graph`)
outputs cos 0.05** while patch_embed (CPU scalar) is cos 1.0 and the scalar ViT
loop is cos 0.96. To localize, `granite_vision_dump_vision` was extended to emit
`vis_patch_embed` + per-feature-layer `vis_layer_N` stages, and
`CRISPEMBED_GRANITE_VIS_SCALAR/CPU` levers added: the break is at the first
encoder layer, identical across q4_k/q8_0 and flash/manual attn.

**CORRECTION (2026-06-21): the alloc-reuse diagnosis below was WRONG.** A later
session re-verified independently: on Metal the graph's input `x_t` is *intact*
after compute (not NaN), so there is no in-place buffer reuse. The "identical
across q4_k/q8_0 and flash/manual attn" evidence does *not* implicate alloc — it
equally fits a systematic weight-layout bug, since both test models carry **Q8_0
vision** weights and the bug is in the FFN, not attention. Real cause: see the
"Q8_0 reshape" entry directly below. The fix (cast the FFN weights to F32 before
the transposed-ne reshape) brings every `gv_run_vit_graph` layer to scalar
parity (cos 0.9996–0.99987 early; late layers track the scalar's 0.96 ref-eps
artifact) and runs ~3 s vs ~100 s scalar. Lesson preserved: a self-consistent
diff (`granite-llm-ref`) can't catch backend/weight bugs — only the HF-blueprint
`granite-vision-ref` can; AND a handover's stated root cause must be re-verified,
not trusted (here "NaN on CPU / alloc reuse" was never reproduced).

Also: the on-disk `*-q4_k.tok.gguf` had **Q4_0 vision weights** (quantized
before the `vis→Q8_0` fix in `tools/quantize.cpp`), collapsing scalar vision to
cos 0.32. Requantize from F16 with the current quantizer (now also keeps
`proj.*` at Q8_0) → vision parity matches q8_0. `projector_hidden_act="gelu"`
is exact **erf** (`ggml_gelu_erf`), NOT the vision tower's tanh.

## ggml_flash_attn_ext handles GQA natively (2026-06)

`ggml_flash_attn_ext` supports Group Query Attention without explicit
`ggml_repeat` to tile KV heads to match Q head count. The CPU kernel
(ops.cpp:8216) uses broadcast factors `rk2 = neq2/nek2` to map Q heads
to KV heads internally. This means you can pass K/V with `n_kv_heads`
directly — the kernel repeats automatically.

Before: 20+ tensor ops per layer (reshape_4d + new_tensor_4d + repeat +
reshape_3d, twice for K and V). After: zero ops, just pass nkv-head
tensors. Applied to internvl2, lightonocr, got_ocr, glm_ocr (-76 lines).

## F16 norm weights need ggml_cast in ggml graphs (2026-06)

Q4_K GGUF models store RMSNorm/LayerNorm weights as F16 (not F32).
When building ggml graphs, `ggml_mul(rms_norm_output, weight)` fails
with "binary_op: unsupported types: dst: f32, src0: f32, src1: f16"
because `ggml_rms_norm` always outputs F32 but the weight is F16.

Fix: `if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);`
before the multiply. This affected smoldocling's ggml LLM decoder and
got_ocr's patch embedding bias on Q4_K models.

## DenseNet ggml graph: ceil_mode pooling + concat dimension (2026-06)

Converting DenseNet encoders to ggml graphs requires careful handling of:

1. **ceil_mode pooling**: `ggml_pool_2d` uses floor mode. For ceil_mode
   equivalent, pad the spatial dims by 1 before pooling (`ggml_pad`).
   Getting this wrong causes spatial dimension mismatches in `ggml_concat`.

2. **DenseNet concat**: `ggml_concat(a, b, 2)` concatenates along ne[2]
   (channels in ggml's W,H,C layout). The assertion `a->ne[0]==b->ne[0]
   && a->ne[1]==b->ne[1]` means spatial dimensions must match exactly.
   Any pooling size mismatch cascades into a concat assertion failure.

3. **Conv weight layout**: ggml_conv_2d needs weights as F16 4D tensors
   [KW, KH, IC, OC]. Weights stored as 2D [IC*KH*KW, OC] need
   `ggml_reshape_4d` + `ggml_cast(F16)` before use.

hmer_ocr's ggml encoder is the working reference (`273969d`).
bttr/posformer share the same architecture but need separate
implementation due to different BN handling (folded vs separate).

## flash_attn_ext requires non-contiguous (permuted) inputs (2026-06)

`ggml_flash_attn_ext` reads Q/K/V strides from the tensor metadata to
handle multi-head layout. The standard pattern (from `vit_embed.cpp`):

```cpp
Q = ggml_permute(g, Q, 0, 2, 1, 3);  // [hd, T, nh] — non-contiguous view
// do NOT ggml_cont()!
attn = ggml_flash_attn_ext(g, Q, K, V, mask, scale, 0, 0);
attn = ggml_reshape_2d(g, attn, D, T);  // output [hd, nh, T] → [D, T]
```

Adding `ggml_cont()` before `flash_attn_ext` copies the data into a new
contiguous layout where the head/token axes are transposed from what
`flash_attn_ext` expects. This produces completely wrong attention output
(cos=0.56 vs reference). The fix: remove `ggml_cont()` and pass the
permuted views directly. Output is `[hd, nh, T]`, `reshape_2d(D, T)`
works directly without an extra permute.

## PIL ImageOps.pad: bicubic + clamp (2026-06)

PIL's `ImageOps.pad` calls `Image.resize(BICUBIC)` then pastes centered.
To match exactly in C++:

1. **Bicubic kernel**: Catmull-Rom (`a = -0.5`), NOT Keys (`a = -1`).
2. **Coordinate mapping**: `sx = (x + 0.5) / scale - 0.5` (center-to-center).
3. **Clamp output to [0, 1]**: Catmull-Rom overshoots at sharp edges (text
   boundaries produce values > 1.0 or < 0.0). PIL clips internally to
   [0, 255] before returning uint8. Without clamping, cos_min drops from
   0.9999 to 0.991 at the patch embedding stage.
4. **Padding value**: `int(mean * 255) / 255.0`, NOT `mean` directly.
   `int(0.5 * 255) = 127`, `127/255 = 0.498039 ≠ 0.5`.

## Never blame quantization for parity failures (2026-06)

Quantization (F16, Q8_0, even Q4_K) explains at most ~5% cosine
deviation from the F32 reference (cos_min ≥ 0.95). If parity is worse
than that, there IS a structural bug — a wrong resize method, a missing
ggml_cont, a wrong weight mapping, a wrong activation function, etc.

During the Unlimited-OCR port, hours were wasted blaming "F16 quantization
noise cascading through 24 CLIP layers" when the actual bugs were:
(1) `ggml_cont()` before `flash_attn_ext`, (2) bilinear vs bicubic resize,
(3) missing pixel clamp, (4) wrong BPE token ID. The BF16→F16 conversion
was LOSSLESS (max_abs_diff=0.000000 for all SAM weights checked).

Rule: when crispembed-diff shows cos < 0.95, always bisect with the diff
harness. Never accept "quantization noise" as an explanation.

## Unlimited-OCR decoder: prefill attention output was scrambled (2026-06)

The MoE decoder produced pure garbage ("by", or open-ended text collapsing
to multilingual noise) even though SAM/CLIP vision parity was fine. Root
cause was in the LLM self-attention output reshape:

```cpp
attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, mask, scale, 0, 0); // [hd, nh, T]
attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));            // BUG → [hd, T, nh]
attn = ggml_reshape_2d(g, attn, D, T);
```

`flash_attn_ext` returns `[hd, nh, T]`; the llama.cpp idiom is to
`reshape_2d` that **directly** to `[D, T]`. The extra `permute(0,2,1,3)`
turns it into `[hd, T, nh]`, so the subsequent `reshape_2d` interleaves the
head and token axes — *unless T==1*, where the permute is a no-op. That is
why the bug was invisible during single-token decode (the persistent-decode
path) but corrupted every multi-token **prefill**: a scrambled attention
output feeds the residual stream → next layer's input → the whole KV cache
is built from corrupted hidden states. Fix: delete the permute, reshape the
flash output straight to `[D, T]` (see `build_llm_layer_attn` and
`build_persistent_decode_graph`). After the fix the pure-text decoder is
coherent and the OCR path reads the first text region correctly
(`<|det|>text [x1,y1,x2,y2]<|/det|>Hello World`).

Two related gotchas found alongside it:

- **Persistent-decode KV cache must be zero-initialized.** The PD graph
  pre-allocates `max_kv` cache slots but fills only `n_past`; flash_attn
  reads the unused slots *before* the (-inf) mask is applied, and on the
  shared scheduler they hold leftover garbage from the vision graphs.
  `NaN + (-inf) = NaN`, which poisons every logit (→ endless `<bos>`). The
  short text-only test didn't trip it because its buffers were clean. Zero
  the whole `t_k_cache`/`t_v_cache` on the first PD step. (The PD path still
  drifts from the rebuild path after the first decode token on vision-heavy
  prefills — a residual Metal flash_attn numerics issue with the padded KV
  layout — so the verified-correct per-step rebuild path is the default;
  PD is opt-in via `UOCR_PD=1`.)

- **Instruction tokenization.** `"\nFree OCR."` byte-level-BPEs to
  `[Ċ=201, Free=21431, ĠOCR=126041, .=16]` (verified against the model's
  own tokenizer.json). Feeding the leading newline `Ċ=201` makes this port
  emit EOS immediately — a sensitivity that points at a small residual
  mismatch in the image-block embeddings (`image_newline`/`view_separator`)
  vs HF. The no-newline form reproduces correct first-region OCR. The grid
  is row-major and correct (a transpose wrongly maps the bottom image line
  to the top).

### Diff-harness findings vs `unlimited-ocr-ref.gguf` (vision stages)

With the reference GGUF (`cstr/unlimited-ocr-crispembed-GGUF/unlimited-ocr-ref.gguf`,
stages `sam_patch_embed` … `vision_features`) and `UOCR_REF=`:

- **Use `cos_mean`, not `cos_min`.** The letterboxed-gray padding tokens are
  near-zero vectors whose cosine is pure noise, dragging `cos_min` to ~0 even
  when the tensor is fine. `crispembed_diff::Report` already computes
  `cos_mean`; the engine now prints it.
- **THE REAL VISION BUG WAS THE PIPELINE SCAN-CLEANUP, not SAM.** The
  `--ocr-pipeline` path ran classical scan-cleanup (deskew/crop) on the image
  *before* the VLM engine, so the engine received e.g. **414×229 instead of the
  original 400×200** — a distorted aspect ratio that shifts content into the
  wrong vision-grid cells. That distortion was the entire `sam_patch_embed`
  0.98 / `sam_output` 0.73 "divergence". Disabling cleanup for VLM engines
  (`main.cpp`: `st.cleanup_enabled = is_vlm ? 0 : 1`) makes the engine receive
  the original image and the vision matches HF essentially perfectly:
  `sam_patch_embed` cos_mean **0.999999**, all SAM layers ≥0.9999,
  `vision_features` cos_mean **0.99981**. It also markedly improves real-page
  OCR (a book title page now yields title/author/degree/affiliation lines).
  VLMs do their own resize+letterbox; never pre-clean their input. (My earlier
  "the ref is a different image" guess was WRONG — re-dumping the ref on
  `test_ocr.png` on this M1 gave byte-identical numbers, proving the ref IS
  that image and the divergence was the cleanup-distorted *input*.)
- **SAM FFN GELU.** `build_sam_layer_graph` called `ggml_gelu` (tanh approx);
  HF SAM's `MLPBlock` uses `nn.GELU` (exact erf). Changed to `ggml_gelu_erf`
  for correctness (CLIP correctly uses `ggml_gelu_quick`; projector is
  `projector_type: "linear"`, no GELU). Effect is small.

- **The "decode failure" was the WRONG PROMPT + a missing logits processor —
  bad code, not quantization (q8/f16 were never the answer).** The port hard-
  coded the prompt as "Free OCR.", but this checkpoint's prompt (per the HF
  model card) is **"<image>document parsing."** ("Free OCR." belongs to a
  different DeepSeek-OCR checkpoint and makes this model emit its training-
  instruction boilerplate — "Do NOT use any punctuation. Treat all tabular
  layout as plain text…" — which is what looked like a decode bug). The card
  also calls `infer()` with **`no_repeat_ngram_size=35, ngram_window=128`**;
  that sliding-window n-gram block is *required*, not optional — without it the
  greedy detection-box decode gets stuck repeating a partial box. With the
  correct prompt (`[document=34030, Ġparsing=76466, .=16]`, no leading newline)
  and the sliding-window processor (`SlidingWindowNoRepeatNgramProcessor` over
  the full input_ids incl. the `<image>`=128815 placeholders), the q4_k decoder
  reads region 1 (`HelloWorld`) byte-for-byte like HF.

- **GROUND TRUTH: the real HF model reads `test_ocr.png` 4/4.** Ran the actual
  `baidu/Unlimited-OCR` via `AutoModel` on this M1 — **bf16 on CPU** (the fp32
  load is ~13 GB and *crashes* the 16 GB machine; bf16 ≈ 6.6 GB fits, with
  `.cuda()`/`autocast("cuda")` monkey-patched to CPU). Output:
  `<|det|>title [47,152,188,202]<|/det|>Hello World` … `This is a test` …
  `CrispEmbed OCR` … `2024-06-22` — all four lines. So the model + prompt +
  config are correct and the original image is the right input.

- **The remaining gap is the q4_k DECODER quant floor, localized with a
  decoder reference.** Extended the dumper (`dump_decoder.py`) to hook the
  HF decoder layers + `lm_head` and emit `llm_layer_0..11` + `logits`; the C++
  already diffs those stages under `UOCR_REF`. With the now-correct vision, the
  C++ prefill matches HF closely at the **dense** layer 0
  (cos_mean 0.9988, max_abs 0.34) but the **MoE** layers jump to max_abs ~7
  (cos_mean 0.997→0.979) and the final `logits` land at cos **0.926** — enough
  to flip a borderline greedy pick (line 2). This is NOT a code bug in the MoE:
  the math matches HF (verified op-by-op), CPU-MoE and Metal-MoE are
  byte-identical, the router/gate are Q8_0. It is the **q4_k quant of the
  experts AND the `lm_head` (both Q4_K in this file)** compounding over 12
  layers — the lm_head alone drops the logits from ~0.979 (hidden) to 0.926.
  **FIXED via the quantize recipe — and it was code, not "use f16".** The
  `lm_head` was being Q4_K'd; `tools/quantize.cpp` now keeps it at Q8_0
  (alongside the embeddings / vision / projector / MoE-router it already
  protected). Re-quantized from a fresh f16 (converted on this M1 with the
  refvenv; the converter output is byte-size-identical to the published f16)
  and the q4_k model now reads `test_ocr.png` **4/4, identical to HF**
  (`Hello World` / `This is a test` / `CrispEmbed OCR` / `2024-06-22`) and the
  real Maréchal book title page **near-perfectly** (full title, sub-title,
  author, degree, publisher block, year — only stray char errors). Cost: +90 MB
  (lm_head 82→168 MB) on a 2.1 GB file.

  Subtlety: the prefill `logits` cos barely moved (0.926→0.928) yet the OUTPUT
  went from cascade-hallucination to perfect — because the failure was a single
  borderline greedy flip early in generation that then snowballed; nudging the
  lm_head precision tipped that one decision. **The cos metric understated the
  fix; the OCR text is the real test.** (The experts stay Q4_K — bumping them
  too is unnecessary and would bloat the file.)

  Lesson (again): when a VLM emits coherent-but-off-task text, suspect the
  prompt/template, the sampling config, the *input pipeline* (don't pre-clean a
  VLM's image), and the *quantize recipe* (protect the lm_head / router /
  vision) — read the model card's exact `infer()` call — not the numerics, and
  never reach for a bigger-precision model. Prove a diff reference is on the
  SAME input before trusting it (re-dump it yourself; bf16 on CPU is enough —
  fp32 OOMs a 16 GB box). And judge by decoded text, not just cosine.

## glm-ocr: five real bugs behind "garbage OCR" — the handover's rope-only diagnosis was wrong (2026-07-01)

`glm-ocr` (zai-org/GLM-OCR, 0.9B) produced garbage OCR. The handover
(`handover-prompts/glm-ocr-vision-rope-fix.md`) said the single confirmed root
cause was a missing vision RoPE. That was a **mis-diagnosis on two counts**:
(1) it read `transformers/models/glm4v/modeling_glm4v.py`, but GLM-OCR is the
**distinct `glm_ocr` / `glm_ocr_vision` architecture** (only in transformers
`main`, not 4.57.x); (2) the reference dumps it told us to trust
(`glm-ocr-ref-*.gguf`) were generated by the **stale, no-rope** dump script, so
"C++ matches the ref with no rope" proved nothing. RoPE *was* needed, but it was
one of **five** independent bugs. End state: fox.png →
`"The quick brown fox jumps over the lazy dog. 12345"` on **f16, q8_0 and q4_k, CPU
and Metal**, matching the real transformers-`main` model exactly.

### The bugs (all verified against the real model)

1. **Missing vision 2D RoPE.** `glm_ocr_vision` applies Qwen2-VL-style 2D rope to
   Q/K in every ViT layer: `VisionRotaryEmbedding(dim=head_dim/2, theta=10000)`,
   per-patch freqs `[h·inv_freq, w·inv_freq]` (each `head_dim/4`), `emb =
   cat(rot,rot)` → `head_dim`, NEOX split-half rotate. CrispEmbed extracts patches
   in raster order and the merger is conv-based, so raster-order rope (each patch
   gets its true `(row,col)`) is exactly equivalent to HF's merge-window ordering
   under full (unmasked) attention — no reordering needed. Q/K after rope match the
   spec at cos 0.99999.

2. **Wrong merger structure.** `GlmOcrVisionPatchMerger` is
   `proj → post_projection_norm(LayerNorm, eps 1e-5) → act1(GELU **erf**) →
   down( silu(gate(h)) · up(h) )` with **no trailing norm**. The code (and the
   dump script) had `proj → SwiGLU → LayerNorm`. This alone made the image embeds
   uncorrelated with the model (cos ≈ 0). After the fix, C++ image embeds match
   the real `vis.merger` output at mean cos 0.99, identity order.

3. **Fixed 336×336 instead of dynamic resolution.** The processor is
   `Glm46VImageProcessor` — Qwen2-VL smart-resize (`min_pixels=12544`,
   `max_pixels=9633792`, dims a multiple of `patch·merge = 28`), **not** a fixed
   square. CrispEmbed squashed every image to 336×336, destroying aspect ratio
   (fox.png is 800×200, 4:1 → unreadable). Fix: smart-resize + a *variable* grid
   flowing through patchify → rope → merger → image-token count → LLM image mRoPE.
   fox.png → 812×196, grid 14×58, matching the reference `image_grid_thw`
   `[1,14,58]`.

4. **LLM image mRoPE positions.** `get_rope_index`/`get_vision_position_ids`:
   text token → all 3 dims = `current_pos` (++ per token); image patch `(row,col)`
   → `temporal=start, h=start+row, w=start+col` (`start` = text pos at the image);
   after the block `current_pos += max(gh,gw)`. Because image tokens **compress**
   positions (144 tokens advance the position by only `max(gh,gw)`), decode must
   continue from the compressed position, not the token count — track it in
   `ctx.mrope_next_pos`. `mrope_section=[16,24,24]` (contiguous MROPE, not
   interleaved). The old code used `h=ih, w=iw` (no `start` offset) and froze the
   text position across the image.

5. **Prompt + EOS + decode.** Correct prompt (from `chat_template.jinja` on the
   recommended message `[{image},{text:"Text Recognition:"}]`, trim/lstrip blocks,
   `add_generation_prompt=True`):
   `[gMASK](59248)<sop>(59250)<|user|>(59253)\n(10)<|begin_of_image|>(59256)
   <|image|>(59280)×N<|end_of_image|>(59257){Text Recognition:=3649,7404,49600,58}
   <|assistant|>(59254)\n`. The old prompt dropped `[gMASK]`, dropped the
   instruction, and injected a spurious empty `<|system|>`. Also: stop on **both**
   eos ids `[59246 <|endoftext|>, 59253 <|user|>]` (the model ends an OCR turn with
   `<|user|>`), and implement **GPT-2 byte-level decode** (reverse
   `bytes_to_unicode`) so pieces render as text instead of `Ġ`/`Ċ`.

### q8_0 / q4_k: dequantize BEFORE reshaping (same class as got-ocr `11c2bc7`)

q8_0 (and q4_k) produced garbage embeds on CPU *and* hit the Metal
`GGML_ASSERT(ne00 % ggml_blck_size == 0)` abort. Root cause: the downsample
weight was `ggml_reshape_4d(w, 2,2,D,out_D)` **before** `ggml_cast(w, F32)` — a
q8_0 tensor reshaped to leading dim 2 splits its 32-element blocks and corrupts
the data (and trips the Metal block-align assert). Cast q8→F32 **first**, then
reshape. Do the same up front for any quantized weight that is reshaped or fed to
a conv (downsample, patch_embed, merger matmuls). q8_0 weights themselves were
fine (dequant vs f16 = cos 0.99997) — the gguf was not corrupt. q4_k works with
the same fix (its HF upload was re-done after the earlier all-zeros corruption);
f16, q8_0 and q4_k all OCR fox.png correctly on CPU and Metal.

### Massive-activation "sink" tokens: why the per-token cos_min diff gate can't hold

This ViT has huge outlier activations (residual `max_abs` grows to ~1900 by the
last layer). On the crispembed-diff **synthetic-gradient** image a handful of
"sink" tokens develop those extremes and are then largely cancelled downstream.
Findings (all measured):
- numpy f32 vs f64: cos_min 0.99999 through L23 → **not** compute precision.
- numpy **f16 weights** vs f32 weights: cos_min 0.9999 through L23 → **not**
  weight precision; the function is well-conditioned to both.
- **C++ (ggml) vs numpy f32** on the synthetic image: cos_min collapses to ≈0 at
  L13+, while **median stays 1.000**. So it's ggml's matmul/softmax **reduction
  order** (SIMD accumulation) diverging from BLAS's blocked/pairwise summation on
  those catastrophic-cancellation tokens — a benign implementation difference.
- On a **real** image (fox.png) C++ vs the real model is median 1.000 / min 0.904
  (no collapse) and OCR is exact. The smooth synthetic gradient is adversarial
  precisely because it maximizes those outlier/cancellation tokens.

Consequence: a per-token `cos_min` diff gate is unsuitable for glm-ocr even
against a perfect reference. The glm-ocr `diff` block was removed from
`tests/regression/manifest.json` (the runner treats `diff` as opt-in per model,
so this affects only glm-ocr); correctness is guarded by the `expected_text`
end-to-end check. If per-layer coverage is ever wanted back, change the *metric*
(median/mean, or early-layers-only), not the engine.

### Methodology: the real model as ground truth, not the stale refs

The decisive move was building ground truth from the actual model instead of the
stale dumps: `pip install git+…/transformers` (main has `glm_ocr`) into a venv,
download `zai-org/GLM-OCR` (~1.8 GB), run `AutoModelForImageTextToText`, and diff
each stage. Key gotchas: hook `vis.merger`'s output for the true image embeds —
`get_image_features().last_hidden_state` returns something else and is *not* the
merger output; compare C++ vision `post_norm` to the real one as a **permutation**
(raster vs merge-window) before comparing merged embeds directly. C++ debug envs
added along the way: `GLM_OCR_ROPE_DEBUG` (dump post_norm / layer-0 Q,K),
`GLM_OCR_DUMP_EMBEDS`, `GLM_OCR_VISION_ROPE=0`, `GLM_OCR_VISION_FLASH`,
`GLM_OCR_VISION_F16MM`, plus the pre-existing `CRISPEMBED_GLM_OCR_SCALAR_MERGER`
and `GLM_OCR_FORCE_CPU`. `tools/dump_glm_ocr_reference.py` now has both the vision
rope and the corrected merger, so regenerated refs match the engine on early
vision layers and both LLM layers at cos 0.9999 (late layers per the sink-token
caveat above). See also the "Self-consistent crispembed-diff reference" and
"Never blame quantization" entries — and the meta-lesson: **independently
reproduce a handover's root-cause claim before building on it.**

## 2026-07 — pcs full ONNX parity + an encoder-parity audit sweep

Chasing a shipped q4_k crash in the `pcs` punctuation engine turned into full parity
work against the ONNX source (`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`,
run via onnxruntime — `tools/dump_pcs_reference.py`). Six root causes, each a distinct
class:
1. **q4_k crash** — the CPU-side SBD/truecase heads read quantized FC weights via raw
   `ggml_backend_tensor_get(..., n*sizeof(float))`, overrunning `ggml_nbytes` → abort. Fix:
   read via a per-row dequant (`to_float` trait, sized by `ggml_nbytes`), or the shared
   `core_cpu::to_f32`. Quantizer DOES quantize 2-D weights incl. `token_embd`, so any
   CPU-side weight read must dequantize.
2. **Tokenizer (dominant)** — XLM-R is SP **Unigram**; greedy longest-match mis-split
   multi-subword words → embeddings cos as low as 0.13. Fix: Unigram Viterbi over
   `tokenizer.ggml.scores` (converter now emits them; new `core_gguf::kv_f32_array`).
3. **Decode** re-counted subtokens greedily → dropped final punctuation on multi-subword
   words; now partitions the actual token_ids by ▁ word-start.
4. **SBD** used argmax; ONNX thresholds `softmax P(boundary) > 0.05`.
5. **Truecase** conditioning used the current token's sbd; ONNX feeds the SHIFTED
   is-sentence-initial flag (`argmax seg[t-1]`).
6. **Encoder numerics**: `ggml_gelu` (tanh) where XLM-R uses exact erf → `ggml_gelu_erf`;
   LayerNorm eps `1e-12` → `1e-5` (RoBERTa/XLM-R; BERT genuinely wants 1e-12).
After all six, tok+post+pre+seg predictions match ONNX 11/11, encoder hidden cos 0.999997,
q8_0/f32 exact. Diagnostics: `PCS_DEBUG`, `PCS_FORCE_CPU`, `PCS_DUMP_HIDDEN`,
`PCS_DUMP_LAYER`, `PCS_FLASH_ATTN`. GGUFs re-uploaded to `cstr/pcs-xlmr-base-GGUF` (+q8_0).

The pcs classes then generalised across the codebase:
- **GELU tanh→erf** wherever `hidden_act="gelu"` (exact erf): `gliner_ner.cpp` (DeBERTa-v3),
  `lilt_kie.cpp` layout FFN (text FFN was already erf — asymmetric miss). Verify each
  against the real `config.json`; SigLIP/CLIP genuinely use tanh/quick_gelu.
- **Quant-read crash class**: `crispembed.cpp`'s MLM/SPLADE head read the quantized
  `token_embd` / `mlm_transform_w` as raw F32 → now `core_cpu::to_f32` (the shared
  backend-safe dequant in `src/core/cpu_ops.h`). The fused-QKV merge was already guarded
  (`if (L.q_w->type != GGML_TYPE_F32) continue`).
- **fullstop-punc** (XLM-R via the fireredpunc SP path) had the full pcs set (greedy
  tokenizer + eps 1e-12 + tanh GELU) — fixed (inline Viterbi + conditional eps + erf),
  verified exact vs HF, GGUFs re-uploaded. The dead `src/{pcs,fireredpunc}.cpp` fallback
  duplicates (built only when the shared `crisp_punc` lib is absent) were unified.
Note the two `pcs.cpp` copies (CrispASR `crisp_punc` = shipped; CrispEmbed `src` = fallback)
must stay logically in sync; each repo's `.clang-format` differs so byte-identity is
impossible — sync the logic. See the memory `pcs-cpp-two-copies-diverged`.

## Cached ggml graphs must own their metadata pool (WASM crash → native segfault, #31)

`math_ocr` cached the encoder graph across calls (`ctx->enc_graph` /
`ctx->enc_batch`) but built it inside a ggml context whose `mem_buffer` was a
**stack-local `std::vector`** — freed as soon as the build block's scope
closed, while the cached graph (and every tensor struct in it) still pointed
into the dead buffer. Classic use-after-free with wildly different symptoms
per allocator:

- **WASM (dlmalloc)**: the freed 16 MB block is immediately handed back to
  the CPU backend as its mul_mat work buffer, so quantize_row_q8_0's
  activation writes land **on top of the cached tensor structs** —
  `src1->data` becomes float garbage (odd pointer like `0x1f826019`) and the
  next row read traps `memory access out of bounds`. This is why every
  ViT-class OCR (pix2tex, TrOCR) "exceeded WASM limits" — it never did; it
  was heap corruption.
- **macOS malloc**: usually silent (different size class → freed region not
  reused), but segfaulted reproducibly on some inputs (dbnet+trocr pipeline
  on a 520×260 crop, exit 139).

Fix (one line per site): pass `mem_buffer = nullptr` so **ggml owns the
pool** and `ggml_free(ctx->enc_graph_g)` releases it —
`ggml_init_params ip = { meta_size, nullptr, true };`

Debug method that found it: Playwright browser e2e (weak node tests had
`passed++` around the crash!) → emcc `-g2 -sASSERTIONS=2` build for a
symbolized stack → per-row fprintf of `srcp/dstp/src1->data` in the mul_mat
quantize loop, which showed wdata's write range covering the `src1` tensor
struct address. Rule: **any** ggml graph cached beyond the building scope
must have a context-owned (or ctx-member-owned) metadata pool; a local
buffer is only safe when build + compute complete within the same scope.
Audit note: `precompute_cross_kv` and the decoder loop use local pools but
compute in-scope — legal. The scan for `std::vector<uint8_t> meta` +
`ctx->…_g = g` found no other offenders.

## Emscripten + ggml: CMAKE_SYSTEM_PROCESSOR is "x86" → WASM SIMD kernels silently dropped

Under `emcmake`, the Emscripten toolchain sets `CMAKE_SYSTEM_PROCESSOR=x86`
(bitness advertisement), so ggml-cpu's arch dispatch hits the "Unknown CPU
architecture → generic implementations" branch and **never compiles
`arch/wasm/quants.c`** — every quantized vec_dot/quantize ran scalar even
though `-msimd128` was set (that flag only lit up the `__wasm_simd128__`
blocks in TUs that have them; the generic quants file has none). Symptom in
stacks: `quantize_row_q8_0` tail-calling `quantize_row_q8_0_ref`. Fix: pass
`-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` (officially supported toolchain
override) → "Wasm detected", arch file compiles, ~1.5-2× on q4_0/q4_K
OCR inference. Applies to any ggml-based wasm build (CrispASR too).

Related wasm-demo architecture (same session): inference must run in a Web
Worker — single-threaded WASM on the main thread freezes the tab for the
whole compute and is indistinguishable from a hang (user report on #31).
Engine stderr, forwarded via `CRISPEMBED_MODULE_OPTS.printErr` →
postMessage, doubles as live progress ("ocr_pipeline: recognizing region
i/N"). For threads on static hosting (GitHub Pages can't set headers), a
COOP/COEP-injecting service worker (`coi-sw.js`) + one guarded reload makes
the page crossOriginIsolated; `controllerchange` (the SW calls
clients.claim()) is the reload signal — polling `controller === null` after
`ready` is racy AND wrong (claim() sets the controller without the document
having the headers).


## Emscripten 6 pthreads inside a Web Worker — two deadlocks and their shims

Running a `-pthread` Emscripten module INSIDE a dedicated worker (module
importScripts'ed into our own worker script) hits two silent hangs:

1. **Factory-inside-onmessage deadlock.** If `CrispEmbedOCR()` (the
   MODULARIZE factory) is first invoked from within an active `onmessage`
   handler, the pthread pool bootstrap never completes — the factory promise
   just never resolves (no error). Same code at worker top level works.
   Pattern: pass config via the worker's query string
   (`ocr-worker.js?loader=...`), importScripts + instantiate at top level,
   stash `globalThis.CRISPEMBED_MODULE_PROMISE`, and have the wrapper's
   `_initModule` reuse it.

2. **Pthread workers spawn OUR worker script.** `mainScriptUrlOrBlob` no
   longer exists in emscripten 6; pthread workers are spawned from
   `_scriptName = self.location.href` — which is the OUTER worker's URL when
   the module was importScripts'ed. Symptom: N nested workers spawn running
   the host worker script, the module's pool wait hangs, and the module
   logs "worker sent an unknown command <x>" for the host's own postMessages.
   Fix: make the host worker **pthread-transparent** — first thing in the
   script: `if (self.name === 'em-pthread') { importScripts(LOADER); }` and
   nothing else (the Emscripten loader, evaluated under that worker name,
   runs its own pthread bootstrap and owns the worker).

Also: `locateFile` must return ABSOLUTE URLs in worker contexts (relative
paths abort with XHR "Invalid URL" in blob workers, and threaded builds live
in a subdirectory). Debug recipe: wrap `self.Worker` before importScripts to
log spawn URLs; forward `printErr` via postMessage.


## ggml WebGPU backend in the browser (emscripten) — porting notes

Shipped as an experimental opt-in tier for the WASM OCR demo (~2.2× vs the
SIMD CPU build for pix2tex on M1, output byte-identical). What it took, in
order of discovery:

1. **ggml snapshot 8be60f8's WGSL templates break the shader embedder** —
   `*.tmpl.wgsl` files get embedded as invalid C identifiers
   (`wgsl_cpy.tmpl`). Upstream later renamed them to plain `*.tmpl` (the
   embed script only globs `*.wgsl`); build-wasm.sh --webgpu mirrors that
   rename in the submodule working tree (idempotent).
2. **JSPI exports**: GGML_WEBGPU_JSPI defaults ON → any export that can
   reach GPU work suspends and MUST be listed in `-sJSPI_EXPORTS` (else
   "trying to suspend without WebAssembly.promising"), and JS must call it
   via `ccall(..., {async:true})`. The wrapper's `_acall` awaits every
   engine-touching call — `await` normalizes plain builds (raw value) and
   JSPI builds (Promise), so one wrapper serves all variants.
3. **Resizable-heap vs browser APIs, round 2**: Chrome rejects
   `GPUQueue.writeBuffer` with views into a resizable ArrayBuffer — the
   exact class as the issue-31 TextDecoder crash. WebGPU build uses
   `-sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=512MB`.
4. **Encoder graph cache is NOT re-entrant across sched resets** (again —
   same class as Parakeet §176s): 2nd recognize on the cached graph traps
   `unreachable` on WebGPU. math_ocr now always rebuilds the encoder graph
   (build cost is µs next to compute). This cache also caused the issue-31
   UAF — it is a bug magnet; do not reintroduce without a re-entrancy test.

Op coverage note: ggml-webgpu (this snapshot) has MUL_MAT/FLASH_ATTN/
SOFT_MAX/ROPE/GET_ROWS/unary but NO GGML_OP_NORM (classic LayerNorm) and no
IM2COL — ViT LayerNorms run on CPU via sched splits (still nets 2.2×);
DBNet conv detection gains little until those shaders exist (upstream-able).


## WebGPU LayerNorm kernel (patches/ggml-webgpu-layernorm.patch)

ggml-webgpu's `row_norm.wgsl` is a clean template (RMS_NORM / L2_NORM
variants); LayerNorm (GGML_OP_NORM) is a small delta: second workgroup
accumulator for the plain sum, `var = E[x^2] - mean^2`, and the update
helper gains a shift param (`dst = scale * (src + shift)`, shift = -mean;
0 for the existing variants). Wiring: pipeline-getter case + encode dispatch
+ supports_op (eps already at op_params[0], same as RMS_NORM). Kept as a
git patch applied idempotently by build-wasm.sh --webgpu (the submodule
tracks upstream ggml-org/ggml). Same-conditions A/B, pix2tex e2e (M1,
Chrome headless): 2.46-3.11 s -> 1.67-1.79 s (~1.4x; ~2.8x total vs SIMD
CPU), output byte-identical. Candidate for upstreaming to
ggml-org/llama.cpp (per AI-policy: mechanical disclosure, human-written
prose). DBNet detection still runs its conv stack on CPU — needs IM2COL +
CONV_TRANSPOSE_2D + POOL_2D + UPSCALE kernels (a real upstream project,
deferred).


## WebGPU OCR kernels round 2 — conv stack + the silent-skip trap

Six WGSL kernels (NORM, IM2COL, POOL_2D, CONV_TRANSPOSE_2D, UPSCALE,
ARANGE) now carried as patches/ggml-webgpu-ops.patch; upstream-PR draft at
CrispASR tools/upstream-prs/22-webgpu-ocr-ops.{md,patch}. Hard-won lessons:

- **ggml-webgpu's encoder silently SKIPS unhandled ops** (default: returns
  nullopt = no-op). On the sched-less compute path (ocr_detect uses raw
  ggml_backend_graph_compute) this yields silently wrong output — DBNet
  "detected 0 regions" because its 7 UPSCALE nodes were dropped. The patch
  adds a stderr warning; when debugging "wrong results on webgpu", grep for
  SKIPPING first.
- **ggml test-backend-ops runs in headless Chromium**: link
  ggml/tests/test-backend-ops.cpp against the build-wasm-webgpu static libs
  (em++ --use-port=emdawnwebgpu -sJSPI -fwasm-exceptions, fixed heap), load
  in a page with Module.arguments=['test','-o',OP,'-b','WebGPU']. Per-op
  validation vs CPU with proper tolerances — found real bugs decimal-literal
  (-FLT_MAX must be bitcast<f32>(0xff7fffffu); WGSL rejects -3.4028235e38),
  missing sf3 (batch-dim rescale), and non-contiguous src (stride_src0).
  Nobody upstream executes browser tests in CI; this setup exceeds it.
- **WebGPU dispatch is capped at 65535 workgroups/dimension** — conv
  lowerings exceed it; dispatch 2D with an nwg_x uniform and linearize.
- Pipeline A/B (scan strip, M1, back-to-back): CPU 291 s vs GPU 164 s
  (1.78x); detection 90 s -> 1.5 s (~60x); same boxes, two borderline
  region texts differ (F16 rounding; GPU text closer to native GT on one).
  Autoregressive TrOCR decode is only mildly faster on GPU (JSPI round-trip
  overhead per step) — batching decode steps is the next perf lever.


## OCR-engine WebGPU sweep + OPFS cache (July 5)

Per-engine CPU-vs-WebGPU sweep in headless Chromium
(tests/wasm-browser/engine-sweep.js), single-model wasm API, warm numbers:

| engine    | wasm CPU | WebGPU | note |
|-----------|----------|--------|------|
| pix2tex   | 6.3 s    | 2.4 s  | 2.6x |
| trocr     | 6.9 s    | 1.7 s  | 4.0x — biggest win |
| parseq    | 0.17 s   | 0.65 s | correct, but tiny model = CPU wins |
| hmer      | 13.0 s   | 11.4 s | 1.15x |
| bttr      | 7.8 s    | 9.0 s  | GPU slightly slower |
| tesseract | 0.16 s   | 0.17 s | tiny LSTM, parity |

All six now produce text matching CPU. Two bugs found:
- **parseq returned garbage on WebGPU** ("MMM"): it computes on a raw
  gallocr (no sched/CPU fallback) and uses ggml_flash_attn_ext, which
  ggml-webgpu compiles OUT under Emscripten *inside its case* — so even our
  default-case skip warning didn't fire. Fix: exact manual attention under
  __EMSCRIPTEN__ in parseq_ocr.cpp (+ tensor-pool bump for the extra ~16
  nodes/layer), and a warning added to the flash-attn Emscripten branch in
  the ggml patch. Rule: any engine that computes WITHOUT a sched must not
  emit ops the webgpu backend lacks — flash_attn_ext is the trap.
- Manual attention overflowed parseq's exactly-sized ggml metadata pool —
  "not enough space in the context's memory pool" — pools must budget for
  per-platform graph variants.

**OPFS model cache** (wllama-pattern, in wasm/crispembed-ocr.js):
opfs://crispembed-models/<encoded-url>, awaited write (a fire-and-forget
write is killed if the page navigates right after load — cost us a
head-scratcher), navigator.storage.persist() attempt, clear API +
demo link. Verified: second page load hits cache, zero network.


## WebGPU compat tier + WebKit findings + decode profiling (July 5)

- **--webgpu-compat** (GGML_WEBGPU_JSPI=OFF -> Asyncify, 4.6 MB vs 3.3 MB
  JSPI): for browsers with WebGPU but no JSPI. Demo picks by
  `typeof WebAssembly.Suspending === 'function'`; `?gpuCompat=1` forces it.
  Verified 15/15 in Chromium (Asyncify path exercises the same wrapper
  _acall contract — ccall {async:true} behaves on all three build flavors).
- **WebKit (Playwright 26.5)**: JSPI **shipped** (WebAssembly.Suspending
  exists — real Safari 26 may run the JSPI GPU build once WebGPU is
  exposed); navigator.gpu absent in headless; OPFS reads work but writes
  throw transient UnknownError in the ephemeral test profile (wrapper
  degrades to no-cache correctly). CPU tier verified on WebKit:
  ground-truth match.
- **coi-sw must not proxy model downloads**: WebKit terminates service
  workers mid-stream ("Service Worker context closed"), killing 17 MB
  fetches routed through respondWith. The SW now only stamps same-origin
  document/script/wasm responses — COEP on the document already covers
  CORS-mode subresource fetches.
- **TrOCR phase profiling (warm, M1)**: encoder CPU 3896 ms -> GPU 714 ms
  (5.5x); decoder CPU 48 ms -> GPU 231 ms (5x SLOWER — per-token
  JSPI/submit overhead on tiny matrices). Next lever, deferred with data:
  encoder-on-GPU + decoder-on-CPU split (needs decoder weights resident on
  both backends) or cross-region batched decode for the pipeline.


## Decoder-on-CPU split for WebGPU (MATH_OCR_DEC_CPU=1)

math_ocr can duplicate the decoder weights into a CPU buffer (second
load_weights pass into wl_dec; `dec.*` lookups in map_tensors route there),
which steers the sched to run the autoregressive decode on CPU while the
encoder stays on GPU. Enabled by the demo worker for the webgpu tiers.

Measured (M1, warm): single-model TrOCR decode 216 -> 48 ms (= pure-CPU
decode speed; total 918 -> 886 ms). Pipeline (8 regions): 164 -> 160 s —
essentially a wash: the pipeline's per-region decode does long cross-attn
against 578-token encoder outputs, which amortizes the per-token GPU
submit/suspend overhead that dominates short decodes. Bonus: region-text
parity with CPU improved (7/8 vs 6/8). Kept ON for webgpu (small consistent
win, strictly faster decode, ~model-size extra wasm-heap for the CPU copy).
Prediction vs reality note: the profiling suggested a bigger pipeline win —
always A/B the actual workload, not the microbench.


## Engine sweep round 2 (12/12) + a wasm-CPU drift lesson

posformer/texo/mixtex/ppformulanet/texteller all MATCH CPU on WebGPU;
texteller-3 q4_k (177 MB) fits the fixed 512 MB webgpu heap and is the
biggest GPU win (5.4x). trocr-small-handwritten DIFFERED between wasm-CPU
and wasm-GPU on a printed-word (out-of-distribution) fixture — the NATIVE
engine sided with the GPU output: the wasm-CPU SIMD accumulation was the
drifting leg. Lesson: on borderline inputs, 'differs from wasm-CPU' is not
'GPU is wrong' — always arbitrate with the native engine.

## Two-detector consensus deskew: opposite signs and a resolution-dependent bias (2026-07-06)

`scan_cleanup` gained a consensus mode (`deskew_consensus`, default on) that
cross-checks the Hough-energy angle against the independent Leptonica-style
differential-square-sum detector (`classical_preproc.h`, `find_skew_angle`)
before rotating. Two empirical facts anyone touching this code needs
(verified on synthetic rotations, `tests/test_scan_cleanup.cpp`):

1. **The two detectors use OPPOSITE sign conventions.** After
   `scan_cleanup_rotate(+3°)`, Hough reports `+3.0` while DSS reports
   `-3.5`. Map with `-dss` before comparing.
2. **DSS overestimates the magnitude with a resolution-dependent bias** —
   ~0.5° on 800px pages, ~1.2° on 400px — because it binarizes and reduces
   4× before the shear sweep. Its SIGN is always reliable. A fixed 1.0°
   agreement tolerance therefore silently rejects genuine ~3° skews on
   small images (the failure looked like "consensus never confirms"); the
   shipped gate is sign agreement + a 1.5° magnitude band. DSS also only
   sweeps ±7°, so Hough angles above 6° pass through uncross-checked.

The consensus detector also backs `scan_cleanup_deskew_rgb` (bilinear,
channel-preserving, white-fill rotation), which is the building block for
the optional per-params deskew on all image-embedding paths
(`crispembed_set_image_deskew`, `vit_embed::set_deskew`, CLI `--deskew`;
off by default). One observed caveat: for CLIP-style photo models on a
synthetic page, deskewing moved the embedding FURTHER from the straight
original — the expanded white corner wedges perturb a square-resize photo
model more than 3° of skew does. Deskew-for-embeddings is a
scanned-document feature; keep it opt-in.

While mirroring the new param into bindings: the Rust `from_stages`
`ScanCleanupParams` literal was missing the four despeckle/blackfilter
fields — an E0063 compile error, i.e. the crate could not have built since
those fields were added (there is no Rust CI). Fixed by basing the literal
on `crispembed_scan_cleanup_defaults()` via struct-update syntax so future
field additions inherit defaults instead of breaking the build.

## Runtime perf: measure the DOMINANT cost before "fixing" a flagged micro-gap (2026-07-11)

An audit / code-review flags mechanical gaps — "weights re-dequantized every
call", "`n_threads` ignored", "graph rebuilt every step". Each is real, but in a
runtime that is **scalar-compute- or dispatch-bound**, fixing the gap moves a
tiny fraction. A runtime re-verification sweep hit three in a row where the
flagged gap was NOT the bottleneck:

- **esrgan `n_threads`**: the real thread-count sink is `fn(be, 1)` at
  `esrgan_sr.cpp:266`, which clobbers the init-time count before every compute.
  Wiring it to honor `n_threads` made decode SLOWER — `-t 8` 33s vs `-t 1` 21s
  (bit-identical output). esrgan tiles into 128px pieces; a per-tile conv is too
  small to amortize thread overhead and oversubscribes 4 P-cores. Contrast
  **safmn**, where the *same* one-line fix gave a real **2.3×** — because safmn
  convolves the WHOLE image in one graph, so its convs thread-scale. Thread
  scaling depends on op size, not on whether the flag is wired.
- **decode-step graph cache** (billed the "#1 lever"): measured on trocr-small
  (lightest decoder, D=256/V=1200) — build+alloc 0.47 ms/step vs compute
  18.5 ms Metal / 6.9 ms CPU → **2–6%**. And build cost is ~constant per step
  (fixed node count) while compute grows with `n_kv`, so the fraction only
  shrinks. The real decode cost is per-op dispatch (Metal), not graph build.
- **scunet uncached dequant**: `to_f32` per Swin block is O(weights); the block
  itself is O(H·W·C) scalar WMSA+MLP per pixel. Caching saves a small fraction
  × tiles — marginal.

Also: the audit's "flip the CPU-pinned SR engines to `init_best` for free GPU"
was a **mirage** — they are CPU-pinned *deliberately* (conv weight residency;
`esrgan_sr.cpp:115`), and NO SR engine runs conv on GPU (all use a CPU
`enc_sched`; `swinir_sr.cpp:447` literally prints `ggml_conv_2d (CPU sched)`).
And tbsrn's PE2D was already cached (`tbsrn_sr.cpp:425`).

**Rule:** before implementing a flagged micro-optimization, measure or reason
about the DOMINANT cost of the hot path — build-vs-compute split (env-gated
timers on a real model), thread-scaling (op size), tile count. The two real wins
of the sweep (safmn whole-image threading 2.3×; tps_locnet dequant hoist for
reuse-callers) were both where the gap WAS a meaningful fraction; the marginal
ones were not. The genuine remaining levers are the scalar-compute hot paths
themselves (SIMD/ggml-ify scunet/mixtex WMSA, layout_detect deformable attn) and
Metal per-op dispatch (ggml-metal ICB replay) — not the mechanical gaps around
them. Verify which case you're in first.

## Two more measured instances + a byte-identical SIMD trick (2026-07-11)

More applications of the rule above, plus a technique for verified perf wins:

- **gliner DeBERTa encoder is GPU-execute-bound, not dispatch-bound.** The
  shared ggml `CRISPASR_METAL_PROFILE` probe split the 942-node encoder graph
  into host-encode ~3.3 ms vs GPU-execute ~70–90 ms → **~96% GPU / ~4% host**.
  So the "cut ggml_cont/permute, fuse ops" lever (which only touches host-encode)
  was the WRONG one. The real cost: the disentangled c2p/p2c position matmuls
  projected the full `[H, T*T]` pair-grid through `k_w`/`q_w`, but only `≤2T-1`
  DISTINCT relative-position buckets exist. Fix: project the unique buckets once
  (`[H, n_used]`), then `ggml_get_rows` to expand — output-identical, **1.28×
  (T≈40) to 1.71× (T≈90)**, win scales O(T²−T). Same "reuse an invariant instead
  of recomputing per element" shape as the rel-pos CPU cache, moved into the graph.

- **layout_detect Phase-2 decoder cost is the scalar `cpu_linear` matmul, NOT
  the deformable-attention sampling loop** (my first guess — corrected by the
  bench). The deformable sample loop is ~17M MACs; `cpu_linear` (`:1018`) is a
  scalar stride-N matmul at up to 256×256×8400, ~10×/layer×6 = ~3–5G MACs. Guess
  cost, then measure; the loud-looking loop wasn't the cost.

- **Byte-identical SIMD via AXPY reordering** (the technique). A `[out,N] = W[out,in]
  @ X[in,N]` matmul written as `for n,o: sum_i W[o,i]*x[i*N+n]` strides `x` by N
  (cache-hostile, un-vectorizable) — the layout_detect hot loop. Reordering to
  `for o: y[o,:] = b[o]; for i: y[o,:] += W[o,i]*x[i,:]` makes `x[i,:]`/`y[o,:]`
  contiguous over N (vectorizes) while keeping the **per-output accumulation order
  (i ascending) identical → byte-identical**, verified by an empty `diff` of the
  region output. This is strictly better than routing through `core_cpu::dot_product`
  / `linear_batch_cpu` when you want byte-identity: those do a SIMD horizontal/
  pairwise reduction that changes summation order (cos≈1, not exact). Measured
  **~1.26× on Phase-2** (best-of A/B of two back-to-back binaries — the only valid
  timing method on a box that was at loadavg 20–137 with competing crispasr/
  flutter; best-of because noise only inflates times). AXPY was also far more
  load-STABLE than the strided baseline (better cache behavior under memory
  contention) — a side signal that the access-pattern change, not just SIMD width,
  was the win.

- **`ggml_conv_2d_direct` is a SLOW Metal kernel for large-spatial shapes; use
  `ggml_conv_2d` (im2col+GEMM).** The layout_detect backbone (RT-DETRv2 @ 640²)
  ran its 505-node graph at ~99.6% GPU-execute, 11.7s gpu_us — pure Metal conv.
  Swapping the two conv sites from `conv_2d_direct` to `conv_2d` (im2col + `mul_mm`
  GEMM) cut Phase-1 **11.4s → 1.2s (~9.8×)**. Metal's simdgroup `mul_mm` is highly
  optimized; the direct-conv kernel is not, for these shapes. Output is cos≈1 (both
  F32 — the fork's `ggml_conv_2d` picks an F32 im2col when the kernel is F32; only
  reduction order differs → ≤0.001 score / ≤0.1px bbox jitter, clears the cos≥0.99
  regression gate). This is the conv-heavy-engine case of the dev-doc's "conv_2d
  port wins for conv-heavy engines" note — flip the default there, gate the direct
  path (`LAYOUT_CONV_DIRECT=1`). **Candidate for the CPU-pinned SR family too** if
  they ever move conv to Metal. `conv_2d_direct`'s only edge is memory (no im2col
  buffer) — irrelevant at a fixed 640² input.

- **The conv-swap win was engine-specific: whole-network conv (layout) wins, a
  neck/patch conv in a decode-dominated model does NOT.** Sweeping the other
  `conv_2d_direct` / `conv2d_cpu` users (got_ocr SAM-neck, glm_ocr patch-embed,
  ppformulanet/unlimited/bttr necks) — they're all a small neck/patch stage inside
  a transformer+decoder model. **got_ocr measured**: neck_projector 396 ms of
  6738 ms (~6%, convs a fraction of that) while **decode = 92%** (499 × ~12 ms).
  So the conv swap is ~4% at best — the flagged-micro-gap-that-isn't-dominant case;
  don't churn it. And the got_ocr **decode step is ~89% GPU-execute / ~11% host**
  (metal-prof: 940 nodes, 2.5 ms encode / ~20 ms gpu synced) → **compute-bound**,
  so the decode-step graph cache / ggml-metal ICB (which only remove host/dispatch)
  cap out at ~10–17% there — modest, and a risky project. Lesson: the layout
  backbone was special because the ENTIRE net is conv (measured 99.6% GPU / 11.7s);
  measure the fraction before assuming a repeat of a win.

- **The ggml gitlink/symlink dance bit twice this session** — see the `git stash`/
  `git checkout` reset trap; a fresh-worktree build needs the ggml symlink, git ops
  need the gitlink, and stash/checkout silently swap it back to the empty gitlink
  → stale-binary. Re-`ln -s` after any tree-touching git op before rebuilding.

## Decode-step graph cache + CPU-decoder wins (2026-07-11, cont.)

Five decoders got a gated **sched-free `ggml_gallocr` decode-step cache**, plus
threading/dequant wins on two CPU engines. The durable lessons:

- **Sched-free decode cache: reserve a `ggml_gallocr` once at max KV length, then
  compute decode steps via `ggml_backend_graph_compute(backend, gf)` — not the
  sched.** A decode step's graph has constant node count (only tensor shapes grow
  with `n_past`), so a gallocr reserved for the longest graph takes
  `ggml_gallocr_needs_realloc`'s no-realloc fast path every step, and the
  sched-free compute skips `ggml_backend_sched_split_graph` (the per-step host
  cost). Shipped byte-identical on got_ocr, internvl2, glm, lightonocr, math_ocr
  (each behind `<ENGINE>_DECODE_CACHE=1`, default OFF). Measured host build+alloc
  ~0.85→0.28 ms/step (got_ocr, quiet) → ~3% on light decoders, ~0% on heavy ones
  (compute-dominated). **Its real value is load-insensitivity:** the sched's
  `alloc` balloons to ~4.3 ms/step under load while the gallocr stays flat ~0.1 ms.
- **Cache DECODE steps only (`n_past > 0`), never prefill.** got_ocr's prefill is
  a separate code path, but internvl2/glm/lightonocr route prefill through the same
  `run_cached_step`. Sending the prefill graph (image splice / full-seq mRoPE — a
  DIFFERENT node count) through a decode-shaped gallocr reservation + sched-free
  compute corrupts output: **glm degenerated into repetition** until the `n_past>0`
  gate was added (internvl2 survived by luck). Always gate on `n_past>0`.
- **"Single-graph decoder" is necessary but NOT sufficient — the decode graph must
  also be single-BACKEND.** qwen2vl looked ideal (single graph, already
  constant-shape) but `GGML_SCHED_DEBUG=2` showed its **attention runs on CPU**
  (per-layer `SPLIT: CPU`) while the rest is Metal. A sched-free
  `graph_compute(ctx.backend=Metal)` forces those CPU ops onto Metal → empty
  output. And with a constant shape there's no realloc to skip, so nothing to gain.
  Reverted. Check `GGML_SCHED_DEBUG=2` for CPU splits before attempting the cache.
- **On an autoregressive decoder, check for CONSTANT WORK re-run per step before
  optimizing any kernel.** mixtex's CPU decoder re-ran ~11 `to_f32()` weight
  dequantizations per layer on EVERY of 30 steps (converting the same f16 weights
  ~120×). Hoisting them into a once-built f32 cache: **decoder 2923→1008 ms
  (~2.9×)**, byte-identical (same-binary A/B via `MIXTEX_DEC_DEQUANT_PER_STEP=1`).
  My first guess (thread the 25681-wide vocab projection) measured as ~4% of the
  step — a dud. The redundant dequant was 65%. Same shape as the rel-pos / weight
  dequant caches: an invariant recomputed in the hot loop. **Audited the other CPU
  OCR decoders for the same bug — clean:** ppformulanet / ppformulanet_l hoist all
  `to_f32()` before their decode loops; bttr / posformer / hmer / parseq have none
  (ggml graphs). mixtex was the lone offender; no need to re-grep.
- **For an already-SIMD scalar kernel over INDEPENDENT units, the next lever is
  loop-level threading, not more SIMD.** mixtex's Swin window attention was already
  dot-product-SIMD; the encoder bottleneck was the **serial per-window loop** (270
  independent windows). Threading it (each `window_mhsa` self-contained, disjoint
  output) → **encoder 1420→733 ms (1.94×)**, byte-identical. Same for layout's
  `cpu_linear` (thread the independent output-row `o`-loop → Phase-2 ~1.24–1.49×,
  partly memory-bandwidth-bound so sub-2×). Both now **honor `ctx->n_threads`**
  (default `-t` = 1 → old serial behavior), the mixtex/layout analog of the safmn
  n_threads fix. Byte-identical by construction (disjoint writes, unchanged
  per-unit math) — verify with a `-t 1` vs `-t 8` diff.
- **Prove the parallel path actually engaged.** A best-of-3 briefly showed mixtex
  threading giving *zero* speedup — a shell-function bug where `-t 8` silently
  didn't reach the binary. Added a one-line `n_threads=N` / `cache active` stderr
  marker (bench-gated) to each change so a run *proves* the fast path ran before
  any timing is trusted (a loaded box already fabricates numbers; a mis-set flag
  fabricates worse).

## surya-det: the default path was broken since the port; only the A/B reference path was ever verified (2026-07-11)

Found while auditing `core_cpu::conv2d_cpu` consumers: **every default-path
surya-det detection aborted** with `GGML_ASSERT(a->ne[2] == b->ne[2])` at
`ggml.c:4472`. Root cause: LiteMLA's `agg_pw` is a *grouped* pointwise conv
(groups = 3·heads — neither depthwise nor groups=1) and `g_conv` routed it
through `ggml_conv_2d`, which has no groups support. The depthwise branch
(`groups == IC` → `ggml_conv_2d_dw`) masked the gap for `agg_dw`, so the
grouped-but-not-depthwise case only exists on this one conv — and it asserts
at graph-BUILD time (hard abort, no fallback possible).

Two lessons:

1. **Verify the DEFAULT path, not just the reference path.** A pre-merge
   build (06c02ee) crashes identically, so this is not a regression — the
   graph path has been broken since the port. The port's parity was evidently
   established via `SURYA_DET_SCALAR=1` (the CPU reference), and the
   regression manifest has no surya entry, so nothing ever exercised the
   path users actually get. Same genus as verify-handover-claims: the
   "working engine" claim was true only for the A/B baseline path.

2. **A 1×1 grouped conv is a batched matmul, not N small convs.** The fix
   expresses `y[g·OCg+oc, hw] = Σ_icg w·x` as ONE `ggml_mul_mat` on
   `[ICg, OCg, groups] × [ICg, HW, groups]` with two cont+permutes — ~6
   graph nodes total vs ~5 *per group* for a split-conv-concat loop (48
   groups at stage3 would have blown the 2048-node gf2 budget and added ~96
   Metal dispatches). Verified: 39/39 boxes byte-identical to the scalar
   reference on Metal AND forced-CPU; the restored graph path is ~2× the
   scalar fallback (18.5 s vs 38 s end-to-end on scan_page_pd).

## The build dir was silently CPU-only for 5 days — check the cache before calling anything "Metal" (2026-07-12)

Found while verifying the `--gpu-backend` sweep: the main working tree's
`build/` had `GGML_METAL:BOOL=OFF` (`GGML_AVAILABLE_BACKENDS=ggml-cpu`),
configured 2026-07-07 — almost certainly by the CPU-only sandbox session that
built the 4D encoder batch (its PLAN note even says "this env is CPU-only,
GGML_METAL=OFF"). It stuck: every measurement made with the main binary from
07-07 to 07-12 that was labelled "Metal" actually ran on CPU. `init_best`
falls back without any error, and on a CPU-only BUILD there is not even the
"embedded metal library" stderr line to miss — silence is the only symptom.

Casualties corrected:
- **The 4D-batch "Metal verdict" was wrong twice over**: the "Metal parity
  failure" (0.99989 < gate) and the mixed-length throughput loss were CPU
  numbers. On a real Metal build, 4D parity PASSES (0.9999996) and **packed
  is 5–7× vs sequential everywhere** (uniform and mixed, interleaved bench)
  — packed is the Metal batching mode; 4D is the CPU tool. PLAN C3 rewritten.
- **Cross-binary baselines were conflated** (CPU main vs Metal worktree):
  yesterday's "mixtex 19 s → 5 s", "layout 21 s → 3.4 s", "gliner 6.7 → 3.1 s"
  wall-clock comparisons overstate the code wins — the branch's own
  same-binary isolated numbers (1.94×, 9.8×, 1.28–1.71×) are the real ones.
  The "gliner Metal score wiggle" was actually CPU-vs-Metal backend diff.
- **ppformulanet-L re-measured** on Metal: encoder 31 s → 8.3 s, so the CPU
  neck is ~18% of total (not ~10%) and the decoder (~69%) is the dominant
  cost. conv2d_cpu skip verdict stands; the engine's lever is its decoder.

Rules: (1) before any backend-attributed measurement, verify
`GGML_METAL:BOOL=ON` in the build cache AND `MTL0` in the run's stderr;
(2) never compare wall-clock across binaries from differently-configured
build dirs — same-binary env-toggled A/B or nothing; (3) after any sandbox
session, assume the build config may have been downgraded.

Bonus bug the same day: `tests/test_encoder_batch.py`'s 4D parity class
leaked `CRISPEMBED_ENCODER_4D=1` into the throughput test, so its "seq" and
"packed" legs silently ran the 4D path. Pop the mode envs at bench start.

## A reference you wrote yourself can be wrong, and then per-stage cosine proves nothing (2026-08-02)

PP-OCRv6 had been "validated" the usual way: per-stage cosine against
`tools/dump_ppocrv6_reference.py` at 0.9999, Metal-vs-CPU agreement, gold
archives, a passing CI lane. It could not read a page. On a clean rendered
pangram it emitted `iiiiii` / `laúieyotiieieioieioni.` at `mean_conf=0.94`, and
an earlier session had recorded the fox-crop decode `涨RiI` in PLAN.md without
treating it as a failure.

The reason every gate passed is that the reference was a **hand-written torch
mirror of a Paddle model**, and the runtime had been debugged until it matched
*the mirror*. Both were wrong in the same places, so every cosine was 1.0 and
every A/B agreed. This is the failure HARD RULE #3 exists for: the decoded
output is the only acceptance test, and nobody had read one.

**What actually settles it: `git clone` the upstream repo.** Reading
`ppocr/modeling/backbones/rec_lcnetv4.py`, `ppocr/modeling/necks/rnn.py` and
`tools/infer/predict_rec.py` took minutes and produced five facts that no
amount of tensor diffing could have:

1. `StemBlock` is built from `ConvBNAct`, whose activation is `ReLU()` — the
   mirror guessed SiLU.
2. The light-SVTR neck's `[1,7]` local conv is a **residual** (`z = z +
   local_conv(z)`), not a replacement.
3. `skip_conv` is computed first but added **after** the SVTR blocks and the
   neck norm; the mirror added it before the blocks and never after.
4. `max_wh_ratio` is *seeded* with `imgW/imgH` and then **grows** to the widest
   crop, so `[3,48,320]` is a **floor**, not a cap — a 520x35 line is 713 px
   wide. Capping at 320 crushed 44 characters into 40 CTC timesteps, which is
   undecodable no matter how correct the graph is.
5. `use_space_char: true` makes the label list `blank + dict + ' '`, which is
   why the head has 18710 outputs against an 18708-entry dict. The missing
   class decoded to nothing, so every space was dropped.

Fixing those took the lane from noise to **CER 0.0031 over 20 fixtures — the
most accurate engine in the comparison**, ahead of system Tesseract (0.0256)
and PaddleOCR's own Python pipeline (0.0185).

Durable rules:

- **A config that does not pin a hyperparameter is a warning, not a licence to
  guess.** `config.json` had no `num_attention_heads`, no activation placement,
  no pooling. Each guess type-checks, runs, and produces plausible activations.
- **Blind sweeps cannot recover a structural error.** Before reading the source
  I swept 16 head counts x 2 activations x 3 poolings, then 4 stem x 5
  depthwise x 4 channel activations — ~200 configurations, judged by decoded
  text. Best was CER 0.667. The output was nearly *insensitive* to head count,
  which correctly said "the error is upstream in the backbone", but no
  combination of tunables could express the missing residual or the misplaced
  skip. Sweeping is a localiser, never a fix.
- **Preprocessing is part of the blueprint.** Two of the five bugs (input width,
  space class) were outside the model graph entirely, in the exact
  harness-blind zone the diff harness ends before.
- **When native and reference agree and both look wrong, believe the output.**
  The converse also held here, usefully: the surviving `e`->`c` confusion on a
  31 px line reproduced byte-identically in the Python reference, which is how
  we know *that* one is model capacity and not a port bug.

## Five "hard blockers", zero hard bugs: every wall in the PP-OCRv6 ledger was a small defect with a big wrong label (2026-08-04)

One session closed five long-standing blockers, and the post-mortem pattern is
identical for all of them — the recorded diagnosis was wrong, the actual defect
was small, and the fix was found by reproducing the claim rather than working
from it:

| recorded label | actual defect | fix size |
|---|---|---|
| "recognizer quality gap vs official ($→S, I→:)" | the pipeline's scan-cleanup (despeckle/blackfilter, no CLI switches) eroded thin strokes BEFORE detection | 1 carve-out |
| "1.48x warm speed gap vs official python" | stage-bench `detect` spanned stage entry, folding a ~1.1 s Metal init into a "load-excluded" column | timestamp base |
| "detector graph geometry divergence (31 vs 30 boxes)" | a stray second `ggml_scale(gate, 0.2f)` squashed one SE hard-sigmoid to 0.04x+0.5 | 1 line |
| "detector graph 2.6-6.8x slower than scalar" | `DET_GRAPH` implied GPU load, so the graph had only ever been timed on Metal; on CPU it was 1.5-1.7x FASTER | measurement |
| "Metal fourth-dimension pooling limitation (batch graph)" | the fused caller passed a zeroed value into an in-out width parameter; every batch graph was built at width 0 and asserted on ANY backend | seed 1 param |

Durable rules from the wreckage:

- **Reproduce the failure on a different backend before accepting a
  backend-specific theory.** The "Metal pooling" abort reproduced on CPU in
  one run, which killed the theory instantly. A limitation that is not
  backend-specific is a bug wearing a backend costume.
- **When a lane misreads but a direct harness reads fine, diff the INPUT each
  one saw before diffing any model stage.** The $→S bisect never needed a
  single tensor: recognizer-on-raw-crops was correct, direct-harness-on-the-
  cleaned-image reproduced the corruption byte-for-byte. The bug was upstream
  of the model entirely.
- **Audit what a bench line actually spans before building comparisons on
  it.** `engine_ms` claimed load-excluded and was not (ppocrv6 stage-bench
  spanned stage entry; the easyocr harness regex captured the load-inclusive
  `total=` although the line itself printed a correct `detect+recognize=`).
  One inflated column produced a whole quarter's "we are 1.48x slower" verdict.
- **Check which backend a "graph is slow" number was measured on.** The env
  flag that enabled the graph also switched the backend to GPU, so the two
  variables were never separated. Same-graph CPU-vs-Metal on these conv shapes
  differs by 9x — in CPU's favour.
- **In-out shape parameters must be seeded at every call site.** The batch
  lane passed output-only variables (zeros) where the build expected the input
  width. The graph built "successfully" and failed deep inside ggml with an
  assert that pointed at pooling, three abstraction layers from the cause.
- **Externally-runnable references end "unrecoverable" stalemates.** The
  official-v6-locally-unavailable premise (true in 2026-08-01) silently
  expired when paddleocr 3.x and community ONNX exports shipped; nobody
  re-checked. Two pip installs turned an "unfalsifiable" quality question into
  a 2-minute experiment. Re-date your impossibility claims.

## A reference dump that mirrors your preprocessing proves nothing about the input contract (2026-08-04, T15 SmolDocling)

The engine sat at tensor parity 0.9999 for months while shipping payload CER
0.86 — because `tools/dump_smoldocling_reference.py` hand-resized the image to
512x512 exactly like the C++ did, instead of running the reference pipeline's
own processor. Both sides of the diff harness saw the same WRONG input (the
real contract is Lanczos longest-edge-2048 → 512-multiple round-up → 512² tiles
+ global view + `<row_r_col_c>` prompt structure), so every stage cosine was
perfect and the decoder's duplicated-region hallucination looked like a "model
quality" issue. Three stacked defects hid there: the converter dropped all 145
added tokens (detok silently deleted every `<loc_N>`; `decode()` skipped
out-of-range ids without a whisper), preprocessing squashed the page, and a
parity-era `max_tokens=128` TODO truncated everything — none visible to the
harness, all visible in one run of the model-card `AutoProcessor` + `generate`
next to the CLI. Durable rules: (a) the reference arm for an OUTPUT gate must
be the reference implementation's OWN pipeline, processor included — a
matched-preprocessing dump is only valid for isolating graph math, and must
never be quoted as lane parity; (b) prompt-token parity vs the real processor
(dump `input_ids`, compare byte-for-byte) is a cheap structural gate that
catches template/tiling drift instantly — SmolDocling passed 347/347 only
after the port; (c) a detokenizer that `continue`s on out-of-range ids
converts a truncated vocab into silently-missing markup — fail loudly or log
once. Bonus symptom table: "model duplicates regions" on non-square pages =
suspect aspect-destroying preprocessing before suspecting the decoder.

## "Non-ASCII means letter" is a Unicode approximation that silently retokenizes German (2026-08-04, `feat/tokenize-simple-audit`)

Every byte-level BPE pre-tokenizer in this repo approximated `\p{L}` as "ASCII
letter, or any byte >= 0x80". That is not a rounding error, it is a different
regex. The Qwen/LFM2/DeepSeek pattern all key off `\p{L}` and its complement, so
the approximation moves token boundaries the moment a non-ASCII **punctuation**
character appears — and the languages that need this repo's German retrieval
work are exactly the ones that use them:

```
sagte „Hallo“ heute   HF: sagte | Ġ„ | Hallo | “ | Ġheute      (5 pre-tokens)
                      us: sagte | Ġ„Hallo“ | Ġheute            (3)
«quote»               HF: Â«quote | Â»          us: Â«quoteÂ»
€£abc                 HF: âĤ¬Â£ | abc           us: âĤ¬Â£abc
中文，测试。            HF: 中文 | ，测试 | 。      us: one piece
```

This survived the T19-E1 fix — that commit replaced the whitespace-collapsing
`tokenize_simple` with a real declared-regex pre-tokenizer and a 39-check
HF-golden guard, and it was *right about whitespace*. The battery just had no
non-ASCII punctuation in it, so the residual defect passed 39/39. **A guard
proves only what its cases contain**: when you transcribe a regex that
references Unicode general categories, put a case in for each category you
approximate, or you have tested the ASCII subset and shipped the rest.

Fix: classify against the real categories. `tools/gen_unicode_class.py` emits
`src/core/unicode_class.h` (774 ranges, binary-searched) for `\p{M}`, `\p{N}`,
`\p{P}|\p{S}` and White_Space, defaulting to letter — the default is safe
because unlisted non-ASCII really is letters (CJK, Cyrillic, Greek, Arabic,
Hangul, Devanagari). `\p{P}` and `\p{S}` share one class deliberately: every
regex in `core/bpe.h` uses them only as the union.

**Fuzz the transcription, don't just fixture it.** 40 curated cases found the
bug; 4000 random mixed-script strings compared against HuggingFace's own
`pre_tokenize_str()` are what proved the replacement correct (0 mismatches for
all three families), and 1508 strings run through the real vocab+merges proved
the *ids* match end to end. Both harnesses are ~40 lines of Python driving a
tiny stdin/stdout C++ binary. Build one before believing a hand-transcribed
regex.

## `std::priority_queue` ties are unspecified — HuggingFace BPE breaks them leftmost (2026-08-04, `feat/tokenize-simple-audit`)

`core_bpe::bpe_one` ordered its merge heap by rank alone:

```cpp
auto cmp = [](const PQEntry & a, const PQEntry & b) { return a.first > b.first; };
```

`std::priority_queue` gives no ordering guarantee among equal keys, so a run of
equal-rank pairs could merge from anywhere in the middle. HuggingFace's BPE
orders its heap by `(rank, pos)` **both ascending**, i.e. leftmost wins a tie.
Result: `"qqqqqc"` with one merge rule came out `qq q qq c` instead of
`qq qq q c` — a genuinely different token id, in a code path shared by every
byte-level BPE engine in the repo. Measured: 4 of 1508 random strings per
vocab, on all three tokenizers tried. The fix is one line
(`a.first != b.first ? a.first > b.first : a.second > b.second`).

Two things made it hard to see. It is **rare** (~0.3%), so any pass/fail
fixture misses it. And it is **not reproducible from short inputs**: with three
symbols the heap happens to come out leftmost anyway, so the obvious four-token
regression test passes on the broken code. The guard needs a five-symbol run
(`tests/test_bpe_pretokenize.cpp`) — a reminder that a test which cannot fail
on the pre-fix build is not a guard, whatever it is named.
