# Handover: load community `modern-bert` (BPE-tokenizer) encoder GGUFs

## Objective
A community/llama.cpp GGUF of a ModernBERT embedding model
(`Alibaba-NLP/gte-modernbert-base`, arch `modern-bert`) does not load in
crispembed, and the naive loader fix makes it load but emit **garbage**. Make it
load AND produce correct embeddings, validated per-stage against the original HF
model, then add it to the community matrix. This is bigger than a #33 alias add —
the real blocker is **tokenizer support for community BPE encoder GGUFs**.

Repo: `/Users/christianstrobele/code/CrispEmbed`. Work in a **git worktree**
(never the main tree); `main` build dir is `build/`. Commit + merge directly to
`main` (no PRs, no Co-Authored-By trailer — retracted for this repo). Read
`CLAUDE.md` + the shared dev guide `../crispasr-crispembed-dev.md` first.

## The model (already downloaded)
- Community GGUF: `~/crispembed-live-cache/gte-modernbert-base-q8_0.gguf`
  (from `eranmazur/gte-modernbert-base-Q8_0-GGUF`).
- HF reference: `Alibaba-NLP/gte-modernbert-base` (needs `trust_remote_code`).
- If gone, re-fetch with `huggingface_hub.hf_hub_download`. For an f16 control
  (step 5) pull `gte-modernbert-base-f16.gguf` if a community one exists, else
  convert one.

## Repro (CONFIRMED 2026-07-16)
```bash
./build/crispembed -m ~/crispembed-live-cache/gte-modernbert-base-q8_0.gguf --prefix "" --json "hello world"
```
On plain `main`: aborts with `missing required tensor layer=0 name=ln1.weight`
(×22). With the loader-alias patch below applied it LOADS (22L/768d, RoPE
θ=10000, pre-LN) but the embedding is garbage:
`cos(related)=0.068 < cos(unrelated)=0.157` (negative margin).

## Root cause (CONFIRMED — two layers deep)

**Surface (loader) — necessary but NOT sufficient.** crispembed already has the
ModernBERT graph (`ggml_geglu`, `modernbert_swa_enabled`, global-attn-every-N,
dual RoPE theta). The community GGUF just uses names/keys crispembed doesn't map:
- tensors: `blk.N.attn_norm.weight`→`ln1`, `blk.N.ffn_norm.weight`→`ln2`,
  `output_norm.weight`→`final_norm`; `blk.N.ffn_up.weight` is **[768, 2304] =
  [H, 2*inter]** (the fused GeGLU weight named `ffn_up`, feed_forward_length=1152)
  — must be routed to `ffn_up_gate_w` by SHAPE, not name.
- metadata (arch `modern-bert`): `pre_ln=true` (architectural); RoPE is INVERTED
  vs crispembed's naming — `rope.freq_base=160000` is the GLOBAL theta,
  `rope.freq_base_swa=10000` is LOCAL, so `rope_theta`=freq_base_swa,
  `rope_theta_global`=freq_base; `attention.sliding_window=128`→
  `local_attention_window`; `attention.sliding_window_pattern=3`→
  `global_attn_every_n`; `pooling_type=2` (CLS, A2 maps it).

**The real blocker (tokenizer).** With the loader patch, the per-stage HF diff
(`tests/test_encoder_diff.py`) shows the **structural gate itself failing**:
`emb_ln_out cos=0.583` (|ours|=40.8 |ref|=30.8) — divergence is BEFORE block 0, so
it is tokenization/embeddings, not the graph. Cause: the GGUF declares
`tokenizer.ggml.model = "gpt2"` (BPE) + `tokenizer.ggml.pre = "modern-bert"`, with
**no** `tokenizer.ggml.type` and merges in the `tokenizer.ggml.merges` KV STRING
ARRAY. crispembed's dispatch (`src/crispembed.cpp`, the `tokenizer.ggml.tokens`
block ≈ L470–540) reads ONLY the numeric `tokenizer.ggml.type` (default 0) plus an
`n>100000→SPM` heuristic; 50368 vocab + absent type → it picks **WordPiece**. HF
tokenizes to BPE ids `[50281, 25521, 1533, 50282]`; crispembed's WordPiece
produces different ids → different embeddings → garbage from token 0.
(e5-small/granite-107m only load correctly by luck: their ~250K vocab trips the
`n>100000→SPM` heuristic, which happens to be right for them.)

## The fix — IN THIS ORDER, each gated by the per-stage harness

Do NOT trust "it loads / it emits a vector" at any step (that is how the garbage
shipped-looking result arose). The gate at each step is a per-stage cosine.

1. **Tokenizer dispatch** (`src/crispembed.cpp`, tokenizer block). When
   `tokenizer.ggml.type` is absent, map `strv("tokenizer.ggml.model")` →
   type: `gpt2`→BPE(1), `bert`→WordPiece(0), `t5`/`unigram`→SPM(2). Make the model
   string authoritative over the vocab-size heuristic.
2. **BPE merges from the KV array.** Community gpt2 GGUFs store merges in the
   `tokenizer.ggml.merges` KV STRING ARRAY (confirmed present), not the
   `tokenizer.merges` TENSOR the current BPE path reads. Load them from the KV
   array for these models.
3. **gpt2 byte-level BPE + `modern-bert` pre-tokenizer.** Verify crispembed's BPE
   (`src/tokenizer_bpe.cpp`, `core/bpe.h`) does gpt2 byte-to-unicode mapping and
   the `modern-bert` pre-tokenizer regex. **GATE:** dump the reference (below) and
   confirm `emb_ln_out` cos reaches **~0.99999** — the structural gate — BEFORE
   interpreting any per-layer number. Also assert the token IDS match
   `[50281,25521,1533,50282]` on both sides (a shifted-token mismatch mimics
   numeric drift — see the dev guide's "gate input alignment" note).
4. **ONLY THEN the graph.** With tokenization correct, re-run the per-stage diff.
   If a specific layer diverges it is the FFN or attention:
   - **GeGLU variant.** Current code uses `ggml_geglu` (tanh, `gelu(first)*second`).
     ModernBERT uses EXACT gelu → try `ggml_geglu_erf`; the split order depends on
     llama.cpp's `ffn_up` layout → the 4 candidates are
     `ggml_geglu{,_swapped,_erf,_erf_swapped}`. Pick the one that matches per-stage.
     Gate it on arch/shape so the existing GTE-v1.5 path (which uses `ggml_geglu`
     and is validated) is untouched.
   - RoPE local/global assignment (step-1 metadata) and the SWA mask.
5. **Validate + guard.** Per-stage q8_0 vs HF must pass; run the f16 precision
   control (`tests/prove_quant_control.py`) to prove any residual gap is quant; add
   a `tests/community_gguf_matrix.json` entry (hf_repo, min_hf_cos, control_file).

## Tools (all exist on `main`, all reproduce for this model)
- `tools/dump_encoder_reference.py --model Alibaba-NLP/gte-modernbert-base
  --trust-remote-code --text "hello world" --output /tmp/ref.gguf` (uses a
  forward-hook fallback since ModernBERT rejects `output_hidden_states`; set
  `HF_HOME` to a writable dir — the default symlink is often unmounted).
- `CRISPEMBED_DUMP_LAYERS_GGUF=/tmp/ours.gguf ./build/crispembed -m <gguf>
  --prefix "" --json "hello world"` dumps our per-stage tensors.
- `python tests/test_encoder_diff.py --ours /tmp/ours.gguf --ref /tmp/ref.gguf`
  reports the first divergent stage (structural gate + per-layer, prints
  `|ours|`/`|ref|` so a same-name/wrong-quantity harness bug is visible).
- `python tests/prove_quant_control.py --name gte-modernbert-base` (after a
  matrix entry with `control_file`).

## Loader-alias code (CORRECT, reverted on `main` — re-apply as step-4's prereq)
In `src/crispembed.cpp`: after the `pre_ln`/rope/`position_buckets` reads, add
```cpp
if (gguf_arch == "modern-bert") {
    ctx->rope_theta          = opt_f32({ ak("rope.freq_base_swa") }, 10000.0f);   // local
    ctx->rope_theta_global   = opt_f32({ ak("rope.freq_base") }, 160000.0f);      // global
    ctx->global_attn_every_n = opt_u32({ ak("attention.sliding_window_pattern") }, 3);
    ctx->local_attention_window = opt_u32({ ak("attention.sliding_window") }, 128);
    ctx->pre_ln = true;
}
```
`m.final_norm_w = get_any({ "final_norm.weight", "output_norm.weight" });`
`ln1_w` add alias `blk + "attn_norm.weight"`; `ln2_w` add alias
`blk + "ffn_norm.weight"`. After fetching `fc1_w`/`ffn_up_gate_w`, add the
GeGLU-by-shape reroute:
```cpp
if (!L.ffn_up_gate_w && L.fc1_w && L.fc1_w->ne[1] == 2 * (int64_t) hp.n_intermediate) {
    L.ffn_up_gate_w = L.fc1_w; L.fc1_w = nullptr; L.fc1_b = nullptr;
}
```
⚠ **These aliases MUST NOT ship without steps 1–3** — alone they turn the loud
`missing tensor` failure into a silent garbage embedding.

## Landmines
- Machine: 16 GB M1; never run >1 heavy model/build at once; the box may be loaded
  by parallel sessions (check `sysctl vm.loadavg` before heavy work).
- ModernBERT layer 0's attention norm is Identity in HF — `ln1_w` may be absent for
  layer 0; the pre-LN path already guards `if (pre_ln && L.ln1_w)`, so keep `ln1_w`
  per-layer optional.
- The GGUF's `general.architecture` is `modern-bert` (with a hyphen).
- Full prior diagnosis is in `PLAN.md → "FOUND (2026-07-16): community modern-bert"`.
