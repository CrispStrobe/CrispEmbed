# Layer0 attention op-by-op same-input replay v1

The [FFN replay](../tests/results/packaging/wasm-layer3-ffn-matmul-v1.json)
showed the layer3 failure is inherited input difference amplified by activation
quantization. In the v4 capture, the earliest bit difference is the norm01
input (1.19e-7): the layer0 attention residual. The norm00 output (the
embedding LayerNorm) is bit-identical across native, wasm0 and wasm1. This
protocol replays every layer0 attention op from that shared input, on both
builds, to find the first op whose bits differ on identical inputs.

Production MiniLM layer0 (post-LN BERT, no RoPE, no relative bias, a single
unpadded sentence) runs: `ln_mul`/`ln_add` (embedding LayerNorm scale/bias),
`q/k/v_raw = mul_mat`, `+bias`, `ggml_flash_attn_ext(q,k,v, no mask, 1/sqrt(32))`,
`o_raw = mul_mat`, `+bias`, and `residual = ln_add + o` (the norm01 input). At 12
query rows the CPU flash path is `flash_attn_ext_f16_one_chunk`; the tiled path
needs at least 64 rows and split-KV needs one row. It uses `ggml_vec_dot_f32`
for Q·K, scalar `expf` online softmax, and `ggml_vec_scale_f32`/`ggml_vec_mad_f32`.

[Workflow](../.github/workflows/wasm-layer0-attention-v1.yml),
[caller](../tools/ci-heavy/wasm_layer0_attention_v1.py),
[analysis](../tools/wasm-layer0-attention/analysis.py),
[probe](../tests/wasm-layer0-attention/probe.cpp).

1. Same production-profile admission as the FFN replay (kernel files equal the
   40852835039 production inventories; probe `-ffp-contract=off`).
2. Weight-free controls before model access, with authored IQ4_NL weights and
   F32 vectors: each build's chain is deterministic across two runs, and every
   isolated op fed the build's own chain reproduces that chain bit-exactly.
3. Pinned v4 captures and the frozen model. Both chains must reproduce their own
   captured norm01 input bit-exactly, and isolated ops must equal the chain.
4. Per op: the WASM op on native's inputs vs native, the native op on WASM's
   inputs vs WASM, and the propagated chain difference. The report names the
   first same-input differing op, all independently differing ops and their
   kinds. A default-off candidate is considered only if one deterministic op
   kind explains every difference.

No candidate is enabled; no gate or default changes.

## Attempts

- First dispatch attempt: no run was created. The first registered workflow
  had invalid YAML (an unquoted colon in a step name), so dispatch returned 422.
  Fixed at `8f7ea115`.
- [41441512356](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41441512356)
  at `8f7ea115` completed ([receipt](../tests/results/packaging/wasm-layer0-attention-v1.json)).
  All controls pass, and both chains reproduce their captured norm01 input
  bit-exactly. On identical inputs every elementwise op is bit-exact. The
  **first differing op is `q_raw`** (IQ4_NL x Q8_0 mul_mat, 9.54e-7: tinyBLAS
  vs generic scalar order). `k_raw`, `v_raw` and `o_raw` (same kernel) and
  `flash_attn_ext` (8.94e-7: `vec_dot_f32`/`vec_mad_f32` FMA and lane width,
  `expf`) differ independently. Two op kinds are involved, so no single-op
  candidate is proposed.
