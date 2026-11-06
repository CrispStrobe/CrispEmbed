# LightOn4B upstream blueprint and quantization audit

This is a static source comparison and integer head-index check, not a new model
run or quality result. It finds no demonstrated missing architectural operation
in the inspected native LightOn path. Arithmetic differences and quantization
remain separate measured issues. No production code or model recipe is changed.

Upstream is pinned to llama.cpp
`bbc2e4b65705132843b9d142f4471dfb8085b233`, inspected6 November2026.
The [receipt](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-blueprint-audit.json)
binds exact source-file hashes and the bounded integer check. Local runtime
comparison uses PR155`248e38f14`; qualification evidence uses`6525e045e`.

The official[Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
shares all15 checked text-geometry fields with immutable LightOn source
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`:32 layers,width2560,FFN9216,
16 query/4 KV heads of width256,16 linear key/32 value heads of width128,
conv kernel4,epsilon,layer types,RoPE parameters and vocabulary size. Vision
24 layers,width1024,output2560,P16 and merge2 also match. This establishes
architecture relevance, not interchangeable checkpoint weights or OCR prompts.

## Actual C++ architecture comparison

[llama_model_qwen35::load_arch_hparams/load_arch_tensors and graph](https://github.com/ggml-org/llama.cpp/blob/bbc2e4b65705132843b9d142f4471dfb8085b233/src/models/qwen35.cpp)
select recurrent layers from explicit metadata or interval4 and recognize the
32-layer/2560-width4B geometry. `graph::build_layer_attn` splits interleaved per-head
Q/gate, normalizes Q/K, applies MRoPE, then sigmoid-gates attention before output
projection. `graph::build_layer_attn_linear` applies convolution,SiLU,Q/K L2 norm,
recurrent attention and gated RMS normalization. `build_qkvz` and
`build_norm_gated` expose the corresponding projections/gate. A missing output
head falls back to tied embeddings. Local attention/GDN owner and execution
follow these operations; no omitted gate, extra scaling or wrong layer interval
was found by this static comparison.

The existing MiniCPM stored-graph candidate in
[src/qwen35_graph_decoder.cpp](../src/qwen35_graph_decoder.cpp) is separately gated
and its recurrent projection expects equal-head`3*group` QKV. It cannot replace
LightOn's compact16-key/32-value GDN without new owner,shape,state and source
controls. Upstream performance is not evidence that this alternate local graph
already implements the LightOn model.

## GDN reorder is a storage contract, not a missing native permutation

Upstream[conversion/qwen.py](https://github.com/ggml-org/llama.cpp/blob/bbc2e4b65705132843b9d142f4471dfb8085b233/conversion/qwen.py)
`_LinearAttentionVReorderBase._reorder_v_heads/modify_tensors` changes grouped HF
value-head order into tiled order for ggml broadcast. It permutes V rows of QKV,
Z rows,a/b rows,A/dt vectors,V convolution channels and output-projection columns
together. For16 key heads and2 values/key, new-to-old head order is
`0,2,...,30,1,3,...,31`. The integer audit verifies all4096 lanes are bijective
and tiled key pairing equals the permuted original grouped pairing.

Local `linear_grouped_core_cpu` instead repeats each complete Q/K head beside
its two original grouped values, retaining HF-order parameters. These are
consistent alternative layouts. Adding upstream V permutation alone would
break parameter/state alignment. Inherited `Qwen3NextModel.modify_tensors`
also stores`-exp(A_log)` and adds1 to ordinary RMS weights, while excluding
linear-attention gated norm. Local code retains logs/RMS offsets and performs
those operations at runtime. Importing upstream-format tensors would therefore
need explicit conversion/admission, not just name aliases. The index proof is
algebraic only; it does not establish real checkpoint/permuted quantized-byte or
floating-summation parity.

## Q4_K recipes and actionable next steps

[llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/bbc2e4b65705132843b9d142f4471dfb8085b233/src/llama-quant.cpp)
`llama_tensor_get_type_impl` uses its `use_more_bits` predicate to protect selected attention-V and
FFN-down tensors withQ6_K forQ4_K_M. `llama_tensor_quantize_impl` passes importance
vectors into`ggml_quantize_chunk`; convolution weights are excluded from ordinary
quantization. These are concrete policy choices, not proof of LightOn quality.
Our frozen plainQ4_K and mixersQ8/FFNQ4 recipes are not Q4_K_M. The latter keeps
all FFN-down matrices atQ4_K, so a blanket mixer upgrade does not test the
upstream FFN-down strategy.

1. Finish same-weight arithmetic attribution before changing quantization. The
   [step33 evidence](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/review.json)
   shows final-head BF16 rounding restores all178 genuine mixed-oracle tokens
   and EOS on the form, while the uncorrected native path diverges at33. That
   mixed oracle itself differs from official unquantized output at25. This
   separates a runtime correction from remaining weight-precision error.
2. After current precision controls, predeclare a new calibration-only FFN-down
   protection or activation-weightedQ4 policy. Existing quantizer supports
   `--tensor-type` and `--imatrix`; `project_cpu` already has a named input
   observer. Verify actual collector coverage/names/counts before claiming an
   imatrix is applied. Preserve the frozen recipe and heldout separation. A
   Q6_K policy additionally needs explicit current-owner/type admission; Q8
   selective-down is another unqualified candidate, not an interchangeable alias.
3. Treat direct stored-type matmul as a separate arithmetic experiment. Current
   `project_cpu` expands weights toF32 before ggml matmul; the pinned ggml CPU
   Q4_K dot trait usesQ8_K activations. That changes operand arithmetic. Require
   same-input stage controls and decoded/EOS comparison before any speed claim.

The [four-case vision result](lighton4b-vision-generalization.md) already finds
136/136 processor/vision hashes exact. No evidence here justifies another vision
rewrite or copying upstream execution wholesale. Larger geometry,other
checkpoints/backends and architecture-wide equivalence remain untested.
