# Actual layer3 image attention stages

The [actual sanitized report](../../tests/results/qwen35-ocr/lighton4b-attention252-stages.json)
records the first full-attention layer on the retained two-column input. Eleven
original tensor payloads (67,237,888 bytes), all 266 layer2 inputs, embeddings and
positions are authenticated. The native component replays the actual tile plan
and reproduces every retained layer3 output byte. The source executes only
prefill and one sample in each observer arm: firstscore, vision, quantized
dispatch and every old captured hidden boundary remain exact. The original
654-token source EOS and 1536-token native cap remain the decoded evidence.

The preceding four-token K/V context matches exactly. On the 252 image rows,
input normalization, Q/K/V projections, normalized Q/K and the attention gate are
bit-identical. The first recorded difference is post-RoPE Q: RMS
7.2642718e-8, maximum absolute error 3.2484531e-6, versus source/native RMS
magnitudes 1.2929293173/1.2929293174. K differs by RMS 9.5905172e-8 and maximum
3.5762787e-6. Native RoPE here is **reconstructed** from the actual normalized
vectors with production scalar arithmetic; it is not directly captured inside
`attention_cpu`. Full-block byte closure is independently required.

The actual source F16 K cache after the image differs from native by maximum
0.001953125; V remains exact. Gated attention differs by RMS 3.3464476e-7 and
maximum 0.00011909008. Propagation through the output projection and FFN reproduces
the frozen layer3 hidden error: RMS 0.000066194246 and maximum 0.0042481422.
These observations locate a runtime arithmetic boundary after equal weights,
inputs and normalization. They do not prove that changing RoPE alone restores
all scores, cache behavior or decoded quality.

The pinned source uses iterative IMROPE frequencies; the native scalar function
recomputes a power per lane. A bounded operator-only comparison on the retained
normalized vectors is next, together with direct test-only native post-rotation
capture and full-block neutrality. Actual source QK/PV operand names, types,
dimensions and strides are preserved for later query-count/dispatch isolation.
Small N4/N16 attention proofs do not establish the 252-query geometry.

The reference patch changes two raw K/V callback labels to avoid collisions and
adds read-only capture; five model-free controls cover that edit and artifact,
layout and scope admission. All probes are excluded from product artifacts.
Production sources, global traits, gates, versions and application pins remain
unchanged. Nothing is qualified or promoted.
