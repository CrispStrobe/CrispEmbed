# LightOn4B post-fused source trace

This test-only reader compares retained arrays without inference. The exact source
and failed native reports are pinned in `audit.py`. Run with `--source`, `--native`
and `--output` on an authorized artifact worker; keep large artifacts remote.
The reader requires NumPy and exports no decoded text or vocabulary IDs.

The [actual sanitized result](../../tests/results/qwen35-ocr/lighton4b-post-fused-source-trace.json)
authenticates all 266 input embeddings and positions, the physical head inversion,
12 first-four layer-zero operators, every available output across 32 layers and
same-prefix scores at positions 0, 84 and 85. The source live firstscore and native
prefill/teacher/greedy firstscore each close byte for byte.

All first-four operators and all four chunks of layers 0–2 match the source
exactly. The earliest remaining recorded difference is layer 3, the first full
attention layer, on the 252 image rows: error RMS 0.00006619425 and maximum absolute
error 0.004248142, against source/native RMS magnitudes 0.279053501/0.279053351.
The preceding four text rows at that layer remain exact. This locates a hidden
boundary; it does not isolate an attention operator, weight or cache defect.

At position 85, both paths have consumed the same generated prefix but choose
different vocabulary entries. The source favors its choice by 0.25988388; native
favors its choice by 0.24337959. Their score RMS/maximum absolute errors are
0.14543341/0.90035629. Source finishes at 654 tokens with EOS; native reaches its
1536-token cap without EOS. Fused ON remains unqualified, and the original failed
panel is unchanged.

The next bounded attribution is the actual layer-3 image chunk with identical
inputs, four-token context/cache, positions, frozen quantized weights and gates.
Its source stage names and layouts must come from the real blueprint. Existing
small attention primitive controls do not establish equality at this geometry.
No runtime version, default, release or application pin changes follow.
