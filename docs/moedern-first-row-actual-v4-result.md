# MoEdern actual document router capture: completed numerical rejection

Exact-version Kaggle CPUv4 at source
`f3364a57e29db6ecd58fa1fe2cb60a794d1f14b1` completes the declared one-document
capture. [Independent review](../tests/results/moedern-colbert/first-row-actual-v4-review.json)
verifies submitted-source Get/Status/every List page and all75 selected small
files,1,524,361 bytes. The400 outputs, including actual tensors, remain remote.
No large arrays or weights were downloaded locally. This is a completed
numerical rejection, not an infrastructure failure or a new reference.

Both ordinary barriers pass first: original strict two-anchor Ref and all93
native byte anchors. Original observation/repeat/unhooked/public slicing preserve
raw feature bits. The actual router uses `aten.softmax.int`, followed by real
`aten.topk.default` and `aten.div.Tensor`; all eight operations/two SDPA calls,
arguments/returns and52 FFN dispatch metadata records repeat exactly. Five
original targeted calls and14 native computations complete within the declared
budget; aggregate original calls across initial attempts are8. No full original
corpus is recaptured.

Actual Ref and independent metrics agree for all44 floating stages:39 pass and
five fail. First bit divergence is embedding projection, max8.9406967e-8 versus
original magnitude0.4492559. Both-layer router logits/probabilities remain within
unchanged floating budgets; that does not establish identical routing.

The first strict disagreement is layer1 top-k, one row only: document token
position58, token29084. Original chooses experts `[2,4]`; native chooses `[2,6]`.

| Router probability at row58 | Original | Native |
|---|---:|---:|
| Expert2 |0.522528350353241|0.5225281715393066|
| Expert4 |0.1681605577468872|0.1681603938341141|
| Expert6 |0.1681604236364365|0.1681604981422424|

These are unequal near-boundary values, not an exact tie. Original expert4
exceeds6 by1.3411045e-7; native expert6 exceeds4 by1.0430813e-7. Renormalized
selected weights happen to be identical, so their cosine/bytes conceal the
changed expert identity. Layer1 FFN output then fails max0.57834816,
RMS0.01056687, cosine0.99660470 versus original magnitude141.67180. Final
normalized output fails max0.02282943/cosine0.99574771. Six native-only
observations, including derived selection margins, remain explicitly without
original gold.

This is an F32 path with protected router tensors. Neither more router storage
precision nor a different exact-tie policy is established as a correction.
Next, reuse the two actual normalized router inputs and captured logits/
probabilities/IDs from this capture in a separately declared same-input,
same-weight router linear/softmax/top-k replay. Preserve full307-row shapes,
real original/native operators, ordinary source/binary controls and per-stage
magnitudes. Compare original-input/native-input cross arms and own-byte replay
before changing upstream arithmetic or routing. No additional full model or
original forward is needed for that component attribution. Keep historical
six matrices, all old gold and every budget unchanged.
