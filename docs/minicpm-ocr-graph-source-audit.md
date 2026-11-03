# MiniCPM stored-graph Python source audit

The bounded source review found no concrete algorithm or layout blocker in the
Mini-only graph decoder at
[`83eac856cc684808e417ebfed7db7c2f677dc09f`](https://github.com/CrispStrobe/CrispEmbed/tree/83eac856cc684808e417ebfed7db7c2f677dc09f).
This is **source evidence only**. It contains no numerical, projected-stage,
decoded-panel or performance result. Graph and quantized model acceptance gates
remain pending and defaults remain off.

The [machine-readable audit](../tests/results/minicpm-ocr/graph-python-source-audit.json)
records native commit/blob/SHA identities, Python entry names and line anchors,
the retrieval provenance and all limits. Native files are byte-identical at the
documentation base `b237132ee7fc3b318db04be306880fb5e2224298`. This audit changes
no runtime, tests, build, source contract or original reference.

## Authenticated Python blueprint

The authority is the official
[Transformers 5.21.0 publisher metadata](https://pypi.org/pypi/transformers/5.21.0/json)
and its exact
[publisher wheel](https://files.pythonhosted.org/packages/73/fa/fb638d54bffff44700c8d487608c1f2872823e606fbd043b3c29c93236cf/transformers-5.21.0-py3-none-any.whl).
The GitHub raw `v5.21.0` tag URL returned 404, so another version was not
substituted. Strict HTTP 206 ranges retrieved the ZIP footer, central directory
and only the two compressed source entries. Every range had its exact declared
extent; each extracted text entry passed both CRC32 and the already declared
individual SHA256.

| Wheel entry | Bytes | Individual SHA256 | CRC32 |
| --- | ---: | --- | ---: |
| `transformers/models/qwen3_5/modeling_qwen3_5.py` (Q35) | 87,748 | `af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c` | 1217625313 |
| `transformers/models/minicpm_ocr/modeling_minicpm_ocr.py` (Mini) | 37,602 | `45d60914094cdc51fab67dc8edcf85a36a4db61a770eaca891cc820675fcf352` | 2975781327 |

Publisher metadata was 30,963 bytes; ZIP/source range transfers totaled 362,329
bytes. The complete 12,978,463-byte wheel was neither downloaded nor saved,
installed or imported. Its publisher SHA is recorded as a declaration, not an
independently verified whole-wheel payload hash. The public receipt preserves
all range offsets, lengths and hashes so the exact source extraction can be
recovered. No upstream Python code is committed by this audit.

## Checked boundaries

Python line numbers below refer to the authenticated wheel entries above, not
an inferred GitHub tag. Exact native links and individual checks are in the
JSON audit.

| Boundary | Python anchors | Native source conclusion |
| --- | --- | --- |
| Profile and tied head | Mini 779–787 | Narrow batch-one, 1024-hidden, 248100-vocabulary, 24-layer profile; the graph requires the tied head and retains other engines' scalar routes. |
| Gated full attention | Q35 736–810 | Per-head Q/gate split, K/V head grouping, head norms, causal KV append, output sigmoid and projection layouts match the scalar path. |
| Positions and RoPE | Q35 155–201, 1247–1262 | Equal axes and past-token offsets give one-dimensional partial RoPE: 64 of 256 dimensions, theta 1e7. |
| Linear mixer | Q35 232–276, 490–651 | QKV/z/a/b ordering, causal convolution history, decay and beta correspond to the original interfaces. |
| Recurrent state | Q35 280–285, 424–485 | Sum-based Q/K normalization, key/value transpose at the GGML boundary and one query-dimension scale agree with the retained scalar formulation. |
| Norms and residuals | Q35 204–220, 812–901 | Transformer norms are zero-centered; gated linear RMS uses direct weights. Attention residual, post norm, SwiGLU and final residual keep their order. |
| Vocabulary boundary | Mini 779–787, 833–838 | The native last-token projection matches the original capture's explicitly retained last-token head slice. No original tensor is reshaped to fit. |
| Storage and ownership | Native/GGML sources in the receipt | Model-owned contiguous host tensors retain their stored types; per-call graph buffers are copied before teardown. Request KV/conv/recurrent state stays caller-owned, with pending outputs published only after success. |

The [pinned GGML fork](https://github.com/CrispStrobe/ggml/tree/a06fc0bb46b73cc10011fc90ae4054aee8438aeb)
uses an F16-weight/F32-activation dot trait. Q8_0 and Q4_K use Q8_0 and Q8_K
activation-dot paths respectively. These stored-type candidate paths are not
claimed numerically equivalent to dequantized F32 controls. The test observer
remains conditional on `CRISPEMBED_MINICPM_GRAPH_AUDIT`; this source review did
not compile or inspect resulting binaries.

## Limits and pending proof

Python prefill uses a chunk-64 triangular delta algorithm; native retains the
scalar recurrent kernel. Floating reduction, scaling placement and rounding
order can differ. This is an existing numerical boundary, not a newly identified
graph semantic bug or an established cause. Formula-level source agreement
does not establish byte or numerical parity and grants no tolerance waiver.

The existing [graph contract](minicpm-ocr-graph-v1.json) still requires all five
frozen cases, all 26 full prefill stages per case, 187 same-input projection
calls with their explicitly bounded first/last-token scope, actual Ref and
independent metrics at 0.002 maximum absolute/0.99999 minimum row cosine,
ordinary/observer/repeat neutrality and exact generated IDs, text, EOS and caps.
F32/F16 scalar/graph and Q8/protected-Q4 model outcomes must come from actual
hosted execution. No original recapture, default promotion, publication or
release qualification follows from this audit.
