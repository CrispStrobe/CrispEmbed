# LightOn4B applicability of the newer Qwen3.5 graph runtime

Source review, 12 November 2026, at `c364feb97` (includes PR216). This is a
bounded implementation assessment; no model execution, numerical qualification,
runtime change, default flip or release is implied.

## Concrete compatibility boundaries

| Boundary | New `qwen35_ocr_gpu.cpp` pipeline | LightOn4B requirement |
|---|---|---|
| Linear-attention heads | One shared H; Q/K/V each H×W; init requires QKV and convolution channels `3*H*W` | `qwen35_lighton4b_text.h` maps 16 key heads, 32 value heads, W=128: QKV/convolution width 8192, versus pipeline's expected 12288 |
| Q/K expansion | Equal-head slices passed directly to `ggml_gated_delta_net` | Expand grouped Q/K to the corresponding value heads without changing head order; retain compact convolution history |
| RMS weights | `one_plus` applied to input/post/final and attention Q/K norm weights | Physical owner carries direct gamma (`lighton4b_upstream_gamma`); applying `one_plus` again changes every affected norm |
| Delta decay | `-exp(A_log)` and graph `softplus(a+dt_bias)` | Physical source profile has transformed A values; source-operator candidate retains scalar thresholded `log1p(exp(x))` after measured source-bit comparison |
| Linear output layout | Flatten grouped V heads and project directly | Physical projection columns use tiled heads: grouped head h maps to `16*(h%2)+h/2`; see `lighton4b_upstream_v_to_tiled` |
| Runtime owner | Shared Ovis/small-Qwen vision and decoder owner | LightOn4B owns separate exact tensor/schema, image composition, tokenizer, physical profiles and cache-policy checks |

The current shape rejection is useful: enabling `CRISPEMBED_QWEN35_OCR_GPU`
does not make the pipeline a compatible LightOn4B decoder. Relaxing only its
shape guard would make subsequent Q/K/V views and convolution state incorrect.
An equal-head expansion before convolution would also change the physical state
layout and waste work; expansion belongs after compact convolution and Q/K
normalization, before recurrence.

## Reusable runtime improvements

The newer pipeline demonstrates a useful implementation pattern: upload stored
matrices once to a backend buffer, use standard ggml operators, and retain KV,
convolution and recurrent states on the backend across calls. A LightOn4B-only
implementation could avoid repeated host matrix dequantization and state copies.
These are candidates, not measured LightOn4B speedups. Its quantized matrix
arithmetic must remain separately attributable from the existing same-weight
scalar path and from physical Q4/Q6 tiling policies.

The safest next code lane is a separate, default-off grouped GDN graph component.
Reuse the existing compact 8192-channel input, split Q/K at 2048 each and V at
4096, repeat Q/K heads in reviewed grouped order, and preserve all 32×128×128
recurrent values. Test prefill, two cached steps, full-prefix replay and reset
against the existing original-module protocol in
`tools/ci-heavy/lighton4b_gdn_reference.py`. That protocol authenticates
Transformers 5.5.4 and four equal/grouped cases, including 16:32 heads and batch2.
Compare QKV, convolution, normalized Q/K, decay/beta, recurrent state and output
with magnitudes and absolute errors before integrating any full decoder.

Keep direct-gamma/physical-A/tiled-output policies explicit per owner; do not
infer these policies from quantization type. A subsequent complete decoder must
retain the current cache-policy identity checks. The physical vision path needs
its own same-input attribution and cannot inherit the small-Qwen vision receipt.
Heavy component/model A/B belongs on an authorized Kaggle GPU, with remote raw
vectors and downloaded small reports only; no build or inference ran for this
review.

## Evidence that remains binding

The [physical source-operator candidate](lighton4b-physical-gdn-source-ops.md)
is already decoded-negative: table character edits rise 349→364/607 and word
edits 97→104/106, despite improved first-score RMS. Its combined Q4-repack arm
matches the table but worsens form character edits 175→194/656, above the
upstream189 floor. Preserve both default-off paths and these rejections.
The [earlier full-score baseline](lighton4b-upstream-physical-decoded-results.md)
has form/table/Arabic maximum absolute score errors 0.576516/1.062353/0.516224
against source maxima 22.7338/20.5416/18.1741. Those are historical baseline
measurements, not a measurement of the newer graph pipeline on LightOn4B.

No new guard test was added: the existing shape guard rejects this unsupported
layout, and a source-text assertion would only mirror that implementation.
Meaningful reusable-layout tests should execute the future grouped graph and
compare retained arrays and cache semantics; a shape-only success cannot admit
it for OCR. Final acceptance additionally requires original/native decoded IDs,
raw/processed text, EOS/caps and document edit floors on the frozen cases.
