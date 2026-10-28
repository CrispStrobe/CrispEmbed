# Josev role-based mixed-precision development panel

This separately versioned experiment retains the original four development
requests, original251 arrays and fixed0.002 maximum absolute/0.99999 minimum row
cosine budgets. F32 must pass all227 mapped stages, original raw final/logit and
probability budgets, ordinal0.001 and complete typed answers before any candidate.
No original forwards are repeated. No new twenty-case holdout is read or run,
no recipe is selected automatically, and this panel cannot qualify publication.

The full150-Q4_K candidate remains negative. Its exactv4 source is
`9c475d0dac70264adc123a63b6c1827b4b7280d5`; reviewed panel SHA256
`c544d9c30b7fc9ede8c1396db2479a6b25bfa2c9dec227195f13ef33c9f4b164`.
All four earliest strict failures are `backbone/layer/0`, maximum errors
0.11353–0.16508 and minimum row cosines0.98059–0.98259. Final norm maximum
error reaches30.49212; two ordinal errors0.04725674/0.01106670 exceed0.001,
although the discrete argmax choices remain unchanged. The strict F32 control
passes227 stages (worst0.0009555817). None of these negatives is replaced.

| Recipe | Attention matrices | FFN down | FFN gate/up | F32/F16/Q4_K counts | Predicted bytes |
| --- | --- | --- | --- | --- | --- |
| A | All78 F16 | All24 Q4_K | All48 Q4_K | 292/78/72 | 3,613,774,304 |
| B | All78 F16 | All24 F32 | All48 Q4_K | 316/78/48 | 4,651,864,544 |
| C | All78 F16 | All24 F32 | First20 layers F16; last4 layers8 Q4_K | 316/118/8 | 5,375,381,984 |

All292 critical tensors (vocabulary, norms, head, a/b, convolution and other
nonselected tensors) retain exact original F32 bytes. B/C protect down matrices
monotonically; C never reintroduces Q4 down weights. The actual existing C
quantizer writes candidates. A bounded parser verifies all442 semantic shapes,
storage extents/types/counts and payload hashes. An independent integer IEEE754
F32-to-binary16 implementation verifies every F16 value with nearest/ties-even,
signed-zero/subnormal preservation and nonfinite/overflow rejection. No Q4_0
fallback or unreported storage substitution is accepted. Header metadata and
embedded tokenizer are preserved; the existing same-length mixed packed-format
marker is sealed only after verification. The recipe is bound by the external
immutable protocol, conversion receipt and whole-model SHA; the marker alone
does not identify which recipe was used.

The pinned runtime materializes F16/Q4_K matrices to F32 in
`qwen35_ocr::project_cpu` before F32 multiplication. There is no Q8_K activation
quantization in this route. Layer0 failure alone cannot distinguish attention
weight loss from FFN weight loss; A is a structural hypothesis, not a causal
claim. A future same-input operator probe must distinguish dequantization bits
and intended stored-weight loss using actual F32 activations. It must not claim
an unobserved Q8 activation path. No new opaque operator implementation is added
here; the existing actual Ref and independent NumPy full-stage checks remain.

The standalone [workflow](../.github/workflows/josev-mixed-recipes-v1.yml) uses
Linux CPU, two threads, CPython3.13 and the reviewed hash-locked NumPy2.2.6 wheel,
requires14GiB available RAM and64GiB disk before payload access, authenticates
the retained original archive through exact GitHub run/source/artifact/member
identities, and preserves models/arrays/adapters remotely for90 days. The source
closure binds the entire unchanged9c runtime, a06 backend, quantizer, old worker,
comparison code, build inputs and frozen references. Both quantization gates
remain required; unset/empty/literal0 rejection and failure/recovery checks run for
every recipe. Plain/observed/repeat raw outputs must be byte-identical. Numeric
rejections are retained and do not prevent later recipes from being evaluated;
structural/input/observer errors stop the panel. All strict original budgets
remain separately reported. NEW20 native evaluation needs a later explicit
reviewed recipe/model/source binding; it cannot tune these recipes or labels.

Source preparation is not execution. No workflow is dispatched by this change;
there is no default, release, app, public alias or model-publication promotion.
