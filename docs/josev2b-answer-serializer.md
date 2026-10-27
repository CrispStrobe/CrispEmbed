# Josev internal answer serializer

This source-prepared lane implements internal SystemOne answer JSON. The
`CRISPEMBED_JOSEV_ANSWER_SERIALIZER` context gate defaults off (unset/empty/0).
It adds no C API, binding, registry, model loader or change to the frozen
[frontend](josev2b-tokenizer-schema.md). The new source closure is explicitly
versioned from frontend `f424ec52`; historical frontend guards are unchanged
and intentionally reject this broader source tree.

The formatting blueprint is the exact Apache-2.0 publisher
[`joint_schema_model.py` at6f0a31c8](https://huggingface.co/mayflowergmbh/Josev-2B/blob/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/joint_schema_model.py),
SHA256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
Its license/notices are [retained](josev2b-tokenizer-schema-LICENSE.txt).
The implementation independently expresses those semantics, with actual gold
produced by the unchanged AST-extracted `systemone_answer`, without importing
Torch or modifying that function.

`answer_serializer::serialize(request_json, encoded_text, probabilities, output,
error)` accepts ordered native F32 probability vectors matching encoded field
and option order. It returns typed JSON `model`, ordered `answers`, and
`usage` with actual encoded input count and zero output tokens. Existing contexts
keep their construction-time gate value. Failure clears output. It validates
required model/state/questions, known text schema, field/type/option identities,
logical spans, lexical IDs, all-ones mask,16 fields/128 total options/512 tokens,
and finite probabilities in [0,1]. Probabilities are not renormalized, and no
additional calibration/temperature is applied. Text-only unsupported inputs
reject explicitly. A model string is echoed exactly; this helper resolves no
alias or model registry entry.

Noul uses true at index0. Choice vectors follow encoded sorted option order,
while argmax ties, probability JSON and output follow original criteria insertion
order. Score uses ordinal index expectation and original list-valued legend;
list values are not reinterpreted as numeric levels. Nested legend JSON retains
request insertion order and numeric types. Question instructions and state do
not enter answer math. No Gemma serializer, enum or state behavior is used.

CPython3.13.15's
[float sum](https://github.com/python/cpython/blob/v3.13.15/Python/bltinmodule.c)
uses ordered Neumaier compensation. The native expectation independently follows
that algorithm, with F32 probabilities promoted to binary64 and exact index
products. It does not use a plain fold or a different summation order. The
standalone build disables floating contraction and does not enable fast math.
Non-nearest runtime rounding modes reject and are covered by recovery controls.
Four-decimal rounding decodes the actual finite binary64 mantissa/exponent,
multiplies the integer mantissa by625, and compares exact remainder to midpoint
with ties to even. It avoids rounding `x*10000` in binary64. Decimal emission
uses integer digits, preserves negative zero and emits `.0` for integer-valued
floats. The bounded helper supports [0,8128], covering even128 unnormalized
probabilities of1; no sum-to-one check silently changes the publisher's math.
The response JSON is compared both semantically and as exact compact UTF8 JSON,
including object order, numeric rendering and signed zero.

The metadata phase `josev-answer-serializer` calls a separate15-minute reusable
workflow. Hosted CPython3.13.15 and binary64 fingerprint checks precede the small
23KB source download. Seven actual F32 fields from all four frozen original51
requests must reproduce every official public response exactly. Thirteen owned
cases cover ties/Unicode/original order, negative zero, nested score legends,
default instructions,512-token usage, non-normalized vectors,16 fields and128
options. A64-option positive counterexample makes compensated and plain-fold
four-decimal scores differ. Additional declared binary64 decimal-midpoint,
adjacent-value, subnormal and halfway cases freeze actual Python `round` output.
These outputs are all frozen before any C++ compilation.

Only then is the standalone two-worker component/probe built, without linking
frontend.cpp, ggml, a tokenizer, inference framework or weights. Every case must
match the original JSON. Gate/context controls, typed input rejection and
valid-before/reject/valid-after recovery run against the actual component.
Ordinary and test-enabled answer outputs must be byte-identical. The diagnostic
rounding symbol exists only in a test-enabled archive, with an actual compiled
ordinary-symbol exclusion check. Failed cases and partial outputs are retained;
no original vectors or accepted budgets are changed.

This is formatting and manual-input evidence only. Hosted execution remains
unmeasured at source preparation. It does not establish composed head/backbone,
tokenizer, task-quality, public-interface or release qualification, and does not
promote either new default-off gate. No combined model loader is added here.
