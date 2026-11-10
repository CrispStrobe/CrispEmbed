# Guarded native-HF Q4KM-style and no-ship controls

Kaggle CPU controls pass at source
`6d751828e61937fe816a04ab355e7305239d63ff`. This establishes guarded component
capability, not model quality. No checkpoint was downloaded, no weights were
quantized and no model inference ran in these controls.

| Actual control | Result |
|---|---:|
| New 33-Q6-head schedule, every matrix type mutation, typed metadata and gates | 1,147 checks pass |
| Existing 32-Q6/Q8-head recipe and operators | 713 checks pass |
| Existing 24-Q6-output recipe | 700 checks pass |
| Existing 24-Q8-output recipe | 112 checks pass |
| Independent Q6 gather/tied streamed head, odd tiles, boundaries, BF16 mode, unchanged storage | 84 checks pass; 65,792 scalar values |
| Valid metadata/direct initializer rejection and unchanged supported formats | 39 checks pass |
| Actual C++ normalized importance vectors | all 248 hashes exact |
| Closed CLI wrong source/method, overrides, legacy header and missing statistics | all five reject without publishing |
| Historical source-pin chain | 15 receipt pins verified against Git |

The public OCR dispatcher rejects raw upstream `qwen35` and experimental 4B
formats 5/6 before choosing a backend or loading weight payload. Static checks
retain the existing format 1–4 boundary, explicit release binary lists and no
4B alias/payload in the registry. The fixture validates its metadata round trip
before rejection tests; source placement in the dispatcher is also checked.
The complete public shared-library call is not linked in this component suite.
Current unqualified Q4 remains a research artifact and is not shipping-approved.

[All raw runs and SHA256 hashes](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-controls/)
are retained. Version 1 stops compiling a new test because its clean-exit namespace
was wrong. Version 2 passes the new 1,146 and old recipe controls but its release
unit lacks STB image linkage. Version 3 corrects those fixture issues, adds the
pure type descriptor check and early public boundary, and passes all controls.
Neither negative was waived or rewritten.

The reviewed sources were integrated into existing PR155 at
`69abebb7e8f0c9235067f76fd5769f8a46f75fb7`, after merging main
`b6d8cf599`. The workflow target-list conflict preserves both PP-OCRv6 policy and
LightOn vision-owner tests; environment documentation was regenerated. All 21
selected source/helper/test files match the hosted passing source byte-for-byte;
CMake, plan and environment changes preserve the unrelated main additions.
The later producer repeats the full model-free union on its exact post-merge
source before staging a model. No GitHub CI was dispatched.

A first producer preflight at
`11131b7fc9eac80d5d82db70070d74725f62dcac` also included an allocation-reuse
prototype. Its OFF/ON stream fixture stops after the legacy pass, before any
checkpoint or quantization. Subsequent review identified observer-counter state
carried between fixture runs; that does not establish an arithmetic mismatch or
qualify the prototype. Its raw 31-file negative is preserved. The performance
commits were removed from the quality producer as instructed; source
`a4f2c908ffaf3f17eb770973017e3de8a08bb121` restores `qwen35_linear.cpp` exactly to
PR155's verified original arithmetic. Performance work remains separate.

The new recipe is deliberately called **native-HF Q4_K_M-style**, not a
publisher-byte-equivalent model: its 216 Q4_K/33 Q6_K type schedule matches the
actual publisher header, while HF grouped-V quant-block coordinates differ.
All 248 body importance vectors are mandatory, including 32 Q6 projections;
the shared Q6 head has no source vector and is intentionally unweighted.
Original-F16 one-pass conversion and complete decoded/EOS-or-cap evaluation are
still required. Two observed calibration cases do not establish broad quality,
heldout success, a default, or any release approval.
