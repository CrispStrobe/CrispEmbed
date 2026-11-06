# LightOn4B vision generalization diagnostic

[Hosted run42166765060](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42166765060)
at source`6838c8106a359b548c854e1865104733cdbd6e33` passes the actual native
RGB frontend and complete vision pipeline on all four predeclared frozen cases.
All34 processor/vision stages per case have identical source/native SHA256:
136/136 pairs, zero differing values, maximum absolute and RMS errors0. No
remaining vision divergence was found on these cases. This is component
correctness evidence; decoder output and EOS were not tested.

| Frozen case | Grid | Native stages exact | Merger values | Merger source max absolute | Native subprocess seconds |
| --- | --- | --- | --- | --- | --- |
| simple_table | 1,12,24 | 34/34 | 184,320 | 7.84375 | 13.4112 |
| arabic_printed_line | 1,8,34 | 34/34 | 174,080 | 4.34375 | 12.1668 |
| receipt_example | 1,28,22 | 34/34 | 394,240 | 7.875 | 47.8349 |
| handwritten_letter | 1,20,28 | 34/34 | 358,400 | 5.0625 | 41.1258 |

Each timed subprocess includes native weight loading and stage file writes;
these are not source/native inference benchmarks. Each case's actual rotary
cosine and sine arrays match source bytes, extending the earlier single-form
composed evidence across these four actual geometries. Existing complete-domain
arithmetic/fallback controls remain separate evidence.

The [source driver](https://github.com/CrispStrobe/CrispEmbed/blob/6838c8106a359b548c854e1865104733cdbd6e33/tools/ci-heavy/lighton4b_vision_generalization.py)
uses the actual pinned Transformers eager`Qwen3_5VisionModel`, original checkpoint
revision`b71010f095dc3735de7ee6b9969bff8f8f50b39a` and all297 original BF16 vision
parameters. It fetches only the667,028,480-byte vision range, SHA256
`b519e0f145bef59b18ddb3486ee35e63ad585675277ad40762399561ce17f521`.
Every loaded BF16 parameter is checked against its source bytes before exact F32
expansion for the ordinary native owner/mapper. No source activations enter
native inference. The full9GB checkpoint and a GGUF payload are not independently
fetched or hashed in this lane.

Before admitting the new diagnostics, the standalone original vision class
reproduces all34 retained original form stage hashes. Source class hash,
Torch2.13.0 CPU git, Transformers5.5.4, NumPy2.3.5 and AVX2/oneDNN/MKL dispatch
are pinned. Rotary inverse frequencies remain F32 during BF16 model construction.
The native execution uses all verified vision gates, including the full-domain
rotary table; rejected rotary AVX2 and decoder BF16 gates stay off. Every new
production gate remains default-off. This test makes no runtime source change.

The four independent source forwards use original frozen prepared input arrays,
not native-generated patches. Native inputs start at each frozen PNG's verified
RGB bytes. Input contracts and original batch/capture hashes are checked; the
original four-case outputs are never replaced. Historical four-case source CPU
dispatch was not recorded, so these new internal source stages are diagnostics
under the qualified AVX2 profile, not a claim about that historical dispatch.
The original receipt remains capped without EOS; the other originals completed.

A test-only observer copy is prepared to capture the earliest differing block,
with a mandatory all-stage ordinary-path byte control. No case diverged, so the
observer did not execute and its control is correctly recorded as null. No
source-input diagnostic retry was needed. No full model, decoder, logits,
generation, frozen gold rewrite or new quality qualification occurred. Existing
Q4/mixed/Q8 decoded limitations remain. The actual still-image wrapper admits
up to1024 patches; these cases do not qualify the wider processor input domain.

The [full small report](../tests/results/qwen35-ocr/lightonocr3-4b-vision-fourcase-generalization.json)
retains all stage hashes, absolute magnitudes/errors, inputs and source controls.
The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-vision-fourcase-generalization-evidence.json)
binds its hash and remote artifact13480215848 (181,680,479 bytes,90-day retention,
SHA256`23e92113dfda8f95a8e623b4d98ecca567821e38e073b9aec7bcf6e32c692e2d`).
Large source/native arrays and parameter hashes remain remote.

Initial[42166067458](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42166067458)
failed at the transport SDK's undeclared`six` import before any source download,
build or forward. Its unchanged failure artifact/hash is preserved in the
receipt. The successful retry pinsKaggle2.2.2 andsix1.17.0.
