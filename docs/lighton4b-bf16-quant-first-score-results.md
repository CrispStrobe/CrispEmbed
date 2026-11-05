# Genuine BF16 original with quantized weights: first-position rejection

The single CPU worker at source `048fd683f4ad4adf50f945f6d7c6b436bacdb9aa`
completed all three one-position arms. Both quantized-weight models reject the
unchanged maximum-error `0.002` / cosine `0.99999` score gate. Q4 changes the first
token inside the genuine original BF16 model, independently of native inference.

| Original BF16 model weights | First ID / decoded prefix | Score max error | Score RMS error | Cosine | Probability of original ID 27 |
| --- | --- | ---: | ---: | ---: | ---: |
| Original checkpoint | 27 / `<` | 0 | 0 | 1 | 0.8218573472 |
| Q8 dequantized then BF16 | 27 / `<` | 0.47265625 | 0.0715803722 | 0.9995343589 | 0.8212135906 |
| Q4 dequantized then BF16 | 20353 / `<table` | 2.546875 | 0.5915698369 | 0.9758833550 | 0.2026010417 |

Q4 assigns probability `0.5507267302` to its selected ID 20353. Original maximum
absolute score magnitude is `22.875`; Q8 is `22.875`, Q4 `22`. Each complete row
has 248,320 finite scores. The original row is byte-exact with frozen SHA
`a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd`.
Q8 row SHA is `d620c91a169dc85b77fc81e2e0f615da4516ccd8dce21d0cc152674932d6c400`;
Q4 is `e59d6054c85e81fb1be7101c9899287fa3a1cdd23a3110c8e03d874992387a0d`.

The [frozen protocol and implementation](https://github.com/CrispStrobe/CrispEmbed/blob/048fd683f4ad4adf50f945f6d7c6b436bacdb9aa/docs/lighton4b-bf16-quant-first-score.md)
use original source `8d3b2331d1b57e285f7ba348fc1eacd689b79aa8`, its unchanged
checkpoint/processor/package authentication, and one AST replacement of the
generation expression. No F32 model is constructed. The complete original
9,078,620,536-byte checkpoint and complete immutable Q8/Q4 files pass their pinned
SHA-256 checks. All five prepared input arrays retain their frozen hashes before
every arm. Exactly one fresh root, vision and text forward occurs in each arm;
the original wrapper independently records three model calls.

Both candidates verify the complete 723-tensor map using two readers: 249 changed
quantized text matrices, including the embedding, and 474 protected original
tensors. Every copied BF16 chunk and whole parameter passes exact readback.
Parameter identity, storage, shapes, dtype, unchanged buffers and tied output
head are checked. Q4 overwrites the same entire 249-matrix set after Q8. Its
embedding remains Q8, with the identical effective BF16 embedding hash in both
arms. Maximum dequantization-to-BF16 rounding is `0.00096893310546875` for Q8 and
`0.0018157958984375` for Q4; the embedding maximum is `0.000244140625`.
This additional BF16 parameter rounding is explicit: these models differ from
the earlier diagnostic retaining F32 dequantized weights. These results cannot
be substituted for that precision protocol or used to subtract runtime errors.

The retained legacy native first-score receipts at `fd150b98f` provide a useful
probability comparison, without another native run. Native Q8 assigns ID 27
`0.8261257729`, compared with BF16 quant-weight oracle `0.8212135906` and original
`0.8218573472`. Native Q4 selects ID 20353 with `0.5256591064` and assigns original
ID 27 `0.2457003999`; the BF16 quant-weight oracle gives `0.5507267302` and
`0.2026010417`. Thus the Q4 first-token reversal persists in the original model.
This does not establish that native arithmetic is correct, quantify its separate
contribution, or make the two architectures equivalent.

The effective/top-level EOS remains 248046 and source text EOS 248044. Each arm
is deliberately capped at one token. The original adapter audit therefore says
`stop=max_tokens`, `eos_completed=false`; its frozen generation configuration
describes the original driver, while the substituted call explicitly uses one
token. The original 177-position French figure reference remains unchanged.
Legacy Q8's different 160-ID EOS output, Q4's incomplete 256-ID cap, mixed
calibration and full decoded-quality rejection remain in force. This image is
an existing diagnostic, not a new held-out calibration or architecture selection
set. There is no EOS, full-document, cross-platform or model qualification here.

Each prefill took 106.66–107.70 seconds. Process peak RSS was 16,292,564,992 bytes;
cgroup peak was 31,310,606,336 of 32,212,254,720 bytes, including cached files.
The bounded worker completed without OOM or retry. Its terminal error is the
intentional numerical rejection: the bootstrap records
`completed-with-numerical-rejection`, the oracle `comparison_completed=true`,
and the unchanged original driver returns zero. No additional inference worker,
runtime modification, conversion publication, default, release or app-pin change
follows. Raw score rows and worker output remain with the original CPU service;
the three needed 993,280-byte score rows were hash-verified for this review.

Complete receipts:

- [Oracle, original parameter hashes, full candidate mapping and readbacks](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-first-score-v1.json)
- [Authenticated original driver audit](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-original-audit-v1.json)
- [Resource/bootstrap receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-bootstrap-v1.json)
- [Hash-verified score patterns and native probability comparison](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-score-review-v1.json)
