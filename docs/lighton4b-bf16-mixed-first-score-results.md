# Fixed mixed BF16 original: completed exploratory rejection

The predeclared attention/embedding Q8 + FFN Q4 policy at source
`50307be7fdb5ec17f367b9b632f506f3d2307126` completed one bounded CPU worker.
The mixed original BF16 model selects first ID 27, decoded prefix `<`, with
probability `0.769899539995431`. Its complete 248,320-score row rejects the
unchanged `0.002` maximum-error / `0.99999` cosine gate: maximum `1.578125`, RMS
`0.2857999084419158`, cosine `0.9925162468579295`. Original maximum absolute score
is `22.875`, mixed `22.5`. ID 20353 (`<table`) is second with probability
`0.11806791842992502`.

This policy was fixed after observing the all-Q4 first-token reversal on this
same form. It is explicitly an exploratory mechanism check, not held-out
calibration, architecture-quality selection, or new gold. The result restores
the original first token on this diagnostic, while its full score row still
fails. It cannot establish native behavior: no native inference ran. Exactly
one token is generated in each of the original and mixed arms, so no native or
original-model EOS/full-document decoding claim follows.

The [pre-dispatch protocol](https://github.com/CrispStrobe/CrispEmbed/blob/50307be7fdb5ec17f367b9b632f506f3d2307126/docs/lighton4b-bf16-mixed-first-score.md)
freezes every name and digest: Q8 embedding 1 + linear attention 120 + dense
attention 32; Q4 FFN 96. The Q8 sorted-name SHA is
`d0608d47ab8dbc77882f8f6b494f44db1023f5fa822ac2214bc4e5b721ab1383`,
Q4 `d6d1fbbc580cc0db0f8d2379b60f97a079f04436f771509b0b23661641cf70f9`.
Both complete existing public files pass their unchanged full SHA-256 pins,
and both independent readers verify all 723 names/shapes/types. No conversion
or model is uploaded. The original authenticated BF16 checkpoint is resident
once; file order is explicitly Q8 then Q4, and only selected matrices are copied.

All 249 final BF16 parameter hashes agree exactly with the corresponding
previous all-Q8/all-Q4 oracle parameter hashes. All 723 original BF16 parameter
hashes also agree across workers. The 474 protected originals, buffers, parameter
identity/storage/dtype and tied embedding/output head remain unchanged. Each
arm has exactly one fresh root/vision/text forward. All five prepared inputs
retain their original hashes. Baseline score SHA remains byte-exact
`a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd`;
mixed score SHA is
`bea1064566634615c955582808c75b0003dc8f24983cbd12cbb661a1b5c33ae9`.
Both needed 993,280-byte rows were hash-verified for review; large source payloads
remain remote. There is no F32 model: bounded F32 dequantization chunks are
converted to BF16 before parameter copy. Maximum rounding in the selected Q8
matrices is `0.00096893310546875`, selected Q4 FFN `0.000972747802734375`.

Compared with the previous genuine BF16 all-Q8 row, only the 96 FFN matrices
change; mixed versus all-Q4 changes only the 152 attention matrices, because
both already use Q8 embedding. Pairwise full-row max/RMS is `1.75` /
`0.3133927028` versus all-Q8 and `1.8125` / `0.4314533211` versus all-Q4.
This identifies first-position sensitivity to the fixed bit allocation on this
image. It does not select an architecture for quality, isolate individual
matrices, or remove the [legacy native/runtime residual](lighton4b-bf16-quant-first-score-results.md).

Six old and six new model-free contract controls pass. Dequant chunks are capped
at 65,536 F32 elements, clean download/hash/tensor cache is advised away, and only
the task-owned checkpoint clean cache is advised away; its bytes are unchanged.
After both files, measured effective headroom is 17,824,071,680 bytes and actual
cgroup hard margin 11,411,169,280 bytes, passing the predeclared 8 GiB / 2 GiB
guards. Process HWM is 13,867,794,432 bytes; cgroup peak 29,645,406,208 of
32,212,254,720 bytes, compared with previous three-arm peak 31,310,606,336.
This is a different bounded workload, not a memory-performance benchmark.
Original/mixed prefills take 90.55 / 93.85 seconds. No OOM or retry occurs.

The service's terminal error is intentional numerical rejection after completed
capture: original driver returns zero, bootstrap says
`completed-with-numerical-rejection`, and `comparison_completed=true`. Effective
EOS remains 248046, source text/pad EOS 248044. Both one-token audits stop at the
cap; the frozen 177-position French figure reference and all earlier full-native
decoded negatives remain unchanged. No further worker, native fix, default,
release/app pin, alias or model publication follows this result.

Complete receipts:

- [Full mixed oracle, policy, all tensor records and final parameter hashes](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-first-score-v1.json)
- [Authenticated original adapter audit](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-original-audit-v1.json)
- [Bootstrap and resource receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-bootstrap-v1.json)
- [Hash-verified score patterns and pairwise scope](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-score-review-v1.json)
