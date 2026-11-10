# Fixed mixed BF16 original: exploratory first-position mechanism check

This protocol is declared after observing the Q4 first-token reversal on the
existing form image. It is an exploratory mechanism check, not a held-out
calibration, architecture comparison for quality selection, or replacement gold.
The existing full decoded rejections remain unchanged. No prompt, cap, threshold,
input, tensor selection or policy may change in response to this run.

The [complete immutable name sets and digests](lighton4b-bf16-mixed-first-score-v1.json)
select precisely 249 text matrices:

| Existing public file | Matrices | Policy | Newline-delimited sorted name SHA-256 |
| --- | ---: | --- | --- |
| Pinned Q8 | 153 | Embedding 1, linear attention 120, dense attention 32 | `d0608d47ab8dbc77882f8f6b494f44db1023f5fa822ac2214bc4e5b721ab1383` |
| Pinned Q4 | 96 | All FFN gate/up/down matrices | `d6d1fbbc580cc0db0f8d2379b60f97a079f04436f771509b0b23661641cf70f9` |

All 474 protected tensors remain the original. The shared embedding/output head
is tied and Q8. Policy classification must agree exactly with both frozen name
lists, including their counts and digests; rehashed changes also reject. This
matches the already considered mixed architecture and introduces no new model
upload or conversion publication. The two complete existing public files retain
their previous full-byte hashes and revision pins.

The original authenticated BF16 model is loaded once. One baseline prefill must
reproduce the frozen first-score hash and ID 27 exactly. In explicit Q8 then Q4
file order, authenticate the complete file and both readers' full 723-tensor
inventories; stream only the selected matrices through independent dequantization
and BF16 conversion. Record each raw tensor hash and selection status. Every
installed BF16 chunk and whole parameter is checked; after both files, all 249
final parameter hashes must match their declared source. Protected tensors,
buffers, parameter storage/identity, prepared inputs and tied head are checked.
Only then execute one mixed prefill and capture one complete finite score row,
ID/probabilities and decoded prefix. Exactly two fresh original root/vision/text
forwards are allowed. There is no native inference or F32 model. Explicit
dequant-to-BF16 rounding remains part of this precision protocol.

Use one CPU worker with the existing frozen original package/checkpoint guards
and a 3600-second model-worker bound. Initial effective RAM remains at least
22 GiB and scratch 27 GiB. Reduce dequant chunks to 65,536 F32 elements (256 KiB),
and retain just one candidate file mapping at a time. Flush/advice completed
64 MiB download regions, bound hash-read cache, and advise each completed tensor
region away. Advise only this task's authenticated source checkpoint clean cache
before/after streaming; never alter or unlink it. Before mixed inference require
at least 8 GiB effective available RAM and 2 GiB actual cgroup hard margin;
streaming requires 4 GiB effective headroom. Cache advice is advisory, so guards
may reject honestly. The previous worker's 31.31 GB cgroup peak of 32.21 GB is
retained as a warning, not treated as ample capacity.

Six additional model-free controls cover exact partition, missing/duplicate or
wrong digest, rehashed architecture swaps, order/count/chunk/scope changes,
unchanged old three-arm protocol and exact bounded-cache hashing. Run the six
old controls too. No hosted dispatch is implied by this document alone. Results
cannot qualify EOS, full-document quality, held-out generalization, defaults,
release/app pins or a new mixed upload.
