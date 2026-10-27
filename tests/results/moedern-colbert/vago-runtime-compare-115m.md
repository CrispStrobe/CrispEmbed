<!-- Kaggle crispembed-moedern-115m-vago-compare v3 at 34a79ed6 (with fast path); diagnostic only -->
# MoEdern 115M: CrispEmbed vs author runtime (diagnostic)

Source `34a79ed63fa62154b1d63c94dc2026b77eac5dbf`, ggml `3612c0af93f2`, workload sha256 `a31c3663e995bf0c`, 3 timed repetitions (median), warm-up excluded.

Batching: the author wrapper batches (documents 16, queries 64 per engine request by default; also measured with batch_size=1). CrispEmbed has no batch API: one encode call per text. CrispEmbed's default path accepts 1-2 threads only, so its 4-thread rows are expected to fail; ours-fast rows (CRISPEMBED_MOEDERN_FAST=1) accept up to 64 threads under the unchanged parity gates.

## Outputs vs frozen original

| arm | IDs equal | selected max abs | min cosine | row-count mismatches | MaxSim max abs diff |
|---|---|---|---|---|---|
| author-F32-native | 9/10 | 0.3603665456175804 | 0.14486976612838712 | 0 | 3.2768 |
| author-F32-portable | 9/10 | 0.3603665456175804 | 0.14486976612838712 | 0 | 3.2768 |
| author-F16-native | 9/10 | 0.3603753298521042 | 0.1448433401734241 | 0 | 3.27616 |
| author-Q8_0-native | 9/10 | 0.3610490560531616 | 0.14651936414308459 | 0 | 3.15584 |
| author-Q4_0-native | 9/10 | 0.3626599907875061 | 0.14896961277205487 | 0 | 3.50306 |
| ours-portable | 10/10 | 2.8312206268310547e-07 | 0.9999999999995037 | 0 | 1.90735e-06 |
| ours-native | 10/10 | 2.8312206268310547e-07 | 0.9999999999995037 | 0 | 1.90735e-06 |
| ours-fast-portable | 10/10 | 4.023313522338867e-07 | 0.9999999999991136 | 0 | 5.72205e-06 |
| ours-fast-native | 10/10 | 4.023313522338867e-07 | 0.9999999999991136 | 0 | 5.72205e-06 |

## CrispEmbed fast vs default path (same build, selected vectors)

| build | max abs | min cosine |
|---|---|---|
| portable | 6.854534149169922e-07 | 0.9999999999980664 |
| native | 6.854534149169922e-07 | 0.9999999999980664 |

## Author vs CrispEmbed (portable F32)

| arm | IDs equal | selected max abs | min cosine | row-count mismatches |
|---|---|---|---|---|
| author-F32-native_vs_ours | 9 | 0.3603663966059685 | 0.14486979785878276 | 0 |
| author-F32-portable_vs_ours | 9 | 0.3603663966059685 | 0.14486979785878276 | 0 |
| author-F16-native_vs_ours | 9 | 0.36037518084049225 | 0.14484337187348417 | 0 |
| author-Q8_0-native_vs_ours | 9 | 0.3610489070415497 | 0.14651939626788366 | 0 |
| author-Q4_0-native_vs_ours | 9 | 0.36265984177589417 | 0.14896964528721707 | 0 |

## Speed

| runtime | build | model | threads | batching | docs/s | queries/s | tokens/s | doc s (median) | query s (median) | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| author | native | F32 | 2 | wrapper default (docs 16, queries 64) | 2.7 | 15.7 | 494 | 95.909 | 16.309 | 730 |
| author | native | F32 | 2 | batch_size=1 | 2.5 | 13.2 | 458 | 101.770 | 19.429 | 588 |
| author | portable | F32 | 2 | wrapper default (docs 16, queries 64) | 2.7 | 15.7 | 495 | 95.645 | 16.282 | 731 |
| author | portable | F32 | 2 | batch_size=1 | 2.6 | 13.4 | 468 | 99.354 | 19.139 | 589 |
| ours-python | portable | F32 | 2 | one call per text (no batching) | 1.0 | 7.5 | 189 | 259.090 | 34.114 | 925 |
| ours-c | portable | F32 | 2 | one call per text (no batching) | 1.0 | 7.6 | 190 | 257.441 | 33.662 | 897 |
| ours-python | native | F32 | 2 | one call per text (no batching) | 0.9 | 6.9 | 177 | 275.995 | 37.191 | 923 |
| ours-c | native | F32 | 2 | one call per text (no batching) | 0.9 | 6.9 | 178 | 273.788 | 36.977 | 897 |
| ours-fast-python | portable | F32 | 2 | one call per text (no batching) | 3.2 | 18.0 | 588 | 80.119 | 14.217 | 923 |
| ours-fast-c | portable | F32 | 2 | one call per text (no batching) | 3.3 | 18.0 | 600 | 78.233 | 14.238 | 897 |
| ours-fast-python | native | F32 | 2 | one call per text (no batching) | 3.2 | 17.8 | 592 | 79.330 | 14.350 | 924 |
| ours-fast-c | native | F32 | 2 | one call per text (no batching) | 3.3 | 18.2 | 601 | 78.168 | 14.073 | 896 |
| author | native | F32 | 4 | wrapper default (docs 16, queries 64) | 3.1 | 18.3 | 571 | 83.215 | 13.975 | 731 |
| author | native | F32 | 4 | batch_size=1 | 3.0 | 15.4 | 539 | 86.319 | 16.601 | 589 |
| author | portable | F32 | 4 | wrapper default (docs 16, queries 64) | 3.1 | 18.2 | 571 | 83.108 | 14.080 | 730 |
| author | portable | F32 | 4 | batch_size=1 | 3.0 | 15.3 | 537 | 86.627 | 16.750 | 588 |
| ours-python | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-16-ours-python-portable-F32-t4.log) |  |  |  |  |  |
| ours-c | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-17-ours-c-portable-F32-t4.log) |  |  |  |  |  |
| ours-python | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-18-ours-python-native-F32-t4.log) |  |  |  |  |  |
| ours-c | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-19-ours-c-native-F32-t4.log) |  |  |  |  |  |
| ours-fast-python | portable | F32 | 4 | one call per text (no batching) | 3.7 | 20.0 | 673 | 69.571 | 12.825 | 923 |
| ours-fast-c | portable | F32 | 4 | one call per text (no batching) | 3.8 | 20.8 | 692 | 67.853 | 12.327 | 897 |
| ours-fast-python | native | F32 | 4 | one call per text (no batching) | 3.5 | 19.6 | 643 | 72.999 | 13.083 | 924 |
| ours-fast-c | native | F32 | 4 | one call per text (no batching) | 3.6 | 20.0 | 665 | 70.521 | 12.804 | 897 |
| author | native | F16 | 2 | wrapper default (docs 16, queries 64) | 2.9 | 16.9 | 528 | 89.818 | 15.117 | 523 |
| author | native | Q8_0 | 2 | wrapper default (docs 16, queries 64) | 3.2 | 18.8 | 587 | 80.794 | 13.594 | 421 |
| author | native | Q4_0 | 2 | wrapper default (docs 16, queries 64) | 3.1 | 18.8 | 578 | 82.297 | 13.585 | 368 |
| author | native | F16 | 4 | wrapper default (docs 16, queries 64) | 3.4 | 19.8 | 621 | 76.411 | 12.903 | 524 |
| author | native | Q8_0 | 4 | wrapper default (docs 16, queries 64) | 3.5 | 20.9 | 657 | 72.174 | 12.255 | 421 |
| author | native | Q4_0 | 4 | wrapper default (docs 16, queries 64) | 4.2 | 25.1 | 775 | 61.352 | 10.216 | 369 |

