<!-- Kaggle crispembed-moedern-115m-vago-compare at cd653779 (pre fast path); diagnostic only -->
# MoEdern 115M: CrispEmbed vs author runtime (diagnostic)

Source `cd653779e42b9e2a42498cf4edecf132ebe04129`, ggml `3612c0af93f2`, workload sha256 `a31c3663e995bf0c`, 3 timed repetitions (median), warm-up excluded.

Batching: the author wrapper batches (documents 16, queries 64 per engine request by default; also measured with batch_size=1). CrispEmbed has no batch API: one encode call per text. CrispEmbed accepts 1-2 threads only, so its 4-thread rows are expected to fail.

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
| author | native | F32 | 2 | wrapper default (docs 16, queries 64) | 1.5 | 8.9 | 285 | 165.756 | 28.860 | 729 |
| author | native | F32 | 2 | batch_size=1 | 1.5 | 7.9 | 269 | 173.739 | 32.597 | 589 |
| author | portable | F32 | 2 | wrapper default (docs 16, queries 64) | 1.5 | 9.3 | 287 | 165.319 | 27.648 | 732 |
| author | portable | F32 | 2 | batch_size=1 | 1.6 | 8.3 | 287 | 162.239 | 30.798 | 589 |
| ours-python | portable | F32 | 2 | one call per text (no batching) | 0.7 | 5.9 | 135 | 366.569 | 43.390 | 922 |
| ours-c | portable | F32 | 2 | one call per text (no batching) | 0.7 | 5.9 | 136 | 366.360 | 43.177 | 896 |
| ours-python | native | F32 | 2 | one call per text (no batching) | 0.6 | 5.2 | 122 | 405.004 | 48.903 | 923 |
| ours-c | native | F32 | 2 | one call per text (no batching) | 0.6 | 5.0 | 120 | 410.745 | 50.849 | 896 |
| author | native | F32 | 4 | wrapper default (docs 16, queries 64) | 1.8 | 10.5 | 331 | 143.119 | 24.422 | 730 |
| author | native | F32 | 4 | batch_size=1 | 1.7 | 9.1 | 317 | 147.098 | 27.989 | 589 |
| author | portable | F32 | 4 | wrapper default (docs 16, queries 64) | 1.8 | 10.5 | 332 | 142.738 | 24.307 | 731 |
| author | portable | F32 | 4 | batch_size=1 | 1.7 | 9.2 | 317 | 147.260 | 27.880 | 589 |
| ours-python | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-12-ours-python-portable-F32-t4.log) |  |  |  |  |  |
| ours-c | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-13-ours-c-portable-F32-t4.log) |  |  |  |  |  |
| ours-python | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-14-ours-python-native-F32-t4.log) |  |  |  |  |  |
| ours-c | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-15-ours-c-native-F32-t4.log) |  |  |  |  |  |
| author | native | F16 | 2 | wrapper default (docs 16, queries 64) | 1.5 | 9.0 | 285 | 166.048 | 28.288 | 522 |
| author | native | Q8_0 | 2 | wrapper default (docs 16, queries 64) | 2.0 | 11.8 | 377 | 125.345 | 21.747 | 422 |
| author | native | Q4_0 | 2 | wrapper default (docs 16, queries 64) | 3.4 | 20.2 | 624 | 76.260 | 12.665 | 370 |
| author | native | F16 | 4 | wrapper default (docs 16, queries 64) | 1.9 | 10.9 | 344 | 137.698 | 23.405 | 523 |
| author | native | Q8_0 | 4 | wrapper default (docs 16, queries 64) | 2.4 | 14.2 | 443 | 107.084 | 18.034 | 422 |
| author | native | Q4_0 | 4 | wrapper default (docs 16, queries 64) | 4.0 | 24.3 | 741 | 64.271 | 10.519 | 370 |

