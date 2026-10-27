<!-- Kaggle crispembed-moedern-27m-vago-compare v2; diagnostic only -->
# MoEdern 27M: CrispEmbed vs author runtime (diagnostic)

Source `4b5aa07a3d5f212d2f27996dd4e0d9d801c05e9f`, ggml `3612c0af93f2`, workload sha256 `a31c3663e995bf0c`, 3 timed repetitions (median), warm-up excluded.

Batching: the author wrapper batches (documents 16, queries 64 per engine request by default; also measured with batch_size=1). CrispEmbed has no batch API: one encode call per text. CrispEmbed's default path accepts 1-2 threads only, so its 4-thread rows are expected to fail; ours-fast rows (CRISPEMBED_MOEDERN_FAST=1) accept up to 64 threads under the unchanged parity gates.

## Outputs vs frozen original

| arm | IDs equal | selected max abs | min cosine | row-count mismatches | MaxSim max abs diff |
|---|---|---|---|---|---|
| author-F32-native | 9/10 | 0.3431236743927002 | 0.09772873494958859 | 0 | 7.29114 |
| author-F32-portable | 9/10 | 0.3431236743927002 | 0.09772873494958859 | 0 | 7.29114 |
| author-F16-native | 9/10 | 0.34316226840019226 | 0.09771180490088835 | 0 | 7.28912 |
| author-Q8_0-native | 9/10 | 0.34121865034103394 | 0.09873818534682141 | 0 | 7.35207 |
| author-Q4_0-native | 9/10 | 0.34973224997520447 | 0.09943874989820634 | 0 | 7.25518 |
| ours-portable | 10/10 | 4.470348358154297e-07 | 0.9999999999984532 | 0 | 1.90735e-06 |
| ours-native | 10/10 | 4.470348358154297e-07 | 0.9999999999984532 | 0 | 1.90735e-06 |
| ours-fast-portable | 10/10 | 3.8929283618927e-07 | 0.9999999999984303 | 0 | 1.90735e-06 |
| ours-fast-native | 10/10 | 3.8929283618927e-07 | 0.9999999999984303 | 0 | 1.90735e-06 |

## CrispEmbed fast vs default path (same build, selected vectors)

| build | max abs | min cosine |
|---|---|---|
| portable | 4.5727938413619995e-07 | 0.9999999999979999 |
| native | 4.5727938413619995e-07 | 0.9999999999979999 |

## Author vs CrispEmbed (portable F32)

| arm | IDs equal | selected max abs | min cosine | row-count mismatches |
|---|---|---|---|---|
| author-F32-native_vs_ours | 9 | 0.3431237190961838 | 0.09772874477800964 | 0 |
| author-F32-portable_vs_ours | 9 | 0.3431237190961838 | 0.09772874477800964 | 0 |
| author-F16-native_vs_ours | 9 | 0.34316231310367584 | 0.09771181474427491 | 0 |
| author-Q8_0-native_vs_ours | 9 | 0.3412186950445175 | 0.09873819438051927 | 0 |
| author-Q4_0-native_vs_ours | 9 | 0.34973229467868805 | 0.09943875876596862 | 0 |

## Speed

| runtime | build | model | threads | batching | docs/s | queries/s | tokens/s | doc s (median) | query s (median) | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| author | native | F32 | 2 | wrapper default (docs 16, queries 64) | 9.0 | 54.9 | 1670 | 28.476 | 4.661 | 312 |
| author | native | F32 | 2 | batch_size=1 | 8.6 | 43.5 | 1550 | 29.882 | 5.892 | 240 |
| author | portable | F32 | 2 | wrapper default (docs 16, queries 64) | 9.1 | 55.4 | 1696 | 28.021 | 4.617 | 312 |
| author | portable | F32 | 2 | batch_size=1 | 8.6 | 43.4 | 1560 | 29.694 | 5.904 | 238 |
| ours-python | portable | F32 | 2 | one call per text (no batching) | 2.0 | 16.2 | 380 | 130.494 | 15.754 | 256 |
| ours-c | portable | F32 | 2 | one call per text (no batching) | 2.0 | 16.9 | 388 | 127.851 | 15.165 | 228 |
| ours-python | native | F32 | 2 | one call per text (no batching) | 1.9 | 15.1 | 359 | 137.583 | 16.954 | 255 |
| ours-c | native | F32 | 2 | one call per text (no batching) | 1.9 | 15.5 | 362 | 136.664 | 16.551 | 228 |
| ours-fast-python | portable | F32 | 2 | one call per text (no batching) | 9.9 | 53.8 | 1814 | 25.793 | 4.760 | 255 |
| ours-fast-c | portable | F32 | 2 | one call per text (no batching) | 10.3 | 56.2 | 1884 | 24.875 | 4.555 | 229 |
| ours-fast-python | native | F32 | 2 | one call per text (no batching) | 9.3 | 51.0 | 1702 | 27.565 | 5.016 | 255 |
| ours-fast-c | native | F32 | 2 | one call per text (no batching) | 9.6 | 54.1 | 1769 | 26.588 | 4.729 | 229 |
| author | native | F32 | 4 | wrapper default (docs 16, queries 64) | 8.2 | 49.2 | 1520 | 31.294 | 5.199 | 313 |
| author | native | F32 | 4 | batch_size=1 | 8.1 | 41.9 | 1466 | 31.723 | 6.109 | 238 |
| author | portable | F32 | 4 | wrapper default (docs 16, queries 64) | 8.5 | 51.2 | 1572 | 30.249 | 4.997 | 313 |
| author | portable | F32 | 4 | batch_size=1 | 8.1 | 42.2 | 1475 | 31.578 | 6.071 | 239 |
| ours-python | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-16-ours-python-portable-F32-t4.log) |  |  |  |  |  |
| ours-c | portable | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-17-ours-c-portable-F32-t4.log) |  |  |  |  |  |
| ours-python | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-18-ours-python-native-F32-t4.log) |  |  |  |  |  |
| ours-c | native | F32 | 4 | one call per text (no batching) | error: exit 1 (see bench-19-ours-c-native-F32-t4.log) |  |  |  |  |  |
| ours-fast-python | portable | F32 | 4 | one call per text (no batching) | 9.9 | 56.5 | 1825 | 25.818 | 4.531 | 256 |
| ours-fast-c | portable | F32 | 4 | one call per text (no batching) | 10.4 | 59.3 | 1913 | 24.647 | 4.320 | 229 |
| ours-fast-python | native | F32 | 4 | one call per text (no batching) | 9.4 | 53.7 | 1731 | 27.312 | 4.765 | 254 |
| ours-fast-c | native | F32 | 4 | one call per text (no batching) | 9.8 | 57.0 | 1813 | 26.093 | 4.491 | 229 |
| author | native | F16 | 2 | wrapper default (docs 16, queries 64) | 7.9 | 47.5 | 1462 | 32.553 | 5.385 | 268 |
| author | native | Q8_0 | 2 | wrapper default (docs 16, queries 64) | 9.5 | 58.6 | 1773 | 26.912 | 4.367 | 242 |
| author | native | Q4_0 | 2 | wrapper default (docs 16, queries 64) | 13.5 | 85.5 | 2518 | 19.027 | 2.994 | 232 |
| author | native | F16 | 4 | wrapper default (docs 16, queries 64) | 8.6 | 51.3 | 1592 | 29.845 | 4.993 | 268 |
| author | native | Q8_0 | 4 | wrapper default (docs 16, queries 64) | 9.2 | 55.8 | 1715 | 27.763 | 4.588 | 243 |
| author | native | Q4_0 | 4 | wrapper default (docs 16, queries 64) | 16.4 | 103.6 | 3060 | 15.588 | 2.471 | 231 |

