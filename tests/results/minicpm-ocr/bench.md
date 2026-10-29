# MiniCPM-OCR-1.0 bench @ c9718788bd

CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4

| arm | exact | text | CER max | wall s (sum) | vision enc ms | proj ms | prefill ms | decode ms/tok |
|---|---|---|---|---|---|---|---|---|
| q8_0-t4 | False | False | 1.0 | 11 | 0 | 0 | 0 |  |
| f16-t4-no-cache | True | True | 0.0 | 2781 | 587516 | 7284 | 386533 | 4194.9 |
| f32-t4 | True | True | 0.0 | 2310 | 288578 | 2436 | 386323 | 3600.2 |
| q8_0-t2 | False | False | 1.0 | 19 | 0 | 0 | 0 |  |
| f16-t4-no-timing | None | True | 0.0 | 2791 | 0 | 0 | 0 |  |

## Per case (first arm)

| case | wall s | preprocess | vision enc | proj | prefill (tok) | decode steps x mean ms |
|---|---|---|---|---|---|---|
| line | 2.212 | None | None | None | None (None) | None x None |
| para-en | 2.226 | None | None | None | None (None) | None x None |
| table | 2.208 | None | None | None | None (None) | None x None |
| tall | 2.202 | None | None | None | None (None) | None x None |
| tiny | 2.183 | None | None | None | None (None) | None x None |

## Baselines


Skipped: []
