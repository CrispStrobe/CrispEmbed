# MiniCPM-OCR-1.0 bench @ 68e003b55a

CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4

| arm | exact | text | CER max | wall s (sum) | vision enc ms | proj ms | prefill ms | decode ms/tok |
|---|---|---|---|---|---|---|---|---|
| f16-t4 | True | True | 0.0 | 3264 | 714478 | 8705 | 447798 | 4765.0 |
| f16-t4-graph | True | True | 0.0 | 1317 | 700529 | 8661 | 93447 | 1175.2 |
| f16-t4-graph-native | True | True | 0.0 | 1000 | 705673 | 8568 | 201979 | 106.8 |
| f16-t4-graph-f32w | True | True | 0.0 | 925 | 700163 | 8604 | 88434 | 168.6 |
| f16-t4-vision-f32 | True | True | 0.0 | 2738 | 334964 | 2952 | 424964 | 4528.7 |
| f16-t4-vision-f32-batch | True | True | 0.0 | 2715 | 284625 | 2578 | 413170 | 4510.4 |
| f16-t4-fast | True | True | 0.0 | 829 | 285482 | 2526 | 80874 | 1010.7 |
| f16-t4-fast-flash | True | True | 0.0 | 762 | 209657 | 2539 | 81356 | 1033.1 |
| f16-t2-fast | True | True | 0.0 | 921 | 339392 | 2978 | 91886 | 1074.9 |

## Per case (first arm)

| case | wall s | preprocess | vision enc | proj | prefill (tok) | decode steps x mean ms |
|---|---|---|---|---|---|---|
| line | 139.103 | 22.848 | 33586.773 | 387.045 | 12788.347 (256) | 18 x 4807.924 |
| para-en | 550.344 | 142.55 | 168172.589 | 2053.432 | 81959.541 (1290) | 62 x 4719.834 |
| table | 533.427 | 136.242 | 167700.098 | 2031.911 | 81253.444 (1290) | 55 x 5055.441 |
| tall | 1973.844 | 261.687 | 311785.625 | 3838.149 | 258996.47 (2372) | 278 x 5001.807 |
| tiny | 66.803 | 14.91 | 33232.688 | 394.432 | 12799.951 (274) | 2 x 4239.8 |

## Baselines

- transformers-f32-t4: line 16.824s CER 0.0, para-en 82.329s CER 0.0, table 80.27s CER 0.0, tall 192.295s CER 0.0, tiny 13.739s CER 0.0

Skipped: []
