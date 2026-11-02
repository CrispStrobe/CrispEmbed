# Unlimited-OCR-2 trial-fix verification

Commit `277d9ea1f5ae1cf11a682fd9878d7eeba883c41d`, Intel(R) Xeon(R) CPU @ 2.20GHz x4 (avx512f=False), build 286.3s. Pages re-rendered; sha256 == trial: {'paragraphs': True, 'table': True, 'receipt': True, 'german': True, 'two-column': True, 'a4-dense': True, 'scan-sim': True}.

## Checks

- **unit_tests_pass**: True
- **f16_default_ids_equal_original_all4**: True
- **f16_default_text_bytes_equal_original_all4**: False
- **q8_default_ids_equal_original**: {'table': False, 'receipt': False, 'paragraphs': True, 'german': True}
- **q8_text_bytes_equal_when_ids_equal**: False
- **before_after_ids_equal**: {'table': True, 'receipt': True, 'paragraphs': True, 'german': True}
- **ablation_ids_equal_after**: {'ablate-cache-only': True, 'ablate-chunk-only': True}
- **vision_bytes_equal_cache_on_off**: {'unlimited-ocr2': True, 'unlimited-ocr2-q8': True}
- **exact_decoder_default_ids_equal_original**: [True]
- **native_build_ids_equal_original**: {'unlimited-ocr2/table': True, 'unlimited-ocr2/receipt': True, 'unlimited-ocr2-q8/table': False}

## Runs

| tag | model | page | exit | wall s | RSS MB | ids | ids==orig | first diff | text==orig |
|---|---|---|---|---|---|---|---|---|---|
| after | unlimited-ocr2 | table | 0 | 301.42 | 12820.9 | 138 | True | None | False |
| after | unlimited-ocr2 | receipt | 0 | 414.66 | 12820.9 | 306 | True | None | False |
| after | unlimited-ocr2 | paragraphs | 0 | 310.13 | 12821.0 | 146 | True | None | False |
| after | unlimited-ocr2 | german | 0 | 303.39 | 12821.0 | 164 | True | None | False |
| after | unlimited-ocr2-q8 | table | 0 | 179.55 | 6867.9 | 138 | False | 7 | False |
| after | unlimited-ocr2-q8 | receipt | 0 | 246.65 | 6868.3 | 306 | False | 131 | False |
| after | unlimited-ocr2-q8 | paragraphs | 0 | 184.15 | 6867.8 | 146 | True | None | False |
| after | unlimited-ocr2-q8 | german | 0 | 179.51 | 6867.7 | 164 | True | None | False |
| before | unlimited-ocr2 | table | 0 | 309.49 | 12820.9 | 138 | True | None | False |
| before | unlimited-ocr2 | receipt | 0 | 444.79 | 11655.8 | 306 | True | None | False |
| before | unlimited-ocr2 | paragraphs | 0 | 356.73 | 12820.2 | 146 | True | None | False |
| before | unlimited-ocr2 | german | 0 | 316.41 | 12820.5 | 164 | True | None | False |
| clean | unlimited-ocr2 | table | 0 | 310.42 | 12820.4 | 138 | True | None | False |
| legacy-text | unlimited-ocr2 | table | 0 | 334.34 | 12820.5 | 138 | True | None | False |
| untiled | unlimited-ocr2 | table | 0 | 78.15 | 12820.9 | 138 | False | 7 | False |
| ablate-cache-only | unlimited-ocr2 | table | 0 | 302.47 | 12820.3 | 138 | True | None | False |
| ablate-chunk-only | unlimited-ocr2 | table | 0 | 307.91 | 12820.8 | 138 | True | None | False |
| vision-dump-cache1 | unlimited-ocr2 | table | 0 | 262.88 | 12820.4 | 1 | False | 1 | False |
| vision-dump-cache0 | unlimited-ocr2 | table | 0 | 292.46 | 12820.9 | 1 | False | 1 | False |
| vision-dump-cache1 | unlimited-ocr2-q8 | table | 0 | 174.31 | 6867.8 | 1 | False | 1 | False |
| vision-dump-cache0 | unlimited-ocr2-q8 | table | 0 | 159.26 | 6867.8 | 1 | False | 1 | False |
| after-exact-decoder | unlimited-ocr2 | paragraphs | 0 | 1777.22 | 12820.0 | 146 | True | None | False |
| after-native-build | unlimited-ocr2 | table | 0 | 344.73 | 12821.2 | 138 | True | None | False |
| after-native-build | unlimited-ocr2 | receipt | 0 | 401.8 | 12820.7 | 306 | True | None | False |
| after-native-build | unlimited-ocr2-q8 | table | 0 | 214.52 | 6868.5 | 138 | False | 7 | False |

## Speed (F16, fast decoder, tiled): before -> after

| page | before s | after s | speedup |
|---|---|---|---|
| table | 309.49 | 301.42 | 1.03x |
| receipt | 444.79 | 414.66 | 1.07x |
| paragraphs | 356.73 | 310.13 | 1.15x |
| german | 316.41 | 303.39 | 1.04x |

## Stage timing (table, F16)

### before
```
[time] sam.patch_embed 54ms
[time] sam.layers 11360ms
[time] sam.neck_downsample 127ms
[time] sam.patch_embed 54ms
[time] sam.layers 11238ms
[time] sam.neck_downsample 126ms
[time] sam.patch_embed 58ms
[time] sam.layers 11205ms
[time] sam.neck_downsample 126ms
[time] sam.patch_embed 56ms
[time] sam.layers 11496ms
[time] sam.neck_downsample 129ms
[time] sam.patch_embed 57ms
[time] sam.layers 11390ms
[time] sam.neck_downsample 127ms
[time] sam.patch_embed 55ms
[time] sam.layers 11286ms
[time] sam.neck_downsample 135ms
[time] sam.patch_embed 62ms
[time] sam.layers 12051ms
[time] sam.neck_downsample 256ms
[time] sam.patch_embed 119ms
[time] sam.layers 16071ms
[time] sam.neck_downsample 154ms
[time] sam.patch_embed 60ms
[time] sam.layers 11333ms
[time] sam.neck_downsample 129ms
[time] sam.patch_embed 55ms
[time] sam.layers 11277ms
[time] sam.neck_downsample 134ms
[time] sam.patch_embed 85ms
[time] sam.layers 11197ms
[time] sam.neck_downsample 126ms
[time] gundam_tiles 175200ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
### after
```
[time] sam.patch_embed 29ms
[time] sam.layers 11069ms
[time] sam.neck_downsample 79ms
[time] sam.patch_embed 30ms
[time] sam.layers 11041ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 29ms
[time] sam.layers 11023ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 28ms
[time] sam.layers 11028ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 30ms
[time] sam.layers 11053ms
[time] sam.neck_downsample 76ms
[time] sam.patch_embed 29ms
[time] sam.layers 11176ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 30ms
[time] sam.layers 11118ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 29ms
[time] sam.layers 11188ms
[time] sam.neck_downsample 77ms
[time] sam.patch_embed 30ms
[time] sam.layers 11309ms
[time] sam.neck_downsample 76ms
[time] sam.patch_embed 30ms
[time] sam.layers 11509ms
[time] sam.neck_downsample 78ms
[time] sam.patch_embed 30ms
[time] sam.layers 11389ms
[time] sam.neck_downsample 77ms
[time] gundam_tiles 160495ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
### ablate-cache-only
```
[time] sam.patch_embed 74ms
[time] sam.layers 16711ms
[time] sam.neck_downsample 83ms
[time] sam.patch_embed 40ms
[time] sam.layers 11557ms
[time] sam.neck_downsample 81ms
[time] sam.patch_embed 31ms
[time] sam.layers 11389ms
[time] sam.neck_downsample 77ms
[time] sam.patch_embed 30ms
[time] sam.layers 11259ms
[time] sam.neck_downsample 76ms
[time] sam.patch_embed 31ms
[time] sam.layers 11219ms
[time] sam.neck_downsample 76ms
[time] sam.patch_embed 29ms
[time] sam.layers 11323ms
[time] sam.neck_downsample 74ms
[time] sam.patch_embed 31ms
[time] sam.layers 11196ms
[time] sam.neck_downsample 75ms
[time] sam.patch_embed 29ms
[time] sam.layers 11251ms
[time] sam.neck_downsample 80ms
[time] sam.patch_embed 35ms
[time] sam.layers 11241ms
[time] sam.neck_downsample 76ms
[time] sam.patch_embed 30ms
[time] sam.layers 11287ms
[time] sam.neck_downsample 75ms
[time] sam.patch_embed 32ms
[time] sam.layers 11212ms
[time] sam.neck_downsample 76ms
[time] gundam_tiles 170927ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
### ablate-chunk-only
```
[time] sam.patch_embed 55ms
[time] sam.layers 11718ms
[time] sam.neck_downsample 128ms
[time] sam.patch_embed 65ms
[time] sam.layers 12430ms
[time] sam.neck_downsample 148ms
[time] sam.patch_embed 59ms
[time] sam.layers 11516ms
[time] sam.neck_downsample 136ms
[time] sam.patch_embed 59ms
[time] sam.layers 11411ms
[time] sam.neck_downsample 128ms
[time] sam.patch_embed 56ms
[time] sam.layers 11309ms
[time] sam.neck_downsample 141ms
[time] sam.patch_embed 57ms
[time] sam.layers 11293ms
[time] sam.neck_downsample 130ms
[time] sam.patch_embed 59ms
[time] sam.layers 11225ms
[time] sam.neck_downsample 128ms
[time] sam.patch_embed 56ms
[time] sam.layers 11142ms
[time] sam.neck_downsample 128ms
[time] sam.patch_embed 58ms
[time] sam.layers 11177ms
[time] sam.neck_downsample 129ms
[time] sam.patch_embed 60ms
[time] sam.layers 11160ms
[time] sam.neck_downsample 128ms
[time] sam.patch_embed 62ms
[time] sam.layers 11164ms
[time] sam.neck_downsample 126ms
[time] gundam_tiles 169319ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
### untiled
```
[time] init.load_tensors 13071ms
[time] init.stack_moe_experts 15ms
[time] sam.patch_embed 175ms
[time] sam.layers 36681ms
[time] sam.neck_downsample 265ms
[time] sam 37122ms
[time] clip 5091ms
[time] fuse_project 297ms
[time] vision_assembly 0ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
[uocr2-fast] prefill layer 6 done
[uocr2-fast] prefill layer 7 done
[uocr2-fast] prefill layer 8 done
[uocr2-fast] prefill layer 9 done
[uocr2-fast] prefill layer 10 done
[uocr2-fast] prefill layer 11 done
[uocr2-fast] prefill done S=279 7681.8 ms
[uocr2-fast] decode done: 138 tokens, 21206.1 ms total
```
### after-native-build
```
[time] sam.patch_embed 39ms
[time] sam.layers 11929ms
[time] sam.neck_downsample 94ms
[time] sam.patch_embed 37ms
[time] sam.layers 11515ms
[time] sam.neck_downsample 92ms
[time] sam.patch_embed 35ms
[time] sam.layers 11284ms
[time] sam.neck_downsample 90ms
[time] sam.patch_embed 36ms
[time] sam.layers 11358ms
[time] sam.neck_downsample 90ms
[time] sam.patch_embed 38ms
[time] sam.layers 11253ms
[time] sam.neck_downsample 91ms
[time] sam.patch_embed 35ms
[time] sam.layers 11231ms
[time] sam.neck_downsample 91ms
[time] sam.patch_embed 35ms
[time] sam.layers 11199ms
[time] sam.neck_downsample 89ms
[time] sam.patch_embed 35ms
[time] sam.layers 18813ms
[time] sam.neck_downsample 158ms
[time] sam.patch_embed 38ms
[time] sam.layers 11505ms
[time] sam.neck_downsample 94ms
[time] sam.patch_embed 39ms
[time] sam.layers 11348ms
[time] sam.neck_downsample 90ms
[time] sam.patch_embed 35ms
[time] sam.layers 11220ms
[time] sam.neck_downsample 90ms
[time] gundam_tiles 172381ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
## `clean` rendering (table)
```
table[56, 99, 921, 658]<table>ProductRegionUnitsRevenueNotebook A5North1,2508,750.00Desk LampSouth43012,040.50Office ChairEast9617,184.00Monitor 27&quot;West21242,188.00Total1,98880,162.50</table>
```
## `legacy-text` rendering (table)
```
table[56, 99, 921, 658]<table>ProductRegionUnitsRevenueNotebook A5North1,2508,750.00Desk LampSouth43012,040.50Office ChairEast9617,184.00Monitor 27&quot;West21242,188.00Total1,98880,162.50</table>
```
