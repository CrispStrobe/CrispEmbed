# Unlimited-OCR-2 vision speed gates

Commit `866d4dbb61cac022a424483231cdbb96f8d230b5`, Intel(R) Xeon(R) CPU @ 2.20GHz x4 (avx512f=False). Pages sha256 == trial: {'paragraphs': True, 'table': True, 'receipt': True, 'german': True, 'two-column': True, 'a4-dense': True, 'scan-sim': True}.

## Checks

- **unit_tests_pass**: True
- **after_ids_equal_original_all4**: True
- **after_text_bytes_equal_original_all4**: True
- **after_wall_seconds_sum**: 1162.2
- **after_peak_rss_mb_max**: 12821.6
- **vf32_ids_equal_original_all4**: True
- **vf32_text_bytes_equal_original_all4**: True
- **vf32_wall_seconds_sum**: 804.9
- **vf32_peak_rss_mb_max**: 12821.2
- **batch_ids_equal_original_all4**: True
- **batch_text_bytes_equal_original_all4**: True
- **batch_wall_seconds_sum**: 1191.1
- **batch_peak_rss_mb_max**: 12821.0
- **vf32-batch_ids_equal_original_all4**: True
- **vf32-batch_text_bytes_equal_original_all4**: True
- **vf32-batch_wall_seconds_sum**: 785.8
- **vf32-batch_peak_rss_mb_max**: 12821.5
- **vf32-batch4_ids_equal_original_all4**: True
- **vf32-batch4_text_bytes_equal_original_all4**: True
- **vf32-batch4_wall_seconds_sum**: 799.7
- **vf32-batch4_peak_rss_mb_max**: 12821.3
- **vision_bytes_batch_eq_unbatched_f16**: True
- **vision_bytes_batch_eq_unbatched_f32**: True

| arm | page | exit | wall s | RSS MB | ids | ids==orig | first diff | text==orig |
|---|---|---|---|---|---|---|---|---|
| after | table | 0 | 258.94 | 12821.3 | 138 | True | None | True |
| vf32 | table | 0 | 174.08 | 12820.8 | 138 | True | None | True |
| batch | table | 0 | 271.28 | 12820.9 | 138 | True | None | True |
| vf32-batch | table | 0 | 162.61 | 12821.5 | 138 | True | None | True |
| vf32-batch4 | table | 0 | 172.58 | 12821.3 | 138 | True | None | True |
| after | receipt | 0 | 369.62 | 12821.6 | 306 | True | None | True |
| vf32 | receipt | 0 | 246.08 | 12820.7 | 306 | True | None | True |
| batch | receipt | 0 | 359.02 | 12821.0 | 306 | True | None | True |
| vf32-batch | receipt | 0 | 248.85 | 12820.8 | 306 | True | None | True |
| vf32-batch4 | receipt | 0 | 248.73 | 12820.5 | 306 | True | None | True |
| after | paragraphs | 0 | 263.05 | 12821.0 | 146 | True | None | True |
| vf32 | paragraphs | 0 | 192.61 | 12820.6 | 146 | True | None | True |
| batch | paragraphs | 0 | 262.29 | 12820.4 | 146 | True | None | True |
| vf32-batch | paragraphs | 0 | 198.75 | 12821.1 | 146 | True | None | True |
| vf32-batch4 | paragraphs | 0 | 190.16 | 12821.1 | 146 | True | None | True |
| after | german | 0 | 270.61 | 12821.4 | 164 | True | None | True |
| vf32 | german | 0 | 192.14 | 12821.2 | 164 | True | None | True |
| batch | german | 0 | 298.49 | 12820.6 | 164 | True | None | True |
| vf32-batch | german | 0 | 175.63 | 12820.9 | 164 | True | None | True |
| vf32-batch4 | german | 0 | 188.24 | 12820.6 | 164 | True | None | True |
| vision-dump-after | table | 0 | 251.59 | 12821.0 | 1 | False | 1 | False |
| vision-dump-vf32 | table | 0 | 137.97 | 12820.3 | 1 | False | 1 | False |
| vision-dump-batch | table | 0 | 239.85 | 12820.6 | 1 | False | 1 | False |
| vision-dump-vf32-batch | table | 0 | 158.03 | 12820.4 | 1 | False | 1 | False |
| vision-dump-vf32-batch4 | table | 0 | 137.68 | 12821.0 | 1 | False | 1 | False |

### after (table) stage timing
```
[time] tile 0,0 sam 9942ms clip 1713ms proj 102ms
[time] sam.patch_embed 27ms
[time] sam.layers 9763ms
[time] sam.neck_downsample 67ms
[time] tile 1,0 sam 9858ms clip 1685ms proj 101ms
[time] sam.patch_embed 27ms
[time] sam.layers 9751ms
[time] sam.neck_downsample 68ms
[time] tile 2,0 sam 9847ms clip 1706ms proj 101ms
[time] sam.patch_embed 27ms
[time] sam.layers 9773ms
[time] sam.neck_downsample 74ms
[time] tile 3,0 sam 9875ms clip 1710ms proj 102ms
[time] sam.patch_embed 28ms
[time] sam.layers 9890ms
[time] sam.neck_downsample 68ms
[time] tile 0,1 sam 9988ms clip 1751ms proj 101ms
[time] sam.patch_embed 31ms
[time] sam.layers 12878ms
[time] sam.neck_downsample 117ms
[time] tile 1,1 sam 13026ms clip 3072ms proj 180ms
[time] sam.patch_embed 50ms
[time] sam.layers 12843ms
[time] sam.neck_downsample 67ms
[time] tile 2,1 sam 12962ms clip 1685ms proj 100ms
[time] sam.patch_embed 27ms
[time] sam.layers 9936ms
[time] sam.neck_downsample 75ms
[time] tile 3,1 sam 10039ms clip 1887ms proj 102ms
[time] sam.patch_embed 28ms
[time] sam.layers 10136ms
[time] sam.neck_downsample 69ms
[time] tile 0,2 sam 10234ms clip 1748ms proj 105ms
[time] sam.patch_embed 27ms
[time] sam.layers 9914ms
[time] sam.neck_downsample 70ms
[time] tile 1,2 sam 10011ms clip 1705ms proj 101ms
[time] sam.patch_embed 27ms
[time] sam.layers 10140ms
[time] sam.neck_downsample 69ms
[time] tile 2,2 sam 10237ms clip 1789ms proj 99ms
[time] sam.patch_embed 27ms
[time] sam.layers 9878ms
[time] sam.neck_downsample 67ms
[time] tile 3,2 sam 9973ms clip 1683ms proj 101ms
[time] gundam_tiles 149606ms
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
[uocr2-fast] prefill done S=1509 38206.5 ms
[uocr2-fast] decode done: 138 tokens, 65503.2 ms total
```

### vf32 (table) stage timing
```
[time] tile 0,0 sam 4774ms clip 718ms proj 100ms
[time] sam.patch_embed 27ms
[time] sam.layers 4471ms
[time] sam.neck_downsample 69ms
[time] tile 1,0 sam 4569ms clip 647ms proj 104ms
[time] sam.patch_embed 28ms
[time] sam.layers 4529ms
[time] sam.neck_downsample 68ms
[time] tile 2,0 sam 4626ms clip 680ms proj 101ms
[time] sam.patch_embed 28ms
[time] sam.layers 4494ms
[time] sam.neck_downsample 69ms
[time] tile 3,0 sam 4591ms clip 655ms proj 99ms
[time] sam.patch_embed 28ms
[time] sam.layers 4548ms
[time] sam.neck_downsample 69ms
[time] tile 0,1 sam 4646ms clip 627ms proj 98ms
[time] sam.patch_embed 27ms
[time] sam.layers 4499ms
[time] sam.neck_downsample 69ms
[time] tile 1,1 sam 4596ms clip 629ms proj 111ms
[time] sam.patch_embed 28ms
[time] sam.layers 4523ms
[time] sam.neck_downsample 69ms
[time] tile 2,1 sam 4621ms clip 626ms proj 99ms
[time] sam.patch_embed 27ms
[time] sam.layers 4476ms
[time] sam.neck_downsample 69ms
[time] tile 3,1 sam 4572ms clip 696ms proj 100ms
[time] sam.patch_embed 27ms
[time] sam.layers 4436ms
[time] sam.neck_downsample 69ms
[time] tile 0,2 sam 4532ms clip 624ms proj 101ms
[time] sam.patch_embed 28ms
[time] sam.layers 4519ms
[time] sam.neck_downsample 69ms
[time] tile 1,2 sam 4617ms clip 620ms proj 101ms
[time] sam.patch_embed 27ms
[time] sam.layers 4475ms
[time] sam.neck_downsample 69ms
[time] tile 2,2 sam 4573ms clip 648ms proj 101ms
[time] sam.patch_embed 30ms
[time] sam.layers 4544ms
[time] sam.neck_downsample 69ms
[time] tile 3,2 sam 4644ms clip 627ms proj 100ms
[time] gundam_tiles 64558ms
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
[uocr2-fast] prefill done S=1509 37955.4 ms
[uocr2-fast] decode done: 138 tokens, 70314.7 ms total
```

### batch (table) stage timing
```
[time] init.load_tensors 25905ms
[time] init.stack_moe_experts 14ms
[time] sam.patch_embed 151ms
[time] sam.layers 31153ms
[time] sam.neck_downsample 250ms
[time] sam 31556ms
[time] clip 4248ms
[time] fuse_project 268ms
[time] vision_assembly 0ms
[time] sam.patch_embed 387ms
[time] sam.layers 120187ms
[time] sam.neck_downsample 80ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 71ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] tile batch 0+12 sam 121420ms clip 19839ms proj 1205ms
[time] gundam_tiles 142666ms
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
[uocr2-fast] prefill done S=1509 38060.9 ms
[uocr2-fast] decode done: 138 tokens, 65688.6 ms total
```

### vf32-batch (table) stage timing
```
[time] init.load_tensors 6504ms
[time] init.vision_f32 2664ms
[time] init.stack_moe_experts 0ms
[time] sam.patch_embed 145ms
[time] sam.layers 16985ms
[time] sam.neck_downsample 236ms
[time] sam 17368ms
[time] clip 1596ms
[time] fuse_project 272ms
[time] vision_assembly 0ms
[time] sam.patch_embed 404ms
[time] sam.layers 57053ms
[time] sam.neck_downsample 87ms
[time] sam.neck_downsample 71ms
[time] sam.neck_downsample 70ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 70ms
[time] sam.neck_downsample 72ms
[time] sam.neck_downsample 68ms
[time] tile batch 0+12 sam 58317ms clip 7162ms proj 1228ms
[time] gundam_tiles 66909ms
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
[uocr2-fast] prefill done S=1509 38840.5 ms
[uocr2-fast] decode done: 138 tokens, 66258.1 ms total
```

### vf32-batch4 (table) stage timing
```
[time] init.load_tensors 6497ms
[time] init.vision_f32 2618ms
[time] init.stack_moe_experts 0ms
[time] sam.patch_embed 138ms
[time] sam.layers 16931ms
[time] sam.neck_downsample 234ms
[time] sam 17305ms
[time] clip 1688ms
[time] fuse_project 335ms
[time] vision_assembly 0ms
[time] sam.patch_embed 171ms
[time] sam.layers 28678ms
[time] sam.neck_downsample 76ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 70ms
[time] sam.neck_downsample 70ms
[time] tile batch 0+4 sam 29140ms clip 2389ms proj 405ms
[time] sam.patch_embed 119ms
[time] sam.layers 18424ms
[time] sam.neck_downsample 67ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 68ms
[time] tile batch 4+4 sam 18821ms clip 2385ms proj 395ms
[time] sam.patch_embed 124ms
[time] sam.layers 18295ms
[time] sam.neck_downsample 68ms
[time] sam.neck_downsample 67ms
[time] sam.neck_downsample 69ms
[time] sam.neck_downsample 68ms
[time] tile batch 8+4 sam 18698ms clip 2442ms proj 401ms
[time] gundam_tiles 75286ms
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
[uocr2-fast] prefill done S=1509 38285.6 ms
[uocr2-fast] decode done: 138 tokens, 67923.2 ms total
```
