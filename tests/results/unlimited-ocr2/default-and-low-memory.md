# Unlimited-OCR-2 vision speed gates

Commit `0d3974abe84508b994b068f66172a8d438fc7b9b`, Intel(R) Xeon(R) CPU @ 2.20GHz x4 (avx512f=False). Pages sha256 == trial: {'paragraphs': True, 'table': True, 'receipt': True, 'german': True, 'two-column': True, 'a4-dense': True, 'scan-sim': True}.

## Checks

- **unit_tests_pass**: True
- **default_ids_equal_original_all4**: True
- **default_text_bytes_equal_original_all4**: True
- **default_wall_seconds_sum**: 799.2
- **default_peak_rss_mb_max**: 12820.9
- **lowmem_ids_equal_original_all4**: True
- **lowmem_text_bytes_equal_original_all4**: True
- **lowmem_wall_seconds_sum**: 1286.7
- **lowmem_peak_rss_mb_max**: 12821.4
- **vision_bytes_batch_eq_unbatched_f16**: None
- **vision_bytes_batch_eq_unbatched_f32**: None

| arm | page | exit | wall s | RSS MB | ids | ids==orig | first diff | text==orig |
|---|---|---|---|---|---|---|---|---|
| default | table | 0 | 175.47 | 12820.9 | 138 | True | None | True |
| lowmem | table | 0 | 287.11 | 12820.8 | 138 | True | None | True |
| default | receipt | 0 | 256.06 | 12820.7 | 306 | True | None | True |
| lowmem | receipt | 0 | 408.48 | 12821.1 | 306 | True | None | True |
| default | paragraphs | 0 | 187.29 | 12820.9 | 146 | True | None | True |
| lowmem | paragraphs | 0 | 280.93 | 12820.8 | 146 | True | None | True |
| default | german | 0 | 180.43 | 12820.8 | 164 | True | None | True |
| lowmem | german | 0 | 310.14 | 12821.4 | 164 | True | None | True |
| vision-dump-default | table | 0 | 164.91 | 12820.9 | 1 | False | 1 | False |
| vision-dump-lowmem | table | 0 | 253.4 | 12821.5 | 1 | False | 1 | False |

### default (table) stage timing
```
[time] tile 0,0 sam 5120ms clip 722ms proj 102ms
[time] sam.patch_embed 31ms
[time] sam.layers 5017ms
[time] sam.neck_downsample 73ms
[time] tile 1,0 sam 5121ms clip 693ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4915ms
[time] sam.neck_downsample 74ms
[time] tile 2,0 sam 5019ms clip 703ms proj 104ms
[time] sam.patch_embed 30ms
[time] sam.layers 4945ms
[time] sam.neck_downsample 74ms
[time] tile 3,0 sam 5051ms clip 689ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4965ms
[time] sam.neck_downsample 73ms
[time] tile 0,1 sam 5068ms clip 693ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4942ms
[time] sam.neck_downsample 75ms
[time] tile 1,1 sam 5047ms clip 699ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4937ms
[time] sam.neck_downsample 74ms
[time] tile 2,1 sam 5041ms clip 691ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4892ms
[time] sam.neck_downsample 73ms
[time] tile 3,1 sam 4996ms clip 690ms proj 104ms
[time] sam.patch_embed 29ms
[time] sam.layers 4935ms
[time] sam.neck_downsample 73ms
[time] tile 0,2 sam 5038ms clip 681ms proj 101ms
[time] sam.patch_embed 31ms
[time] sam.layers 4887ms
[time] sam.neck_downsample 72ms
[time] tile 1,2 sam 4990ms clip 762ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 4882ms
[time] sam.neck_downsample 72ms
[time] tile 2,2 sam 4984ms clip 686ms proj 101ms
[time] sam.patch_embed 28ms
[time] sam.layers 4993ms
[time] sam.neck_downsample 74ms
[time] tile 3,2 sam 5096ms clip 686ms proj 104ms
[time] gundam_tiles 70425ms
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
[uocr2-fast] prefill done S=1509 43417.7 ms
[uocr2-fast] decode done: 138 tokens, 71863.3 ms total
```

### lowmem (table) stage timing
```
[time] tile 0,0 sam 11113ms clip 1925ms proj 104ms
[time] sam.patch_embed 29ms
[time] sam.layers 10936ms
[time] sam.neck_downsample 74ms
[time] tile 1,0 sam 11040ms clip 1940ms proj 104ms
[time] sam.patch_embed 29ms
[time] sam.layers 11101ms
[time] sam.neck_downsample 76ms
[time] tile 2,0 sam 11208ms clip 1944ms proj 104ms
[time] sam.patch_embed 31ms
[time] sam.layers 11172ms
[time] sam.neck_downsample 76ms
[time] tile 3,0 sam 11279ms clip 1965ms proj 102ms
[time] sam.patch_embed 30ms
[time] sam.layers 11052ms
[time] sam.neck_downsample 77ms
[time] tile 0,1 sam 11161ms clip 1948ms proj 114ms
[time] sam.patch_embed 31ms
[time] sam.layers 11064ms
[time] sam.neck_downsample 73ms
[time] tile 1,1 sam 11169ms clip 1925ms proj 101ms
[time] sam.patch_embed 30ms
[time] sam.layers 11161ms
[time] sam.neck_downsample 76ms
[time] tile 2,1 sam 11268ms clip 1953ms proj 101ms
[time] sam.patch_embed 31ms
[time] sam.layers 11054ms
[time] sam.neck_downsample 75ms
[time] tile 3,1 sam 11160ms clip 1916ms proj 101ms
[time] sam.patch_embed 29ms
[time] sam.layers 11012ms
[time] sam.neck_downsample 74ms
[time] tile 0,2 sam 11117ms clip 1930ms proj 102ms
[time] sam.patch_embed 29ms
[time] sam.layers 10949ms
[time] sam.neck_downsample 74ms
[time] tile 1,2 sam 11053ms clip 1911ms proj 101ms
[time] sam.patch_embed 30ms
[time] sam.layers 11615ms
[time] sam.neck_downsample 105ms
[time] tile 2,2 sam 11751ms clip 2555ms proj 191ms
[time] sam.patch_embed 57ms
[time] sam.layers 14890ms
[time] sam.neck_downsample 73ms
[time] tile 3,2 sam 15021ms clip 1981ms proj 101ms
[time] gundam_tiles 163811ms
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
[uocr2-fast] prefill done S=1509 43403.4 ms
[uocr2-fast] decode done: 138 tokens, 74480.4 ms total
```
