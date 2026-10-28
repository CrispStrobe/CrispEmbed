# Unlimited-OCR-2 trial-fix verification

Commit `8ca9d3de2a8160e6f5235156d3550c9719583744`, Intel(R) Xeon(R) CPU @ 2.20GHz x4 (avx512f=False), build 253.2s. Pages re-rendered; sha256 == trial: {'paragraphs': True, 'table': True, 'receipt': True, 'german': True, 'two-column': True, 'a4-dense': True, 'scan-sim': True}.

## Checks

- **unit_tests_pass**: True
- **f16_default_ids_equal_original_all4**: True
- **f16_default_text_bytes_equal_original_all4**: True
- **q8_default_ids_equal_original**: {'table': False, 'receipt': False, 'paragraphs': True, 'german': True}
- **q8_text_bytes_equal_when_ids_equal**: True
- **before_after_ids_equal**: {'table': False, 'receipt': False, 'paragraphs': False, 'german': False}
- **ablation_ids_equal_after**: {}
- **vision_bytes_equal_cache_on_off**: {}
- **exact_decoder_default_ids_equal_original**: []
- **native_build_ids_equal_original**: {}

## Runs

| tag | model | page | exit | wall s | RSS MB | ids | ids==orig | first diff | text==orig |
|---|---|---|---|---|---|---|---|---|---|
| after | unlimited-ocr2 | table | 0 | 281.11 | 12820.9 | 138 | True | None | True |
| after | unlimited-ocr2 | receipt | 0 | 389.79 | 12820.4 | 306 | True | None | True |
| after | unlimited-ocr2 | paragraphs | 0 | 291.01 | 12820.4 | 146 | True | None | True |
| after | unlimited-ocr2 | german | 0 | 278.28 | 12820.2 | 164 | True | None | True |
| after | unlimited-ocr2-q8 | table | 0 | 179.62 | 6868.2 | 138 | False | 7 | False |
| after | unlimited-ocr2-q8 | receipt | 0 | 236.25 | 6868.0 | 306 | False | 131 | False |
| after | unlimited-ocr2-q8 | paragraphs | 0 | 159.74 | 6868.2 | 146 | True | None | True |
| after | unlimited-ocr2-q8 | german | 0 | 176.61 | 6868.0 | 164 | True | None | True |
| clean | unlimited-ocr2 | table | 0 | 281.79 | 12820.4 | 138 | True | None | False |
| legacy-text | unlimited-ocr2 | table | 0 | 282.02 | 12820.2 | 138 | True | None | False |

## Speed (F16, fast decoder, tiled): before -> after

| page | before s | after s | speedup |
|---|---|---|---|

## Stage timing (table, F16)

### after
```
[time] sam.patch_embed 35ms
[time] sam.layers 10785ms
[time] sam.neck_downsample 70ms
[time] sam.patch_embed 27ms
[time] sam.layers 10741ms
[time] sam.neck_downsample 70ms
[time] sam.patch_embed 27ms
[time] sam.layers 10757ms
[time] sam.neck_downsample 72ms
[time] sam.patch_embed 28ms
[time] sam.layers 10816ms
[time] sam.neck_downsample 72ms
[time] sam.patch_embed 27ms
[time] sam.layers 10777ms
[time] sam.neck_downsample 70ms
[time] sam.patch_embed 27ms
[time] sam.layers 10828ms
[time] sam.neck_downsample 73ms
[time] sam.patch_embed 27ms
[time] sam.layers 10842ms
[time] sam.neck_downsample 77ms
[time] sam.patch_embed 27ms
[time] sam.layers 10762ms
[time] sam.neck_downsample 72ms
[time] sam.patch_embed 28ms
[time] sam.layers 11000ms
[time] sam.neck_downsample 72ms
[time] sam.patch_embed 30ms
[time] sam.layers 10781ms
[time] sam.neck_downsample 71ms
[time] sam.patch_embed 28ms
[time] sam.layers 10782ms
[time] sam.neck_downsample 72ms
[time] gundam_tiles 155356ms
[uocr2-fast] prefill layer 0 done
[uocr2-fast] prefill layer 1 done
[uocr2-fast] prefill layer 2 done
[uocr2-fast] prefill layer 3 done
[uocr2-fast] prefill layer 4 done
[uocr2-fast] prefill layer 5 done
```
## `clean` rendering (table)
```
<table><tr><td>Product</td><td>Region</td><td>Units</td><td>Revenue</td></tr><tr><td>Notebook A5</td><td>North</td><td>1,250</td><td>8,750.00</td></tr><tr><td>Desk Lamp</td><td>South</td><td>430</td><td>12,040.50</td></tr><tr><td>Office Chair</td><td>East</td><td>96</td><td>17,184.00</td></tr><tr><td>Monitor 27&quot;</td><td>West</td><td>212</td><td>42,188.00</td></tr><tr><td>Total</td><td></td><td>1,988</td><td>80,162.50</td></tr></table>
```
## `legacy-text` rendering (table)
```
table[56, 99, 921, 658]<table>ProductRegionUnitsRevenueNotebook A5North1,2508,750.00Desk LampSouth43012,040.50Office ChairEast9617,184.00Monitor 27&quot;West21242,188.00Total1,98880,162.50</table>
```
