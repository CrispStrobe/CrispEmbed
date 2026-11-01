| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 865787 | 865787 | 5007 |  |  |
| f16 | gpu | forced | 6/6 | - | 35015 | 35015 | 2696 |  |  |
| f16 | gpu | default | 6/6 | - | 35955 | 35955 | 2696 | 0.999883 | code prefill_final_norm(row) 0.895186 |
| q4_k-attn8 | gpu | forced | 6/6 | - | 32791 | 32791 | 2177 |  |  |
| q4_k-attn8 | gpu | default | 5/6 | code@364 | 33382 | 33382 | 2176 | 0.843844 | text prefill_final_norm(row) 0.551769 |
| q4_k | gpu | forced | 6/6 | - | 35060 | 35060 | 2093 |  |  |
| q4_k | gpu | default | 5/6 | code@364 | 32126 | 32126 | 2093 | 0.873399 | synthetic prefill_final_norm(row) 0.550061 |
| q4_k-m | gpu | forced | 6/6 | - | 32480 | 32480 | 2121 |  |  |
| q4_k-m | gpu | default | 6/6 | - | 33586 | 33586 | 2121 | 0.921535 | code prefill_final_norm(row) 0.876486 |
| q8_0 | gpu | forced | 6/6 | - | 32427 | 32427 | 2303 |  |  |
| q8_0 | gpu | default | 6/6 | - | 33084 | 33084 | 2303 | 0.993890 | code prefill_final_norm(row) 0.893852 |
