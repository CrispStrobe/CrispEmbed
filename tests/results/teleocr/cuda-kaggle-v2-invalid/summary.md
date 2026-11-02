| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 839089 | 839089 | 5007 |  |  |
| f16 | gpu | forced | 6/6 | - | 37213 | 37213 | 2696 |  |  |
| f16 | gpu | default | 6/6 | - | 35869 | 35869 | 2696 | 1.000000 | code vis_merger_output(row) 1.000000 |
| q4_k-attn8 | gpu | forced | 6/6 | - | 34424 | 34424 | 2177 |  |  |
| q4_k-attn8 | gpu | default | 5/6 | code@364 | 33225 | 33225 | 2177 |  |  |
| q4_k | gpu | forced | 6/6 | - | 31166 | 31166 | 2093 |  |  |
| q4_k | gpu | default | 5/6 | code@364 | 31902 | 31902 | 2093 |  |  |
| q4_k-m | gpu | forced | 6/6 | - | 31957 | 31957 | 2122 |  |  |
| q4_k-m | gpu | default | 6/6 | - | 34218 | 34218 | 2122 |  |  |
| q8_0 | gpu | forced | 6/6 | - | 35999 | 35999 | 2303 |  |  |
| q8_0 | gpu | default | 6/6 | - | 33255 | 33255 | 2303 |  |  |
