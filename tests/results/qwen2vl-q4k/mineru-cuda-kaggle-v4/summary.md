| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f32 | cpu | forced | 8/8 | - | 622563 | 622563 | 9024 |  |  |
| r2 | gpu | forced | 8/8 | - | 30896 | 30896 | 1032 |  |  |
| r2 | gpu | default | 6/8 | german@7, layout-page@48 | 30280 | 30280 | 1032 | 0.961082 | german prefill_final_norm(row) 0.103501 |
| r4 | gpu | forced | 8/8 | - | 31286 | 31286 | 1053 |  |  |
| r4 | gpu | default | 6/8 | german@7, layout-page@48 | 30328 | 30328 | 1053 | 0.972382 | german prefill_final_norm(row) 0.102419 |
