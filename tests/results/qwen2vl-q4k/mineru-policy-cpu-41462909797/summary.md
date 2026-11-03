| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f32 | cpu | forced | 8/8 | - | 279772 | 279772 | 8875 |  |  |
| mineru_q4k_v1 | cpu | forced | 8/8 | - | 274677 | 274677 | 1509 |  |  |
| mineru_q4k_v1 | cpu | default | 6/8 | german@7, layout-page@52 | 280794 | 280794 | 1509 | 0.967280 | german prefill_final_norm(row) 0.082078 |
