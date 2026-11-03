| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 388451 | 388451 | 4857 |  |  |
| teleocr_q4k_v1 | cpu | forced | 6/6 | - | 415959 | 415959 | 2319 |  |  |
| teleocr_q4k_v1 | cpu | default | 5/6 | table@35 | 433536 | 433536 | 2319 | 0.967719 | code prefill_final_norm(row) 0.489743 |
