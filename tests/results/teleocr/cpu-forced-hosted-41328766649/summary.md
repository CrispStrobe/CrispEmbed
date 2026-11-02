| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 634251 | 634251 | 4856 |  |  |
| q4_k-m | cpu | forced | 6/6 | - | 635007 | 635007 | 3707 |  |  |
| q4_k-m | cpu | default | 6/6 | - | 645521 | 645521 | 3707 | 0.904747 | code prefill_final_norm(row) 0.901205 |
| q8_0 | cpu | forced | 6/6 | - | 513665 | 513665 | 4069 |  |  |
| q8_0 | cpu | default | 6/6 | - | 519161 | 519161 | 4069 | 0.998452 | code prefill_final_norm(row) 0.990959 |
