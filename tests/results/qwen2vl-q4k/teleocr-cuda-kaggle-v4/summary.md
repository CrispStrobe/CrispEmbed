| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 842421 | 842421 | 5007 |  |  |
| r1 | gpu | forced | 6/6 | - | 26464 | 26464 | 1514 |  |  |
| r1 | gpu | default | 5/6 | table@35 | 26698 | 26698 | 1514 | 0.963945 | code prefill_final_norm(row) 0.347633 |
| r9 | gpu | forced | 6/6 | - | 24421 | 24421 | 1428 |  |  |
| r9 | gpu | default | 5/6 | table@35 | 25846 | 25846 | 1427 | 0.956991 | code prefill_final_norm(row) 0.435228 |
