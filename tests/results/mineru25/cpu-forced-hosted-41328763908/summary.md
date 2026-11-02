| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 8/8 | - | 913995 | 913995 | 4987 |  |  |
| f16 | cpu | default | 8/8 | - | 914803 | 914803 | 4987 | 1.000000 | german prefill_final_norm(row) 1.000000 |
| f32 | cpu | forced | 8/8 | - | 508035 | 508035 | 8875 |  |  |
| q4_k | cpu | forced | 8/8 | - | 465946 | 465946 | 2241 |  |  |
| q4_k | cpu | default | 6/8 | german@3, arabic@4 | 470183 | 470183 | 2241 | 0.914481 | german prefill_final_norm(row) 0.288843 |
| q4_k-df16 | cpu | forced | 8/8 | - | 532519 | 532519 | 3466 |  |  |
| q4_k-df16 | cpu | default | 7/8 | layout-page@180 | 534623 | 534623 | 3466 | 0.999372 | german prefill_final_norm(row) 0.290208 |
| q4_k-dq8 | cpu | forced | 8/8 | - | 464511 | 464511 | 2583 |  |  |
| q4_k-dq8 | cpu | default | 8/8 | - | 468303 | 468303 | 2583 | 0.998114 | german prefill_final_norm(row) 0.289641 |
| q4_k-vf16 | cpu | forced | 8/8 | - | 831571 | 831571 | 3366 |  |  |
| q4_k-vf16 | cpu | default | 6/8 | german@3, arabic@4 | 835306 | 835306 | 3366 | 0.925344 | layout-rot90 prefill_final_norm(row) 0.960672 |
| q8_0 | cpu | forced | 8/8 | - | 832138 | 832138 | 3707 |  |  |
| q8_0 | cpu | default | 8/8 | - | 836983 | 836983 | 3708 | 0.998803 | layout-rot90 prefill_final_norm(row) 0.998417 |
