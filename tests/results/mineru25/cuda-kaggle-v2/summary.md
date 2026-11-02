| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | gpu | forced | 8/8 | - | 24442 | 24442 | 2769 |  |  |
| f16 | gpu | default | 8/8 | - | 24946 | 24946 | 2769 | 0.999946 | german prefill_final_norm(row) 0.864709 |
| f32 | cpu | forced | 8/8 | - | 629246 | 629246 | 9025 |  |  |
| q4_k-df16 | gpu | forced | 8/8 | - | 21125 | 21125 | 2009 |  |  |
| q4_k-df16 | gpu | default | 7/8 | layout-page@180 | 20334 | 20334 | 2009 | 0.998766 | german prefill_final_norm(row) 0.282020 |
| q4_k-dq8 | gpu | forced | 8/8 | - | 15248 | 15248 | 1567 |  |  |
| q4_k-dq8 | gpu | default | 7/8 | layout-page@180 | 15217 | 15217 | 1567 | 0.998809 | german prefill_final_norm(row) 0.280315 |
| q4_k | gpu | forced | 8/8 | - | 14305 | 14305 | 1397 |  |  |
| q4_k | gpu | default | 6/8 | german@3, arabic@4 | 14341 | 14341 | 1397 | 0.923131 | german prefill_final_norm(row) 0.281793 |
| q4_k-vf16 | gpu | forced | 8/8 | - | 21011 | 21011 | 1959 |  |  |
| q4_k-vf16 | gpu | default | 5/8 | german@3, arabic@4, layout-page@38 | 20426 | 20426 | 1959 | 0.939804 | german prefill_final_norm(row) 0.840991 |
| q8_0 | gpu | forced | 8/8 | - | 20615 | 20615 | 2130 |  |  |
| q8_0 | gpu | default | 8/8 | - | 20227 | 20227 | 2130 | 0.999314 | german prefill_final_norm(row) 0.857525 |
