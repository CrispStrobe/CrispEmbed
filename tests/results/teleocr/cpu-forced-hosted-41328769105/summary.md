| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | final-output cosine (logits) | worst stage |
|---|---|---|---|---|---|---|---|---|---|
| f16 | cpu | forced | 6/6 | - | 645033 | 645033 | 4857 |  |  |
| q4_k-attn8 | cpu | forced | 6/6 | - | 528399 | 528399 | 3817 |  |  |
| q4_k-attn8 | cpu | default | 4/6 | table@35, code@364 | 535779 | 535779 | 3817 | 0.852627 | code prefill_final_norm(row) 0.555446 |
| q4_k | cpu | forced | 6/6 | - | 536348 | 536348 | 3649 |  |  |
| q4_k | cpu | default | 5/6 | code@364 | 540043 | 540043 | 3649 | 0.815384 | code prefill_final_norm(row) 0.553913 |
