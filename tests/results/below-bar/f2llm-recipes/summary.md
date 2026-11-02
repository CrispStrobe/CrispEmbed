### below-bar recipes f2llm (f40c6a4d)

| arm | MB | dominant | bar | score | embed orig min/mean |
|---|---:|---|---:|---:|---|
| source-f32 | 2279.4 | f32 | 0.99 | 1.00000 | 1.00000/1.00000 |
| A q4_k+im; ffn_down,attn_v,blk.26,blk.27 q8_0 | 466.8 | q4_k | 0.94 | 0.96077 | 0.96077/0.97429 |
| B q4_k+im; ffn_down q6_k; attn_v,blk.26,blk.27 q8_0 | 447.9 | q4_k | 0.94 | 0.96117 | 0.96117/0.97227 |
| C q4_k+im; ffn_down,attn_v q8_0; blk.27 f16 | 475.4 | q4_k | 0.94 | 0.95537 | 0.95537/0.96687 |
| D q4_k+im; ffn_down,attn_v q6_k; blk.26,blk.27 q8_0 | 441.6 | q4_k | 0.94 | 0.95544 | 0.95544/0.96912 |
| E q4_k+im; ffn_down,attn_v,blk.24-27 q8_0 | 477.8 | q8_0 | 0.97 | 0.96785 | 0.96785/0.98007 |
| F q4_k+im; ffn_down q5_k; attn_v,blk.26,blk.27 q8_0 | 437.6 | q4_k | 0.94 | 0.95802 | 0.95802/0.97210 |
| G q4_k+im; ffn_down q6_k; attn_v q8_0; blk.25-27 q8_0 | 454.2 | q4_k | 0.94 | 0.96441 | 0.96441/0.97619 |
