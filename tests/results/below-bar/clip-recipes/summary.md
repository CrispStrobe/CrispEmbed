### below-bar recipes clip (f40c6a4d)

| arm | MB | dominant | bar | score | embed orig min/mean |
|---|---:|---|---:|---:|---|
| source-f32 | 243.8 | f32 | 0.99 | 0.99890 | 0.99890/0.99999 |
| source-f32 CRISPEMBED_CLIP_TEXT_KEEP_EOS=0 (old truncation) | 243.8 | f32 | 0.99 | 0.56845 | 0.56845/0.97536 |
| q4_k+im | 48.2 | q4_k | 0.94 | 0.97816 | 0.97816/0.99223 |
| q4_k+im; enc.0 q8_0 | 49.7 | q4_k | 0.94 | 0.98524 | 0.98524/0.99480 |
| q4_k+im; enc.0,attn.v q8_0 | 51.1 | q4_k | 0.94 | 0.98779 | 0.98779/0.99533 |
| q4_k+im; enc.0,attn.v,attn.o,text_proj q8_0 | 52.6 | q8_0 | 0.97 | 0.99196 | 0.99196/0.99703 |
