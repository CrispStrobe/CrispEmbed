### Big-LaMa lama-v2 quantization research (live, fa7900a) — internal only
```
deps=pass (37s)
strict-f32=pass (345s)
build-quantize=pass (2s)
quantize-f16=pass (1s)
quantize-q8_0=pass (0s)
quantize-q4_k=pass (4s)
quantize-q4_k-s8e8=pass (4s)
live-f32-opt=pass (94s)
live-f16-opt=pass (93s)
live-f16-opt-matmul=pass (120s)
live-q8_0-opt=pass (92s)
live-q8_0-opt-matmul=pass (81s)
live-q4_k-opt=pass (93s)
live-q4_k-opt-matmul=pass (91s)
live-q4_k-s8e8-opt=pass (92s)
live-q4_k-s8e8-opt-matmul=pass (92s)
live-original=pass (37s)
live-compare=pass (2s)

102554560 big-lama-f16.gguf
204299072 big-lama-f32.gguf
35504608 big-lama-q4_k-s8e8.gguf
29422048 big-lama-q4_k.gguf
54858208 big-lama-q8_0.gguf
```
