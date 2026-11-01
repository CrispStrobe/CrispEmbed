### Big-LaMa lama-v2 quantization research (live, 152a26e) — internal only
```
deps=pass (26s)
strict-f32=pass (195s)
build-quantize=pass (1s)
quantize-f16=pass (0s)
quantize-q8_0=pass (0s)
quantize-q4_k=pass (3s)
live-f32-storage=pass (149s)
live-f32-stored=pass (144s)
live-f16-storage=pass (144s)
live-f16-matmul=pass (164s)
live-q8_0-storage=pass (141s)
live-q8_0-matmul=pass (135s)
live-q4_k-storage=pass (141s)
live-q4_k-matmul=pass (140s)
live-original=pass (20s)
live-compare=pass (1s)

102554560 big-lama-f16.gguf
204299072 big-lama-f32.gguf
29422048 big-lama-q4_k.gguf
54858208 big-lama-q8_0.gguf
```
