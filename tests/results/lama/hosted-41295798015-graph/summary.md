### Big-LaMa lama-v2 quantization research (live, 56af837) — internal only
```
deps=pass (30s)
strict-f32=pass (237s)
build-quantize=pass (2s)
quantize-f16=pass (0s)
quantize-q8_0=pass (1s)
quantize-q4_k=pass (3s)
quantize-q4_k-s8=pass (2s)
quantize-q4_k-s8e8=pass (3s)
live-f32-storage=pass (179s)
live-f32-stored=pass (176s)
live-f32-graph=pass (315s)
live-f16-storage=pass (174s)
live-f16-matmul=pass (202s)
live-f16-graph=pass (315s)
live-f16-graph-matmul=pass (340s)
live-q8_0-storage=pass (176s)
live-q8_0-matmul=pass (171s)
live-q8_0-graph=pass (315s)
live-q8_0-graph-matmul=pass (316s)
live-q4_k-storage=pass (177s)
live-q4_k-matmul=pass (172s)
live-q4_k-graph=pass (315s)
live-q4_k-graph-matmul=pass (315s)
live-q4_k-s8-storage=pass (178s)
live-q4_k-s8-graph-matmul=pass (329s)
live-q4_k-s8e8-storage=pass (183s)
live-q4_k-s8e8-graph-matmul=pass (323s)
live-original=pass (24s)
live-compare=pass (3s)

102554560 big-lama-f16.gguf
204299072 big-lama-f32.gguf
34730464 big-lama-q4_k-s8.gguf
35504608 big-lama-q4_k-s8e8.gguf
29422048 big-lama-q4_k.gguf
54858208 big-lama-q8_0.gguf
```
