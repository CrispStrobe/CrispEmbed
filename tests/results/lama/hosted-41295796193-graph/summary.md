### Big-LaMa lama-v2 quantization research (fixtures, 56af837) — internal only
```
deps=pass (37s)
strict-f32=fail(1) [informational] (423s)
build-quantize=pass (1s)
quantize-f16=pass (0s)
quantize-q8_0=pass (1s)
quantize-q4_k=pass (4s)
quantize-q4_k-s8=pass (3s)
quantize-q4_k-s8e8=pass (3s)
fixtures-f32-stored=pass (82s)
fixtures-f32-graph=pass (129s)
fixtures-f16-storage=pass (79s)
fixtures-f16-matmul=pass (107s)
fixtures-f16-graph=pass (127s)
fixtures-f16-graph-matmul=pass (154s)
fixtures-q8_0-storage=pass (78s)
fixtures-q8_0-matmul=pass (76s)
fixtures-q8_0-graph=pass (125s)
fixtures-q8_0-graph-matmul=pass (122s)
fixtures-q4_k-storage=fail(1) (77s)
fixtures-q4_k-matmul=fail(1) (76s)
fixtures-q4_k-graph=fail(1) (123s)
fixtures-q4_k-graph-matmul=fail(1) (124s)
fixtures-q4_k-s8-storage=fail(1) (77s)
fixtures-q4_k-s8-graph-matmul=pass (124s)
fixtures-q4_k-s8e8-storage=pass (77s)
fixtures-q4_k-s8e8-graph-matmul=pass (123s)

102554560 big-lama-f16.gguf
204299072 big-lama-f32.gguf
34730464 big-lama-q4_k-s8.gguf
35504608 big-lama-q4_k-s8e8.gguf
29422048 big-lama-q4_k.gguf
54858208 big-lama-q8_0.gguf
```
