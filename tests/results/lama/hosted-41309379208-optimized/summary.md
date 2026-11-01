### Big-LaMa lama-v2 quantization research (fixtures, fa7900a) — internal only
```
deps=pass (35s)
strict-f32=fail(1) [informational] (383s)
build-quantize=pass (2s)
quantize-f16=pass (0s)
quantize-q8_0=pass (1s)
quantize-q4_k=pass (4s)
quantize-q4_k-s8e8=pass (4s)
fixtures-f32-opt=pass (57s)
fixtures-f16-opt=pass (55s)
fixtures-f16-opt-matmul=pass (86s)
fixtures-q8_0-opt=pass (52s)
fixtures-q8_0-opt-matmul=pass (51s)
fixtures-q4_k-opt=fail(1) (51s)
fixtures-q4_k-opt-matmul=fail(1) (51s)
fixtures-q4_k-s8e8-opt=pass (53s)
fixtures-q4_k-s8e8-opt-matmul=pass (51s)

102554560 big-lama-f16.gguf
204299072 big-lama-f32.gguf
35504608 big-lama-q4_k-s8e8.gguf
29422048 big-lama-q4_k.gguf
54858208 big-lama-q8_0.gguf
```
