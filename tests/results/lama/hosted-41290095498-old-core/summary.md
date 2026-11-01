### Big-LaMa quantization research lane (a27dc8d) — internal only
```
build=pass (208s)
native-units=pass (0s)
py-units=pass (2s)
lama-deps=pass (29s)
lama-fetch=pass (6s)
lama-convert=pass (4s)
quantize-f16=pass (1s)
quantize-q8_0=pass (0s)
quantize-q4_k=pass (3s)
lama-reference=pass (15s)
f32-strict=fail(1) [informational]
fixtures-f32-matmul=pass (41s)
fixtures-f32-lowmem=pass (41s)
fixtures-f16-storage=pass (38s)
fixtures-f16-matmul=fail(1) (3s)
fixtures-f16-lowmem=fail(1) (2s)
fixtures-q8_0-storage=pass (37s)
fixtures-q8_0-matmul=pass (39s)
fixtures-q8_0-lowmem=pass (38s)
fixtures-q4_k-storage=pass (36s)
fixtures-q4_k-matmul=pass (37s)
fixtures-q4_k-lowmem=pass (37s)
live-f32-storage=pass (199s)
live-f32-matmul=pass (198s)
live-f32-lowmem=pass (208s)
live-f16-storage=pass (197s)
live-f16-matmul=fail(1) (3s)
rss-f16-matmul=fail
live-f16-lowmem=fail(1) (2s)
rss-f16-lowmem=fail
live-q8_0-storage=pass (196s)
live-q8_0-matmul=pass (196s)
live-q8_0-lowmem=pass (206s)
live-q4_k-storage=pass (197s)
live-q4_k-matmul=pass (198s)
live-q4_k-lowmem=pass (208s)
live-original=pass (58s)
live-compare=pass (5s)

102545824 big-lama-f16.gguf
204290336 big-lama-f32.gguf
29413312 big-lama-q4_k.gguf
54849472 big-lama-q8_0.gguf
```
