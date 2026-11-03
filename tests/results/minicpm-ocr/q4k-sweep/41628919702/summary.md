## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=portable
    commit=f5484251abe0dab1543053bf7a7335c7ff1c760d
    cpu=AMD EPYC 7763 64-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| mix2 | 970.8 | 0.9671 (para-en) | 4/5 | para-en@54 |
| pol-min | 902.4 | 0.9489 (para-en) | 4/5 | para-en@54 |
| pol-v1 | 970.8 | 0.9671 (para-en) | 4/5 | para-en@54 |
