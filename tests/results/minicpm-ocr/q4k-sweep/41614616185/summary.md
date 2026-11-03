## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=unset
    commit=fbf6772f0d7611c9fc36b370bdf38f9e4cdeebfe
    cpu=AMD EPYC 7763 64-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| mix1 | 902.4 | 0.9489 (para-en) | 4/5 | para-en@54 |
| mix3 | 923.4 | 0.9075 (para-en) | 4/5 | para-en@54 |
| mix5 | 1033.3 | 0.9706 (para-en) | 4/5 | para-en@54 |
