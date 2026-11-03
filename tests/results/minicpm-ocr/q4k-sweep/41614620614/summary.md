## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=unset
    commit=fbf6772f0d7611c9fc36b370bdf38f9e4cdeebfe
    cpu=AMD EPYC 9V45 96-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| mix2 | 970.8 | 0.9671 (para-en) | 4/5 | para-en@54 |
| mix4 | 991.8 | 0.9357 (para-en) | 4/5 | para-en@54 |
| q4k | 1205.6 | 0.9724 (para-en) | 4/5 | para-en@54 |
| q8 | 1455.2 | 0.9998 (para-en) | 5/5 | - |
