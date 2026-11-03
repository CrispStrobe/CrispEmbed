## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=unset
    commit=828aa38ab8176036a31824291ed8cfc4537159ac
    cpu=AMD EPYC 7763 64-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| dec4-ffn | 1329.2 | 0.9790 (para-en) | 4/5 | para-en@60 |
| dec4-ffndown | 1413.2 | 0.9917 (para-en) | 5/5 | - |
| dec4-full | 1434.2 | 0.9775 (para-en) | 5/5 | - |
| dec4-lin | 1365.0 | 0.9958 (para-en) | 4/5 | para-en@54 |
| dec4 | 1218.0 | 0.9683 (para-en) | 5/5 | - |
