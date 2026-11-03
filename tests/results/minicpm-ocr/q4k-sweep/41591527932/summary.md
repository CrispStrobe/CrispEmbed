## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=unset
    commit=828aa38ab8176036a31824291ed8cfc4537159ac
    cpu=AMD EPYC 9V74 80-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| f16 | 2491.7 | 1.0000 (para-en) | 5/5 | - |
| f32 | - | failed: forced run | - | - |
| q4k-old | 1809.5 | 0.9690 (para-en) | 5/5 | - |
| q4k | 1205.6 | 0.9724 (para-en) | 4/5 | para-en@54 |
| q8 | 1455.2 | 0.9998 (para-en) | 5/5 | - |
