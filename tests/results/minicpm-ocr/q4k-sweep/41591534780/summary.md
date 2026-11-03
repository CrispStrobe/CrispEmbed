## MiniCPM-OCR quantization sweep

    build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)
    heavy_cpu_build_mode=unset
    commit=828aa38ab8176036a31824291ed8cfc4537159ac
    cpu=AMD EPYC 9V45 96-Core Processor
    threads=4
    CRISPEMBED_NATIVE:BOOL=OFF
    GGML_NATIVE:BOOL=OFF

| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |
|---|---|---|---|---|
| emb4 | 1334.1 | 0.9939 (para-en) | 5/5 | - |
| emb6 | 1396.6 | 0.9994 (para-en) | 5/5 | - |
| merger4 | 1442.9 | 0.9980 (para-en) | 4/5 | para-en@54 |
| vis4-attn | 1386.9 | 0.9895 (para-en) | 5/5 | - |
| vis4-ffn | 1391.4 | 0.9936 (para-en) | 5/5 | - |
| vis4 | 1323.1 | 0.9749 (para-en) | 5/5 | - |
| vmerge4 | 1405.4 | 0.9998 (para-en) | 5/5 | - |
