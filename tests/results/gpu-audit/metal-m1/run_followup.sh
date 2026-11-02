#!/bin/bash
# Follow-ups after the first OCR batch (2026-11-01). One lock per model run.
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
I=/Volumes/backups/code/crispembed-metal-audit-20261101/tests/regression/images
H=$(cd "$(dirname "$0")" && pwd); P=~/miniconda3/bin/python; O=$H/ocr
LOCK=/Volumes/backups/code/model-cache-20261101/m1-gpu-lock.sh
ab() { local out=$1; shift; $LOCK $P $H/ab_cmd.py $O/$out "$@" || echo "FAIL $out"; }
# ESRGAN: new per-kind default (GPU on Metal, no env) vs --gpu-backend cpu
ab esrgan-default.fox.json -- $B -t 4 --esrgan-model $M/esrgan-x4-f32.gguf --esrgan-sr $I/fox.png
ab esrgan-default.scan_strip.json -- $B -t 4 --esrgan-model $M/esrgan-x4-f32.gguf --esrgan-sr $I/scan_strip.png
# DBNet: default lane (det CPU, TrOCR Metal) on the larger page, to isolate the detector cost
ab dbnet-trocr.scan_page_pd.json -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/scan_page_pd.png
ab dbnet-gpu-deconvmm-trocr.scan_page_pd.json --metal-env OCR_DETECT_USE_GPU=1 CRISPEMBED_DBNET_DECONV_MATMUL=1 -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/scan_page_pd.png
