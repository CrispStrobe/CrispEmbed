#!/bin/bash
# OCR / detector / recognizer / SR Metal-vs-CPU batch (2026-11-01). Run under m1-gpu-lock.sh.
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
I=/Volumes/backups/code/crispembed-metal-audit-20261101/tests/regression/images
H=$(cd "$(dirname "$0")" && pwd); P=~/miniconda3/bin/python; O=$H/ocr; mkdir -p $O
ab() { local out=$1; shift; $P $H/ab_cmd.py $O/$out "$@" || echo "FAIL $out"; }
# 1. default document pipeline lane: DBNet (CPU on Metal boxes) + TrOCR (GPU)
for img in fox scan_strip; do
  ab dbnet-trocr.$img.json -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/$img.png
done
# 1b. DBNet detector forced onto Metal (re-measure issue #25 verdict)
ab dbnet-gpu-trocr.fox.json --metal-env OCR_DETECT_USE_GPU=1 CRISPEMBED_OCR_DETECT_BENCH=1 --cpu-env CRISPEMBED_OCR_DETECT_BENCH=1 -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/fox.png
# 1c. DBNet on Metal with the gated mul_mat deconv (CRISPEMBED_DBNET_DECONV_MATMUL)
ab dbnet-gpu-deconvmm-trocr.fox.json --metal-env OCR_DETECT_USE_GPU=1 CRISPEMBED_DBNET_DECONV_MATMUL=1 CRISPEMBED_OCR_DETECT_BENCH=1 --cpu-env CRISPEMBED_OCR_DETECT_BENCH=1 -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/fox.png
ab dbnet-gpu-deconvmm-trocr.scan_page_pd.json --metal-env OCR_DETECT_USE_GPU=1 CRISPEMBED_DBNET_DECONV_MATMUL=1 CRISPEMBED_OCR_DETECT_BENCH=1 --cpu-env CRISPEMBED_OCR_DETECT_BENCH=1 -- $B -t 4 --json -m $M/trocr-small-printed-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/trocr-small-printed-q8_0.gguf --ocr $I/scan_page_pd.png
# 1d. same rewrite on the CPU detector (both arms CPU det + Tesseract, so only the deconv form differs)
ab dbnet-cpu-deconvmm-tesseract.scan_page_pd.json --allow-cpu-metal --metal-env CRISPEMBED_DBNET_DECONV_MATMUL=1 CRISPEMBED_OCR_DETECT_BENCH=1 --cpu-env CRISPEMBED_OCR_DETECT_BENCH=1 -- $B -t 4 --json -m $M/tesseract-eng-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/tesseract-eng-q8_0.gguf --ocr $I/scan_page_pd.png
# 2. PP-OCRv6 small det + small rec (rec GPU default, det CPU on Metal)
for img in fox scan_page_pd; do
  ab ppocrv6-small.$img.json -- $B -t 4 --json -m $M/PP-OCRv6_small_rec-q8-head.gguf --ocr-det $M/PP-OCRv6_small_det-f16.gguf --ocr-rec $M/PP-OCRv6_small_rec-q8-head.gguf --ocr $I/$img.png
done
ab ppocrv6-small-detgpu.scan_page_pd.json --metal-env CRISPEMBED_PPOCRV6_DET_GPU=1 -- $B -t 4 --json -m $M/PP-OCRv6_small_rec-q8-head.gguf --ocr-det $M/PP-OCRv6_small_det-f16.gguf --ocr-rec $M/PP-OCRv6_small_rec-q8-head.gguf --ocr $I/scan_page_pd.png
# 3. EasyOCR CRNN (GPU default) behind DBNet
ab dbnet-easyocr.fox.json -- $B -t 4 --json -m $M/easyocr-english-g2-f16.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/easyocr-english-g2-f16.gguf --ocr $I/fox.png
# 4. Tesseract (no GPU code; checks that neither arm initialises Metal)
ab dbnet-tesseract.fox.json --allow-cpu-metal -- $B -t 4 --json -m $M/tesseract-eng-q8_0.gguf --ocr-det $M/dbnet-ic15-q8_0.gguf --ocr-rec $M/tesseract-eng-q8_0.gguf --ocr $I/fox.png
# 5. PARSeq line recognizer
ab parseq.word.json -- $B -t 4 -m $M/parseq-q8_0.gguf --ocr /Volumes/backups/ai/crispembed-gguf/word.png
ab parseq-tiny.hello.json -- $B -t 4 -m $M/parseq-tiny-q8_0.gguf --ocr /Volumes/backups/ai/crispembed-gguf/ppocrv6-hello-320x48.png
# 6. TexTeller (math_ocr, GPU default)
# (done) ab texteller.formula.json -- $B -t 4 -m $M/texteller-3-q8_0.gguf --ocr $I/formula_quadratic.png
# 7. ESRGAN gated GPU path vs default CPU
ab esrgan-gpu.fox.json --metal-env CRISPEMBED_ESRGAN_GPU=1 -- $B -t 4 --esrgan-model $M/esrgan-x4-f32.gguf --esrgan-sr $I/scan_strip.png
# 8. InstructIR gated GPU path vs default CPU
ab instructir-gpu.scan_strip.json --metal-env CRISPEMBED_INSTRUCTIR_GPU=1 -- $B -t 4 --instructir-model $M/instructir-f16.gguf --instructir $I/scan_strip.png
