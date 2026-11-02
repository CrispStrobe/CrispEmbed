#!/bin/bash
# Five-case Metal A/B of the qualifying OXR quantized recipes; one case+precision per GPU lock.
W=/Volumes/backups/code/crispembed-metal-oxr-20261101; M=/Volumes/backups/code/model-cache-20261101/oxr
LOCK=/Volumes/backups/code/model-cache-20261101/m1-gpu-lock.sh; OUT=$M/ab2
cd $W; mkdir -p $M/quant $OUT
export STAGES="vision.merge,visual.embeddings,llm.layer.27,llm.norm,llm.logits" REPEAT=1
runcases() { # model name
  for c in formula japanese table receipt german; do
    $LOCK env CASES=$c tools/metal/oxr_batch_ab.sh build-metal "$1" $M/ref/pages $OUT/$2 metal
  done
}
runcases $M/f16/oxr-1.0-f16.gguf f16
CRISPEMBED_QUANTIZE_OXR_LARGE=q8_0 nice build-metal/crispembed-quantize $M/f16/oxr-1.0-f16.gguf $M/quant/q8large.gguf q8_0 > $OUT/quantize-q8large.log 2>&1
runcases $M/quant/q8large.gguf q8large; rm -f $M/quant/q8large.gguf
CRISPEMBED_QUANTIZE_OXR_LARGE=mix nice build-metal/crispembed-quantize $M/f16/oxr-1.0-f16.gguf $M/quant/q4mix.gguf q8_0 > $OUT/quantize-q4mix.log 2>&1
runcases $M/quant/q4mix.gguf q4mix; rm -f $M/quant/q4mix.gguf
echo ALLDONE
