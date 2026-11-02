#!/bin/bash
# Heron (rt_detr_v2) layout: CPU vs Metal arms; regions printed by test-layout-detect at threshold 0.3.
W=/Volumes/backups/code/crispembed-metal-oxr-20261101; M=/Volumes/backups/code/model-cache-20261101/oxr
OUT=$M/heron-ab; mkdir -p $OUT; cd $W
for p in receipt german scan table japanese; do
  /Volumes/backups/code/model-cache-20261101/m1-gpu-lock.sh bash -c "
    for arm in cpu metal metal-prec metal-prec-hpmm; do
      case \$arm in cpu) e='LAYOUT_DETECT_FORCE_CPU=1';; metal) e='X=1';; metal-prec) e='CRISPEMBED_LAYOUT_GPU_PREC_F32=1';;
        metal-prec-hpmm) e='CRISPEMBED_LAYOUT_GPU_PREC_F32=1 GGML_METAL_PREC_F32_MM=1';; esac
      for r in 1 2; do s=\$(date +%s.%N 2>/dev/null || python3 -c 'import time;print(time.time())')
        env \$e build-metal/test-layout-detect $M/heron/layout-heron-f32.gguf $M/ref/pages/$p.png 0.3 > $OUT/$p-\$arm.txt 2> $OUT/$p-\$arm.log
        echo \"$p \$arm run\$r \$(python3 -c \"import time;print(round(time.time()-\$s,3))\")\" >> $OUT/timing.txt
      done
    done"
done
echo HERONDONE
