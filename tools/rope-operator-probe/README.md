# Retained-input RoPE operator proof

The [actual operator result](../../tests/results/qwen35-ocr/lighton4b-same-input-rope-operators.json)
closes all sixteen Q/K comparisons across native and pinned upstream ggml,
using the captured 4/252/6/4 rows and normalized vectors. Every output is
byte-identical to actual source post-RoPE. The fixed IMROPE profile uses 64
rotated dimensions, sections `[11,11,10,0]`, theta 10,000,000 and no scaling or
extrapolation. Inactive YARN preset fields are not independently observed metadata.

Direct observation inside the unchanged native attention function separately
preserves all 22 block-stage/cache files and the complete 266-row layer3 output.
Actual native post-rotation Q/K match the earlier reconstructed vectors exactly.
Thus the scalar/native versus ggml/source RoPE arithmetic boundary is measured
on equal normalized inputs and positions. The source uses iterative frequencies;
the old native scalar path recomputes a power per lane.

The [first setup failure](../../tests/results/qwen35-ocr/lighton4b-rope-operators-initial-setup-negative.json)
stopped at a launcher working-directory error before transfers or operators.
The [second retained result](../../tests/results/qwen35-ocr/lighton4b-direct-native-rope-lookup-negative.json)
contains successful direct native observation, followed by a source callback-name
lookup failure before ggml invocation. The corrected reader reuses all successful
native data through a closed remote transfer and performs no native, source or
model recapture. Mutable outer log snapshots are distinguished from final logs.

A separate default-off physical-profile candidate can now use the standard ggml
RoPE operation. Context/cache mode identity, unsupported backend rejection,
component proof, observed regressions and the actual fixed decoded panel remain
required. This operator result does not qualify decoded quality or alter any
runtime, vendor trait, default, release or app pin.
