# Authenticated book prefill stage comparison

Artifact-only execution144611129/version1 completed in an explicitly authorized
GPU session while CPU-session capacity was full. NumPy performed the comparison;
no model was loaded or inference repeated. The [complete safe numerical report](../tests/results/qwen35-ocr/lightonocr3-4b-book-prefill-stage-compare.json)
has SHA256 `14e0f58ea15be3cd5f39e232d92ee0a90caae59e7d4bc23425c878a8cf606209`.
Two CPU capacity rejections and a title-reservation409 remain in private operator
records. No unrelated job was canceled.

The report authenticates the neutral source405 producer and preserved incomplete
native plannerON replay. Exact266 embedding rows, three position axes and original
input seal match; the source's separate two-zero-token compatibility graph is
excluded explicitly. Physical head mapping, source output crops and live firstscore
closures are checked. Original native timeout remains incomplete; no decoded claim
follows. Text and token sequences are absent from the numerical report.

First-four-token layer0 input norm, QKV, Z, alpha/beta, attention projection,
residual, postnorm and FFN gate/up are bit-exact. The earliest recorded operator
difference is SwiGLU: RMS1.39921e-9/max5.96046e-8 versus source RMS0.0381853.
Down projection differs RMS6.37168e-9/max3.72529e-8. Scalar native SiLU/multiply
versus source ggml SwiGLU is therefore a concrete same-input arithmetic boundary,
without proving its unique contribution to the final output.

| Actual source image graph | Hidden RMS error | Max absolute error | Source RMS magnitude |
|---|---:|---:|---:|
| Layer0 |1.43308e-6|8.72016e-5|0.255373|
| Layer1 |0.000235987|0.00566673|0.258190|
| Layer2 |0.000922016|0.0333672|0.252153|
| Layer3 |0.00145016|0.0418098|0.279054|

Final selected hidden RMS error is0.0802045/max0.588835; final norm differs
0.187096/0.793867, and firstscore0.143886/0.749206 (source score RMS1.95162).
Source image batching uses252 rows while native components use at most16.
No matching native image-layer0 operator capture exists here, so the image hidden
residual cannot be uniquely attributed to the first-four-token SwiGLU difference
or to batching. Full recurrent caches are not compared without authenticated
matching layouts. The next bounded capture is the first image chunk's native
layer0 operators, retaining historical prefill/firstscore neutrality.
