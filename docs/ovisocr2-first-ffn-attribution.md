# OvisOCR2 first-FFN attribution protocol

This bounded audit investigates the rejected attention-F16/FFN-Q4 candidate
from [40172527229](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40172527229).
It does not select a new recipe, regenerate original targets or change quality
gates. Published F16 and all four rejection receipts stay immutable.

The pinned Transformers5.16.1 `Qwen3_5MLP` executes
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The preceding original RMSNorm
executes F32 squared-mean/rsqrt, then multiplies by `1 + weight`. The audit uses
these actual modules and forward hooks, requiring observed/unobserved output
bits to match. It loads only first-layer weights from the original pinned
safetensors checkpoint and reconstructs the original post-attention residual
from frozen embeddings and attention output. The reconstructed original layer
must pass the unchanged0.002 absolute/0.99999 minimum cosine anchor gates;
bit drift and magnitudes remain reported separately.

The source-derived F16 gate/up/down payloads must hash-identically to the
retained prequantization tensor records. The independent Python `gguf0.19.0`
Q4_K decoder supplies candidate matrices to the same original MLP. Every native
dequantized value must match these independent values bit-for-bit. Both packed
weight and decoded-array hashes, block counts, norms and errors are retained.
The production path materializes weights as F32 before F32 matmul; it does not
use quantized activations or Q4 vector-dot kernels. This protocol therefore
audits that actual execution path rather than assuming activation quantization.

`test-ovisocr2-ffn` compiles the actual decoder implementation into a test-only
translation unit with projection calls observed. No reference probe enters a
production object. It loads only layer0 and the global norm; a zero test head
satisfies the outer decoder interface. Vocabulary logits from that head have
no quality meaning and are not compared. Natural instrumented and ordinary
layer output and both linear-attention caches must be bit-identical.

The frozen acceptance gates are0.002 maximum absolute error and0.99999 minimum
row cosine, with all values finite and shapes/counts exact. The arms are:

- **Natural:** retained candidate-native embeddings and original native layer0
  output; reproduce the rejected boundary. Original full-precision and quantized
  local-target differences remain diagnostic, including any upstream drift.
- **Source input:** original embeddings for the real first layer, and bit-exact
  original Python normalized FFN input supplied at the real gate/up projections.
  Compare gate/up/product/down and the residual layer output with the independent
  quantized-weight reference. This deliberately isolates the FFN; it is not an
  unmodified composed forward.
- **Source gate/up:** inject independent gate/up results into the actual decoder
  activation/product expression, then compare product and ordinary down output.
- **Source SiLU:** inject original gate and an all-ones up result; observe the
  actual decoder expression to isolate SiLU. This is a local operator probe.
- **Source product:** inject the independent product into the ordinary production
  down projection. This isolates the final matrix multiplication.

All same-weight arms and bit controls must pass. Intended Q4-versus-original
precision loss is reported separately and must not turn into new reference gold.
The small producer preserves every failure, source/runtime/environment hash,
per-stage absolute/RMS errors, magnitudes and cosine. Raw arrays and the new
local reference archive stay in remote artifacts. Existing eleven-case outputs,
IDs/EOS and receipt/two-column regressions are linked; this operator audit
cannot override their negative quality outcome or qualify publication.

Run the existing metadata workflow on the feature source with `phase=first-ffn`.
The reusable workflow builds portable CPU validation code and performs model
execution on hosted CI. Retrieve only `ovisocr2-ffn-reports-RUN` for review.
Before closing the lane, independently review its actual producer, log hashes,
raw coverage and initial failures. If a same-weight mismatch exists, identify
the earliest failing operator before proposing any gated runtime correction.
A positive result instead scopes a separately declared development/evaluation
precision protocol; it never authorizes fitting observed held-out documents.
