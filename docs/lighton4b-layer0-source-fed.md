# Retained-input LightOn4B layer0 attribution

The retained stage comparison first differs in shared vision arithmetic. Both
native recipes use identical decoder input but different quantized layer0
matrices. Their first recipe-specific layer0 difference alone cannot distinguish
quantization from a decoder implementation error.

The test-only `tools/ci-heavy/lighton4b_layer0_source_fed.py` uses the exact
original154-token decoder input. It requires original BF16 layer0 hidden output
to reproduce its retained anchor exactly, then executes diagnostic arms:

- Official BF16-derived weights in original/native F32 operators, with identical
  input. This is an operator diagnostic, not replacement reference gold.
- Each frozen Q8/Q4 candidate's independently dequantized layer0 weights in the
  original F32 layer, compared against actual native quantized execution.
- The same dequantized weights in native F32 execution, separating quantized
  matmul activation/kernel effects from the remaining decoder operators.

The original executes one full154-token layer. Native uses the existing
sixteen-token chunk/cache path, matching production's prefill schedule. Every
arm retains input norm, attention output/residual, post norm, gate/up/activation,
down/hidden and final convolution/recurrent cache arrays. Reports include
original/native absolute magnitudes, maximum absolute error, row/global cosine,
unchanged strict thresholds and earliest failing stage. Ordinary original output
must match hook-observed output. Protected original BF16 backend settings are
separate from the audited F32 component setup.

The official checkpoint is fetched only as its contiguous226MB layer0 interval
from an immutable public revision. The frozen header, interval size and HTTP
Content-Range are verified; interval and per-tensor hashes are retained. This
does not rehash the entire9GB checkpoint. Exact frozen BF16 hidden reproduction
is a required anchor, and failures stop all later attribution. Each quantized
candidate is fetched and fully SHA-256 authenticated in turn; large models and
raw arrays remain hosted artifacts. No generation/image forward/full decoder or
new gold is produced; no decoded-quality, publication or release claim follows.

Hosted execution is pending. Production `src/`, ggml and VERSION are unchanged.
