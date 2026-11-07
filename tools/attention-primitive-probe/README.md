# Source-fed attention operator probe (preparation)

This diagnostic consumes authenticated actual native layer2 output and positions
from the frozen physical form capture. It must reproduce native layer3 first16
hidden bytes before interpreting duplicate-helper Q/K/V. It does not capture
upstream hidden2 and cannot establish full-model or decoded quality parity.

The upstream reference is pinned to `6c01da6278ab6c69ed67d9124806b50d2b299ba7`.
Native ggml is `3a27b6c665b191cb230be367db5c52d81dd3e2ac`.
Both builds use LLAMAFILE/AVX2 on, NATIVE/AVX512/BLAS off and four threads.
No model forward, conversion, vocabulary head or generation is allowed.

Required comparisons on identical post-RoPE native Q/K and projected V:

- Legacy scalar F32 cache, with bit-exact actual native gated context closure.
- Scalar F16 K/V writes only; separately round Q and softmax probabilities to
  F16 to isolate mixed-type kernel conversions without assuming equivalence.
- Pinned upstream ggml KQ + softmax_ext + PV graph, cache F16 and F32 variants.
- Batched4 and16 versus tokenwise replay of the same exported Q/K/V.
- Source sigmoid/multiply gate versus native divide-by-exp gate, on identical
  ungated contexts. Keep O projection kernel fixed for cache comparisons.

Mirror the actual source cache strides: context2048, visible KV padded256,
K token-major, transposed V lane-major, ggml_set_rows F32-to-F16 writes. Use
causal ordinal mask for this row-major still-image input, Q/KQ scale0.0625,
16 query heads,4 KV heads,256 head width. Gather each positions axis from the
154-token capture independently. Do not truncate its first48 contiguous values.

Mark observed intermediate tensors OUTPUT to prevent fusion leaving zero or
unmaterialized buffers; require observed/composed final output bit equality.
Record stage shape/stride/type/hash, maximum magnitude, RMS/max errors and all
initial failures. A primitive match or speed difference cannot promote the
experimental physical profile, whose measured decoded result remains negative.

Preparation only: no worker is submitted by these files yet. `worker.py` performs
batched4/16 controls; tokenwise replay is explicitly not yet implemented and
cannot be inferred from these results. The protocol
freezes eleven whole source tensor ranges (including native layer3 FFN solely
for actual-helper byte closure), three actual capture hashes and resource caps.

Pre-dispatch review found and corrected two setup defects: scalar logits had
been scaled/causally restricted while source graph logits are raw QK over all
populated keys; and command-only timeouts did not bound HTTP/scalar work or
compiler descendants. The corrected probe stores raw all-key logits separately
from scaled causal softmax scores, uses a whole-worker SIGALRM deadline and
process-group cleanup, checkpoints command phases, validates all three actual
build caches, and reports all predeclared scalar rounding arms. Focused tests
cover the raw/future/padding distinction, deadline interruption, descendant
cleanup, success/nonzero receipts and build-profile mutations. The initial
descendant test used a 0.2-second Python startup window and timed out before
creating its fixture; its bounded two-second fixture window passes. No model
or operator result follows from these setup controls.
