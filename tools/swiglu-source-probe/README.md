# Same-input FFN SwiGLU attribution

This reference-only worker measures captured observed-table layer0 FFN operators;
it changes no production source and runs no model or held-out case. The prior
hybrid GDN decoded table regression remains binding and its gate remains off.

The fixed inputs are actual upstream4 and completed native GDN-OFF16/ON16
gate/up/activation/down/postnorm arrays. Prefix1/4/16 cases remain distinct from
the actual source4/native16 contexts. Scalar replay must close both actual
native16 activations; pinned upstream fused `ggml_swiglu_split` must close the
actual live source4 activation before comparisons are interpreted.

Three whole payloads from the published hash-pinned GGUF total45,895,680bytes: Q4_K gate/up and Q6_K
down. Their unchanged coordinates/hashes are checked against the admitted
physical import receipt and range responses. Preferred-Q4 gate/up and original
Q6 down must reproduce actual source4 outputs; RAW native projections must close
both actual native16 sets. There is no Q4-down or global-repack substitution.

Identical gate/up inputs are then replayed with native scalar and native/upstream
GGML SwiGLU; each activation feeds identical Q6 down weights. Public Q8_K
converter vectors are retained as diagnostic interventions. They are not a live
internal-buffer capture: forced-Q8 versus F32-input output equality is measured
separately for every case/backend. Errors include absolute and RMS magnitudes.

CPU-only: two build/four compute threads,1800seconds, minimum4GiB RAM/6GiB scratch
and1GiB output. Builds, real synthetic backend controls and activation closures
precede parameter transfer. Raw arrays/negatives remain remotely retained. No
component result establishes decoded improvement, a tensor recipe or shipping.

Primary blueprint: pinned [Qwen3.5 FFN](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/qwen35.cpp) selects parallel SiLU; [build_ffn](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/llama-graph.cpp) implements that path with `ggml_swiglu_split`. Actual live-node closure is required rather than assuming equivalent algebra.
