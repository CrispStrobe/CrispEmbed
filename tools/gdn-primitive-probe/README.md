# Same-input GDN operator attribution

This reference-only target compares the first GDN block on retained native QKV,
Z, alpha and beta projections. It loads no model and performs no decoding.
`protocol-v1.json` freezes the actual capture hashes, five complete parameter
ranges (6,030,080 bytes), source function hashes, geometry and resource limits.
The worker rejects any protocol mutation before transfer or compilation.

The upstream blueprint is [qwen35.cpp](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/qwen35.cpp),
[delta-net-base.cpp](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/delta-net-base.cpp), and
[models.h](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/models.h).
The no-draft context defaults enable fused AR and CH with automatic selection
disabled; zero initial state uses the K=1 fused operator. Source convolution,
mean-based RMS plus scaling for Q/K L2 normalization, softplus/sigmoid, fused
recurrence and gated post normalization remain distinct stages.

The source graph runs with both pinned ggml revisions. The native reconstruction
must first match the unchanged real native grouped helper values and state byte
for byte. Its 16-token dequantized output projection must then match the retained
actual main attention output byte for byte. Observation sinks must leave composed
results unchanged. Projection-only controls feed identical native post-normalized
values to retained Q4_K and dequantized F32 matrix multiplication. All raw stages,
build caches, hashes and phase logs remain remote.

Four tokens reproduce the actual initial text-prefix batch length. Sixteen tokens
match the native capture length and include image tokens; this source graph is a
same-input component experiment, not a capture of upstream's full multimodal
execution. Initial history and recurrent state are zero. This probe cannot prove
later-state, full-attention cache, decoded-quality or complete-runtime parity.
It deliberately measures backend and arithmetic differences without a tolerance
waiver. Strict reconstruction failures stop attribution and remain evidence.

Run model-free protocol controls with
`python tools/gdn-primitive-probe/test_protocol.py`.
The production graph never depends on this executable. Hosted invocation uses
`tools/kaggle/lighton4b_gdn_source_primitives.py`, with four CPU threads, two build
threads, a 30-minute outer cap and 2 GiB effective RAM / 4 GiB scratch admission.
