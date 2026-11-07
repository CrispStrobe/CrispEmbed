# Neutral upstream form text trace

This source-only diagnostic replays the frozen form through pinned llama.cpp Q4_K_M, with live callbacks disabled and enabled. Both arms must reproduce all 174 original IDs, text, probabilities, EOS, the original first two score rows and vision features. Every new score row and dispatch record must match between arms.

The observer records actual batch geometry, prefill intermediates and the same-prefix decision at token 31. Retained combined native arrays supply comparisons without native inference. Startup compatibility probes are excluded only by exact independently pinned token/position semantics. Unsupported crops are reported, never inferred. A valid neutral capture remains distinct from reader admission.

Bounds: CPU four threads, build two, 3600 seconds overall and 900 per source arm; selected tensors at most 8 MiB each and 128 MiB total. No BF16 or F32 model, held-out data, runtime changes or quality promotion. Large arrays stay remote.
