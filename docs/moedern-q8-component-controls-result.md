# MoEdern Q8 component controls

[Actual40837105606](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40837105606)
at`dd592b15f8191be0b441eda3932c98364324e776` completes the hosted model-free
build and actual C quantizer fixture. The [independent receipt](../tests/results/moedern-colbert/q8-component-controls-review.json)
authenticates all eleven small ZIP members, checks successful command outcomes,
fixture/manifest identities and actual storage counts. Binaries and authored
GGUF payloads remain remote; this small-file review does not reread their bytes.

Three named profiles cover20/20/52 quantized tensors for1.5M/27M/115M, with
9 rejected storage mutations, gateOFF/0/concurrency controls and6 actual authored
Q8 direct/expert graphs. The C fixture writes two Q8 matrices and seven byte-exact
critical F32 tensors with standard file type7. Runtime is the frozen default-off
protected-Q8 implementation; this experiment does not change defaults.

This is component evidence only. No real model, corpus, original forward or
full-model quantization ran. Existing F32 corpus and router blockers remain;
resolve them before separately closed real Q8 and Q4_K qualification, final task
quality, public consumers and publication. Passing authored graph/storage tests
does not establish real embedding or retrieval parity.
