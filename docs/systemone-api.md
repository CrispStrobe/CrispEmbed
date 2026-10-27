# SystemOne public API candidate

This candidate exposes the publisher's complete Josev joint-schema request and
response through a distinct opaque C context, `CrispSystemOne` in Python/Dart,
`SystemOne` in Rust, CLI `--systemone JSON`, and HTTP `POST /systemone` loaded
with `--systemone-model MODEL`. The response retains `model`, `answers` and
`usage`. Raw JSON preserves the insertion order of questions and criteria.

The profile is CPU/F32, batch one, with state-only clipping to512 tokens. It
requires a single self-contained packed GGUF and all seven explicit gates:
`CRISPEMBED_JOSEV_PACKED`, `CRISPEMBED_JOSEV_COMPOSED`,
`CRISPEMBED_JOSEV_BACKBONE`, `CRISPEMBED_QWEN35_OCR`,
`CRISPEMBED_JOSEV_HEAD`, `CRISPEMBED_JOSEV_TEXT_FRONTEND` and
`CRISPEMBED_JOSEV_ANSWER_SERIALIZER`, each set to `1`. Defaults remain off.
Requests are bounded to64KiB of NUL-free UTF-8 JSON. One context accepts one
request at a time; use separate contexts for concurrent calls. The HTTP handler
serializes access. C returned pointers expire on the next request or free.

The [model-free build](../.github/workflows/systemone-api-build.yml) checks the
actual root shared library, CLI, HTTP compilation, C exports/gates/errors,
thread-local initialization errors, wrapper lifecycle and compiler arithmetic.
It does not qualify a packed model or establish public model parity. A separate
full comparison must run all four frozen original raw requests through the
packed owner and then the actual C/Python/CLI/HTTP/Rust/Dart surfaces, checking
full typed responses, numeric errors, failure/recovery and offline resolution.
No model registry alias, publication or release is authorized by a build pass.

Root builds explicitly disable floating-point contraction for the twelve Josev
and shared Qwen source units to match the audited standalone CPU profile.
Because the Qwen units also serve OCR, final OCR acceptance must run the actual
integrated source and preserve its previous evidence and gates. Windows/Apple,
GPU, quantization, long context, batching and task quality remain separate gates.
