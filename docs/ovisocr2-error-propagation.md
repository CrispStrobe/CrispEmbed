# Ovis application error propagation candidate

The [frozen application comparison](../tests/results/crispsorter-ovis-comparison-20261112.json)
and [six-arm diagnostic](../tests/results/crispsorter-ovis-diagnostic-20261112.json)
separate an unsupported input from successful blank recognition. The real page
fails preprocessing with `image exceeds native patch bound`, zero selected IDs
and error stop 4 before vision or decoder execution. The small frozen fixture
returns `Hello OCR\n\n12345`, identical nine CPU/GPU IDs and EOS. No intermediate
arithmetic is reached on the rejected page, so numerical magnitudes or absolute
errors cannot explain this failure. The failed real-page panel is not Ovis model
quality evidence.

The candidate preserves reported recognition failures through the unified stage
and explicit Qwen3.5/Ovis recognition calls. Stage outcomes distinguish positive
success (including a non-NULL blank/EOS string), reported failure, and unknown
legacy emptiness. Nonempty legacy text is positive output evidence. A reported
failed stage cannot pass an acceptance gate. An unknown empty stage retains its
existing numeric permissive-gate behavior when no earlier stage reported failure;
it cannot erase an actual earlier failure. Unknown-only blank results do not
become new errors. This is not a complete error audit of legacy engines or existing
gated profile refusals, and unknown emptiness remains unqualified as recognition.

The shared fallback selector retains earlier positive output, including a valid
blank page and its engine/stage/geometry provenance, after a later reported
failure. It copies owned diagnostics before temporary stage storage expires.
The existing unified model handle does not reload when a later stage changes its
model path; multi-model unified cache ownership is a separate unresolved limit.

The C pipeline distinguishes a valid blank result (NULL region array, non-NULL
empty `full_text`) from a reported native failure (NULL `full_text`, zero regions
and confidence). `crispembed_ocr_pipeline_last_error` exposes an owned diagnostic
until the next run/free. Failed calls clear prior text, regions and review data.
`crispembed_ocr_model_last_error` exposes Qwen3.5/Ovis diagnostics for unified
models, returning empty when a model does not expose this diagnostic.

C layouts and existing function signatures remain unchanged; both accessors are
additive. Rust checks the existing NULL-text sentinel and does not require the new
symbols to link against an older immutable library. Its error is generic; native
stderr retains the exact cause. Python and Dart optionally query the new accessor
for details. The standalone CLI returns a failure exit code. An already compiled
consumer using the old Rust wrapper still needs a binding rebuild; swapping its
native library alone does not repair its false-success handling.

Four stdlib-only controls execute the actual Python wrapper method against native
failure, valid blank, older-library and failure-then-success outputs. Added native
model-free controls exercise real lazy unified initialization failure, cleared C
outputs, diagnostic lifetime and unavailable failed-run review data. Selection
controls cover failure plus unknown empty, known blank followed by failure with
provenance, successful fallback, nonempty legacy fallback and owned error/text.
Actual C-API controls contrast a mixed reported failure plus unknown zero-threshold
stage (failure) with a standalone unknown zero-threshold stage (the existing
accepted empty result). Existing no-text controls remain. Native compilation and execution,
Rust/Dart checks, real successful/failed fallback, valid EOS-only output, and the
same frozen synthetic/real-page candidate replay remain hosted acceptance gates.

No resizing, patch-bound extension, prompt/cap/model/scorer change, default flip,
new model qualification or release follows from this preparation. The version
remains unchanged until a fully qualified runtime release follows `VERSION`,
`scripts/sync-version.py` and the release gates. Shipping also requires a separately
qualified immutable consumer rebuild. Private execution configuration and raw
service/account identifiers remain outside Git.
