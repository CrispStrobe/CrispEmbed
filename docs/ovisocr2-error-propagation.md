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

The [bounded CPU evidence](../tests/results/qwen35-ocr/ovisocr2-error-propagation.json)
records source `458458b2a308a1e6e7d3e55f7276a47761d487fa`: all 27 hosted
commands pass, including 107 native orchestrator controls (zero failures), review
compilation/execution, Rust semantic/build controls and the actual two-input
Rust/Python consumers. The frozen synthetic retains all nine IDs/raw text/EOS;
the real page remains rejected before vision with zero IDs and error stop 4.
The new Rust consumer also links/runs with the old immutable runtime, retaining
that old runtime's prior empty-result behavior. This is link compatibility, not
a repair of an already compiled old consumer.

Four stdlib-only controls execute the actual Python wrapper method against failure,
blank, older-library and failure-then-success outputs. Native controls contrast an
actual mixed reported failure plus unknown zero-threshold stage (failure) with a
standalone unknown zero-threshold stage (its existing accepted empty result).
Selection controls cover known blank followed by failure with provenance,
successful/legacy-text fallback and owned diagnostics/results. An unknown legacy
blank that fails its gate cannot prove success over a later reported failure;
this ambiguity requires a richer engine status contract to resolve.

Public reference-only [Rust](../tests/ovisocr2_error_bindings.rs),
[Python pipeline](../tests/ovisocr2_error_bindings.py) and
[native diagnostic](../tests/ovisocr2_error_native.py) programs match the hosted
program bytes recorded in the evidence. They are test controls, outside production
source and binary assets. The initial bounded CPU receipt predates the additional
mixed-chain coverage and remains retained separately. One source-sealed T4/SM75
CUDA build/replay completed: all 29 commands pass, including the same two-input
replay and strict CUDA dispatch. The shared CUDA build took 1739.317 seconds; the
bounded worker took 1842.736 seconds. Synthetic raw output, all nine IDs and EOS
remain exact; the real page still fails the unchanged native patch bound before
vision. Native bundles remain remotely retained; these are unpublished prototypes. Standalone CLI,
Rust/Dart semantics on other native platforms, browser error reporting and final
merged-source/release/consumer qualification remain separate gates.

No resizing, patch-bound extension, prompt/cap/model/scorer change, default flip,
new model qualification or release follows from this preparation. The version
remains unchanged until a fully qualified runtime release follows `VERSION`,
`scripts/sync-version.py` and the release gates. Shipping also requires a separately
qualified immutable consumer rebuild. Private execution configuration and raw
service/account identifiers remain outside Git.

The subsequent Python result-mirror correction adds the native record's existing
`orientation_angle` and `orientation_confidence` fields: native 64-bit stride 48
versus the old Python stride 40. Dart now uses a complete typed FFI record for
reading and rendering, replacing manual offsets/strides; rendering counts UTF-8
bytes. Four Python ABI/array controls, four Python failure controls and one Dart
header-declaration control pass locally. These controls are not hosted production
binding qualification. The [model-free native-memory control](../tests/ocr_binding_native_memory.py)
loads the full production Python module and production Dart sources, exercises
three native regions, failure/recovery and native render packing. Its hosted
combined-source execution remains pending. No 32-bit or other-platform Dart
execution has been performed; those platforms remain explicitly unqualified.
