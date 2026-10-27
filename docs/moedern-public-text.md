# Unfinished MoEdern1.5M public-text source draft

**This branch is a recoverable source draft. It has not been compiled or run.**
The hosted consumer driver/workflow, actual packed artifact, ordinary archive
exclusion, compiled ABI/lifetime controls and all three actual binding clients
remain unfinished. No public model parity, release, default,27M/115M, retrieval
quality or production packaging claim follows from its source/stdlib checks.
Shared root CMake/header/exports, CLI, server, registry and module export lists
are untouched. The user-requested wrap-up stopped expansion at this boundary.

The reviewed predecessor is
[40455093591](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40455093591)
at `1a053116908224cf1c9358d7d24c85ba36e82eff`: raw text/explicit roles passed all
ten frozen cases,20 original final-vector arrays and three MaxSim pairs.
Its SMALL report is `98af43c6780bbd11b1ec9ec4f26c7b2e3a2348507072408408bf726b2e89482e`.
Its20 native final-vector hashes also match the earlier native-v2 exactly.
Those targets, masks, gate0.002/cosine0.99999 and MaxSim0.002 remain unchanged.
Do not rerun originals or turn the owned tiny packer fixtures into model gold.

## Draft files and interfaces

`moedern_text_api.h/.cpp` declares a separate opaque C handle and five exports:
`crispembed_moedern_init/free/encode/score/error`. Init takes a local packed F32
GGUF and threads0(default2),1 or2. Encode takes UTF8 text and explicit role0(query)
or1(document), returning an ABI-versioned, size-checked borrowed struct with IDs,
supplied/effective/scoring masks, query expansion positions and both complete
and selected row-major128-dimensional vectors. No dense pooling, generic token
dropping, IDs or hidden-vector injection is exposed. Buffers expire at the next
encode, score or free, including failed calls; callers must copy first. Score
takes two raw texts and clears its output on failure. Null-handle/init errors
are thread-local. One call per context at a time; destruction requires no calls
in flight. No public reference callbacks or diagnostic types are declared.

Python `_moedern_text.py`, Rust `moedern_text.rs` and Dart `moedern_text.dart`
copy every result buffer before another native call. Roles remain explicit.
Dynamic bindings test all five capabilities before model access, so importing an
older library remains possible. Python/Rust reuse existing model resolution;
Dart currently accepts a local packed path only. Bindings reject NUL/oversize
text; Rust UTF8 is intrinsic, Python strict UTF8 encoding and Dart scalar-string
checks reject unpaired surrogates. These sources still require actual hosted
compilation, Dart analysis, Rust linking and native-backed lifetime/recovery
proof. The stdlib fake-library test is only a wrapper contract check.

New public and packed gates add to the three existing model gates, all default
off: `CRISPEMBED_MOEDERN_PUBLIC_TEXT`, `CRISPEMBED_MOEDERN_PACKED`,
`CRISPEMBED_MOEDERN_TEXT_COMPOSED`, `CRISPEMBED_MOEDERN_TEXT_FRONTEND`,
`CRISPEMBED_MOEDERN_COLBERT`. The packed frontend snapshots its tokenizer gate;
public/composed/native/packed gates are checked on each encode/score. New packed
load methods preserve the old zero-tensor tokenizer load and old two-file init.
The engine file and frontend normalization/token/DP/mask bodies are unchanged.

## Byte-preserving pack draft

`tools/pack_moedern_text_f32.py` authenticates the old F32 model(5941568 bytes,
SHA256 `9f9792f0ae76430496ac93951cc09a0804468ba3f22b4314b303aea85dfbe8fa`),
reviewed tokenizer component(1095840 bytes,
SHA256 `3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1`)
and exact reviewed composition report before its hosted-only production entry.
It retains the original model architecture,32 tensor names/shapes/F32 payloads
and original metadata records, embeds the complete tokenizer metadata including
F64 score bytes and normalized matcher history, and records replaced tokenizer
architecture plus component/report provenance. It performs no tensor arithmetic.
The1MiB streaming copier independently reads every tensor back and records its
SHA. The new loader bounds16MiB file/4MiB raw header, metadata allocations,
32 F32 extents, padding and1484448 parameters before GGUF allocation; the old
engine then applies its unchanged exact32-shape/source contract. Structural and
provenance declarations do not authenticate arbitrary user-edited weight bytes:
actual packed file/tensor hashes must be externally pinned by the hosted audit.

## Exact next wiring and acceptance work

1. Review/compile the draft on hosted CI first; do not dispatch a model job merely
   because source checks pass. Add compiled C prefix/offset, gate0, null/error,
   failed-output clearing and valid/reject/valid ownership/lifetime controls.
2. Parent coordinates root wiring: add `src/moedern_text_engine.cpp`,
   `src/moedern_text_frontend.cpp`, `src/moedern_text.cpp`, `src/moedern_text_api.cpp`
   to static/shared sources; apply `-ffp-contract=off` to those units. Keep
   `CRISPEMBED_MOEDERN_COMPOSED_TESTING` off. Install the new C header; expose it
   deliberately without changing existing dense/ColBERT semantics. Register
   Python `CrispMoEdern`, Rust `mod moedern_text`/reexports and the Dart module only
   after review. CLI/server/registry require separately coordinated additions.
3. Freeze a new explicit integrated source closure. Preserve the old1a,3da and
   native-v2 whole-source guards; they correctly reject this new draft. Check
   actual archive symbols and strings for exclusion of reference callbacks.
4. Authenticate old original/Unigram/composed service+ZIP+member identities
   remotely, then pack the immutable components. Record complete file/metadata/
   all32 tensor readback proof before any candidate forward. Keep all artifacts
   remote. A hosted consumer driver/workflow is still to be written.
5. Run the ten raw-text cases/three pairs through actual C/Python/Rust/Dart,
   checking exact IDs/three masks/expansion, plain/repeat/observer neutrality,
   old20 arrays using actual `crispembed_diff::Ref` plus independent metrics,
   three old scores plus independent MaxSim, offline paths and all recovery/
   ownership controls. No original token/vector injection into the model.
6. The predecessor forced LLAMAFILE OFF; root production defaults ON. Qualify
   actual root ON separately under unchanged gates, record compile commands and
   actual backend paths. A standalone OFF result cannot qualify production ON.
   No release/version/model publication before the complete integrated proof.
