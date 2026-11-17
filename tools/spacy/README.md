# Native German spaCy development lane

This optional standalone lane implements the pinned `de_core_news_sm` 3.8.0
inference subset for Quds. It follows the [native spaCy handoff](https://github.com/CrispStrobe/quds_office_suite/tree/3ee7e5eeefdec0570246d4f0995e29995d5343e0/docs/roadmaps/native-spacy)
and the [source reconciliation receipt](../../docs/validation/native-spacy-20261117/README.md).

## Current boundary

W0/W1 source reconciliation, deterministic conversion, strict model/resource
admission, and the frozen development oracle are complete for the recorded
Linux environment. The W3 numerical stage now runs through the compiled C ABI:
hash embeddings, shared CNN, three classifier heads, and parser-state scoring.
It passes all 1,201 frozen stages and exact decisions on 16 development cases.
Tokenization, linguistic decoders, full analysis, Dart/editor integration,
grammar qualification, mobile packaging, and production release remain separate
gates. The eight acceptance inputs have not been used during development.

The implementation starts with CPU/F32 and a scalar numerical reference path
using ggml's GGUF/CPU weight ownership. Each source operation has been compared
before selecting an optimized ggml execution graph. This is an
explicit staging decision within the C++/ggml implementation, not a change of
model. No existing model path, release version, or default changes.

## Reproduce the conversion

Use CPython **3.12.14** in an isolated developer environment. The hash lock was
installed and verified independently on Linux x86-64; other platforms must
record their own environment and repeat the reference comparison.

```sh
python3.12 -m venv .venv-spacy
.venv-spacy/bin/python -m pip install --require-hashes --only-binary=:all: \
  -r tools/spacy/requirements-reference.txt
.venv-spacy/bin/python models/convert-spacy-to-gguf.py \
  --wheel /path/to/de_core_news_sm-3.8.0-py3-none-any.whl \
  --output /path/to/new-spacy-bundle
```

Supply the [official wheel](https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl)
yourself. It must be 14,639,490 bytes with SHA-256
`fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7`.
The converter authenticates it before archive parsing, extracts only model
data, and loads that data through the pinned developer spaCy installation.
It never imports Python code from the model package and never downloads.
Existing output directories are refused to protect frozen artifacts.

The output consists of `model.gguf`, `resources.json`, `manifest.json`,
`LICENSE`, and `LICENSES_SOURCES`. Model weights are kept outside ordinary Git.
The immutable source wheel plus the pinned converter/environment reproduce
the bundle; output byte hashes and tensor inventories are retained in the
validation receipt. A separately published native/model distribution does not
exist yet. The application must not run this converter or embed Python.

## Contract and reference evidence

[MODEL_FORMAT.md](MODEL_FORMAT.md) defines version 1. The authoritative tensor
shape inventory is `models/spacy_bundle.py:expected_tensors()`. All 39 trained
tensors are read back independently from GGUF after conversion and compared
byte-for-byte with their original F32 values. Resources preserve tokenizer
rules, Unicode tables, normalization precedence, ordered labels/moves,
learned edit trees, attribute rules, and model notices.

The Quds branch [feat/native-spacy-reference-20261117](https://github.com/CrispStrobe/quds_office_suite/tree/feat/native-spacy-reference-20261117/tools/proofing/spacy)
owns authored inputs, full annotations, exact lexical IDs, padded intermediate
traces, parser decisions, and fail-closed fixture validation. Empty input is
an explicit fixture; it cannot substitute for nonempty inference. Acceptance
inputs are frozen separately and stay untouched during numerical development.

Local verification is required for this lane. The user explicitly prohibits
querying, polling, awaiting, or rerunning GitHub Actions. Commits are saved
incrementally with `[skip ci]` and their remote trees are verified.

## Build and run the native numerical gate

The standalone target does not configure the parent OCR/document stack.
The tested configuration is Linux x86-64, GCC 14.2.0, CMake 4.4.5, Ninja
1.13.2, and the repository's AVX2 CPU baseline. Other platforms are unqualified.
The native target has no Python or daemon dependency; Python below runs the
developer comparison harness around the compiled C ABI.

```sh
git submodule update --init ggml
cmake -S tools/spacy -B build-spacy -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-spacy --parallel 2
ctest --test-dir build-spacy --output-on-failure
python scripts/check-cpu-baseline.py build-spacy
cmake --install build-spacy --prefix /path/to/spacy-install

.venv-spacy/bin/python tests/spacy_parity.py \
  --library /path/to/spacy-install/lib/libcrispembed-spacy.so \
  --bundle /path/to/verified-spacy-bundle \
  --quds-root /path/to/quds-spacy-reference-checkout \
  --report /path/to/new-native-receipt.json
```

Use the paired Quds checkpoint
`dcf2ec9b4d6fba05fa5eec2634c6654266332cc9` or a descendant preserving its
exact fixtures. The comparator enforces the externally frozen model and
reference manifests, complete stage inventory, nonzero denominators, and
unchanged tolerances. Missing assets fail; no empty/skipped qualification is
accepted. It refuses to overwrite an existing receipt.

Both CTest executables are model-free controls. They cover hash arithmetic,
tensor/profile admission, malformed inputs, empty input, cancellation and
trace-abort recovery, concurrent callers, and result ownership. The separate
parity command runs the actual trained model on the oracle's prepared features
and parser states. These are distinct claims and are recorded separately.

## Native API and limits

[`spacy_pipeline.h`](../../src/spacy_pipeline.h) is the complete ABI contract.
Set `CRISPEMBED_SPACY=1` explicitly before creating a context; unset, empty,
`0`, `false`, `off`, and `no` leave it disabled. Supply a trusted model SHA-256.
The loader authenticates bytes, validates the closed profile, and copies the
weights. Keep the model file immutable during opening. Later replacement of
the file does not mutate an admitted context.

`crispembed_spacy_forward_features` accepts six uint64 feature IDs per token
and returns owned shared vectors and unnormalized head scores. Inputs are
limited to 2,048 tokens without truncation. `crispembed_spacy_parser_scores`
scores eight supplied state indices; the caller still supplies legal actions
and performs transitions. Capabilities advertise these two numerical services
only. No raw-text analysis function is exposed yet.

Calls on a context are serialized. Cancellation callbacks can read an external
atomic flag; do not reenter/free the context from a callback. Free contexts only
after active calls finish. Results may outlive the context, and borrowed views
remain valid until their result is freed. Failed calls return explicit status
and publish no successful partial output. These are C ABI guarantees, not an
implemented Dart worker or editor cancellation protocol.

## Next implementation steps

1. W2: port German tokenizer, lexical features, string IDs and exact original
   source spans using the exported resources and frozen lexical fixtures.
2. W4: implement parser state/validity/transitions/deprojection, edit-tree
   application/fallback, morph/POS decoding and final attribute rules.
3. Qualify complete raw-text annotations and only then connect the optional
   Dart worker and analysis UI. Measure any grammar benefit against the current
   deterministic Quds context rules before activation.

The [native receipts](../../docs/validation/native-spacy-20261117/README.md)
retain the first and final numerical runs, negative setup evidence, dependency
inventory, installed notices and exact source hashes. No model alias, runtime
release, app pin, or grammar provider was enabled.
