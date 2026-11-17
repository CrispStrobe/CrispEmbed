# Native German spaCy development lane

This optional standalone lane implements the pinned `de_core_news_sm` 3.8.0
inference subset for Quds. It follows the [native spaCy handoff](https://github.com/CrispStrobe/quds_office_suite/tree/3ee7e5eeefdec0570246d4f0995e29995d5343e0/docs/roadmaps/native-spacy)
and the [source reconciliation receipt](../../docs/validation/native-spacy-20261117/README.md).

## Current boundary

W0/W1 model/reference admission, W2 lexical processing, W3 numerical inference,
and W4 linguistic decoding now compose through the compiled C ABI. Original
UTF-8 produces token spans, tags, POS, morphology, lemmas, dependency heads and
labels, and sentences without Python or a daemon. The installed Linux candidate
matches all 16 development cases / 113 tokens / 22 sentences, all 244 native
parser transitions and 339 classifier choices, and all 1,201 numerical stages.
See the [complete analysis summary](../../docs/validation/native-spacy-20261117/text-summary.json).
Separate lexical/decoder replays isolate failures; they are not additional
unique development documents. An expanded lexical panel covers 4,658 inputs.

The eight acceptance inputs remain unused while this development candidate is
frozen. Its separate acceptance gate requires a remotely saved candidate and
protocol, then a complete original reference saved before native execution.
Dart/editor integration, incremental grammar benefit, mobile packaging and
production release remain separate gates. NER and other model/locale profiles
are not part of this target.

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
Install the native PCRE2-8 headers/library before configuration. The tested
version is 10.46; it is a runtime dependency, with original notices installed
alongside the library. CMake accepts explicit `SPACY_PCRE2_INCLUDE_DIR` and
`SPACY_PCRE2_LIBRARY` paths for a prepared toolchain. Other PCRE2/platform
combinations require their own parity and packaging evidence.
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

.venv-spacy/bin/python tests/spacy_text_parity.py \
  --library /path/to/spacy-install/lib/libcrispembed-spacy.so \
  --bundle /path/to/verified-spacy-bundle \
  --quds-root /path/to/quds-spacy-reference-checkout \
  --report /path/to/new-text-receipt.json

.venv-spacy/bin/python tests/test_spacy_text_api.py \
  --library /path/to/spacy-install/lib/libcrispembed-spacy.so \
  --bundle /path/to/verified-spacy-bundle \
  --report /path/to/new-api-receipt.json
```

Use the paired Quds checkpoint
`dcf2ec9b4d6fba05fa5eec2634c6654266332cc9` or a descendant preserving its
exact fixtures. The comparator enforces the externally frozen model and
reference manifests, complete stage inventory, nonzero denominators, and
unchanged tolerances. Missing assets fail; no empty/skipped qualification is
accepted. It refuses to overwrite an existing receipt.

The four CTest executables are model-free controls. They cover hash arithmetic,
tensor/profile admission, malformed inputs, empty input, cancellation and
trace-abort recovery, concurrent callers, and result ownership. The separate
prepared-feature parity command runs the trained model on the oracle's supplied
features/states. The text comparator passes only original UTF-8 to native code,
then compares every discrete annotation, generated parser state/mask/action and
the full numerical trace. The 21 API control groups use the real admitted model
and authored adverse/lifecycle inputs. These are distinct recorded claims.

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
scores eight supplied state indices for numerical diagnostics. The original
`crispembed_spacy_capabilities()` retains its numeric-service mask of 3, and
the original ABI remains version 1 for existing clients.

For text, call `crispembed_spacy_create_with_resources()` with both file paths
and both externally trusted SHA-256 values. It also binds the resource hash to
the admitted model profile. `crispembed_spacy_context_capabilities()` reports
15 for this context (numeric scoring, tokenizer and complete analyzer), or 3
for a legacy model-only context. `crispembed_spacy_text_abi_version()` is 1.
`crispembed_spacy_tokenize_utf8()` and `crispembed_spacy_analyze_utf8()` accept
counted original UTF-8, including embedded NUL. Token views contain counted
UTF-8 fields, source spans in bytes/scalars/UTF-16, lexical IDs, and, for full
analysis, linguistic annotations. Roots use their own absolute token index.
Text-only contexts do not exist; both services share the admitted model owner.

Input bounds are 1 MiB and 2,048 tokens, with no truncation. Malformed UTF-8 and
over-limit input fail. Empty text returns an explicit empty owned document.
The parser allows at most `8*n+32` transitions; edit trees have depth/visit and
output-size limits. These are engineering bounds, not device latency budgets.

Calls on a context are serialized. Cancellation callbacks can read an external
atomic flag; do not reenter/free the context from a callback. Free contexts only
after active calls finish. Results may outlive the context, and borrowed views
remain valid until their result/document is freed. The context keeps no document
text after the call. Cooperative cancellation covers lexical loops, numerical
stages, parser transitions and dependency reconstruction; PCRE2 searches also
have work limits. Failed calls return explicit status and publish no successful
partial output. These are C ABI guarantees, not an
implemented Dart worker or editor cancellation protocol.

## Next implementation steps

1. Freeze this candidate remotely, publish a protocol, export/save the original
   acceptance reference, and run the eight untouched cases with
   `tests/spacy_acceptance.py`. Keep failures and never tune on this split.
2. W5: connect the optional Dart worker and analysis UI using counted strings,
   immutable results, original UTF-16 offsets and lifecycle/cancellation guards.
3. Measure grammar benefit against current deterministic Quds context rules
   before activation. Complete platform/device packaging and model/native
   distribution evidence separately.

The [native receipts](../../docs/validation/native-spacy-20261117/README.md)
retain the first and final numerical runs, negative setup evidence, dependency
inventory, installed notices and exact source hashes. No model alias, runtime
release, app pin, or grammar provider was enabled.
