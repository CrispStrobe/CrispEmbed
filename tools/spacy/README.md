# Native German spaCy development lane

This optional standalone lane implements the pinned `de_core_news_sm` 3.8.0
inference subset for Quds. It follows the [native spaCy handoff](https://github.com/CrispStrobe/quds_office_suite/tree/3ee7e5eeefdec0570246d4f0995e29995d5343e0/docs/roadmaps/native-spacy)
and the [source reconciliation receipt](../../docs/validation/native-spacy-20261117/README.md).

## Current boundary

W1 provides a real deterministic converter, a strict model/resource contract,
the frozen development oracle, and the numeric interface for the next native
stage. Conversion does not qualify text analysis. Tokenization, linguistic
decoders, full analysis, Dart/editor integration, grammar qualification, mobile
packaging, and any production release remain separate gates.

The implementation starts with CPU/F32 and a scalar numerical reference path
using ggml's GGUF/CPU weight ownership. This permits comparison of each source
operation before selecting an optimized ggml execution graph. This is an
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
