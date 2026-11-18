# English and French spaCy v2 distribution — 18 November 2026

Design: one Hugging Face repository per language holds only the model bundle;
the language-independent native library is the platform capsule published
once in `cstr/crispembed-spacy-de-core-news-sm-3.8.0`. Only capsules built from
source containing the v2 runtime (PR #244) open these bundles, so the existing
platform lane republished capsules from main `45b05770` (run 44369523105).
Tooling: PRs #245, #246, #248, #249, #250. All receipts: [receipts.json](receipts.json).

## Bundles

| Repository | Bundle revision | Card revision | License |
| --- | --- | --- | --- |
| `cstr/crispembed-spacy-en-core-web-sm-3.8.0` | `eb659259b489593ddf29c747bc4562fbad5570f4` | `e1808ee0a8083336a609822b7b4aa8e75a03ed02` | MIT |
| `cstr/crispembed-spacy-fr-core-news-sm-3.8.0` | `5e90593195cc4ecd20ceb864d932d7d7f5e3d762` | `577d2711cdd147a0cfa81b5794034003d6141cbb` | **LGPL-LR** (the official fr_core_news_sm license, not MIT) |

Published by run 44372635419 (main `e91b054d`): pinned wheel, hash-locked
conversion on hosted Linux, every pin (model.gguf EN `17b18871…`, FR
`8b8f9ab4…`, resources, verbatim notices) checked, one commit, exact immutable
tree and fresh empty-cache readback of every file. Cards/catalogs by run
44377019364 with the same readback.

## Libraries known to load them (run 44377019364)

Published bundle + published capsule, both verified by SHA-256/length, capsule
installed, admission controls and a ctypes-only analysis compared with
original spaCy 3.8.18 on UD r2.14 test text:

| Platform | Library | Capsule revision | EN (EWT) | FR (GSD) |
| --- | --- | --- | --- | --- |
| linux-x86_64 | `6e21ae15…` | `d143d26a` | 2,077/2,077 texts, 25,531 tokens, 1,876/1,876 entities | 416/416, 9,786 tokens, 469/469 |
| linux-arm64 | `21d15c1e…` | `fd75184d` | same | same |
| macos-arm64 | `f3300152…` | `cf833ae5` | same | same |
| windows-x86_64 | — | not published | — | — |

## Open issues

- **Windows v2 capsule not published**: in run 44369523105 the Windows library
  raised an MSVC C++ exception (`0xe06d7363`) out of
  `crispembed_spacy_create_with_resources` for a wrong model hash
  (`test_spacy_text_api.py` admission case 0); all other controls and 60/60
  smoke texts passed. The pre-v2 Windows capsule passed this control. Runtime fix needed.
- **iOS SDK** job of run 44369523105 was cancelled while queued; the Android SDK
  from `45b05770` was published. Neither is run with the EN/FR bundles.
- **Converter drift**: after PR #247, the converter produces a different
  `model.gguf` (EN `c05f7947…`, FR `42f9dd3b…`, same sizes). The pin check
  refused it (run 44376408858, no Hub write); `use_published=true` consumes the
  published bundles instead. A new bundle version needs its own qualification.
