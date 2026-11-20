# Binary proofing database for the Camel Morph analyzer — 19 November 2026

This change gives the native Camel Morph analyzer
([text-database receipts](../camel-morph-msa-20261119/README.md)) a second
database representation: a versioned binary format
([spec](../../arabic-morph-binary-format.md)). The binary database is
memory-mapped instead of parsed, and it is reduced to a **feature profile**.
The text path is unchanged and still opens version-1 bundles. ABI 2 adds
`crispembed_ar_morph_analyze_compact`, a batched call that returns a binary
buffer instead of JSON.

## Feature profile and why

Quds Office's Arabic rules (`arabic_grammar_rules.dart`) read these features:
`pos`, `lex`, `cas`, `stt`, `num`, `gen`, `per`, `asp`, `mod`, `form_gen`,
`form_num`, `prc0`–`prc3` and `enc0`. The writing rules read none.

The profile keeps those plus:

- `diac`, `root`: lemma and form identity;
- `vox`, `rat`, `prc1.5`, `enc1`, `enc2`: the remaining inflectional and
  clitic features;
- `source`: its computed value `spvar` marks spelling-variant readings.

Each low-cardinality column costs one byte per row.

The analyzer itself needs only:

- `diac`, which is concatenated and rewritten for every analysis;
- `form_gen`/`form_num`, which `fix_form` copies into a `-` `gen`/`num`;
- `bw` on the `NOAN` stems, for the PROP backoff test.

The converter enforces these dependencies and stores `bw` on the NOAN rows
only, without outputting it.

Dropped: `gloss`, `bw`, `caphi`, `pattern`, `pattern_abstract`, `cm_*`,
`catib6`, `ud`, the tokenization features, `stem`/`stemcat`/`stemgloss`,
`correctness`, `mid_root_diac`, and the `*_logprob` features (a constant -99.0
in noprob).

## Exactness ([exactness.json](exactness.json))

The input is the same 9,571 words as the text receipt: all UD Arabic-PUD
words plus 124 stress inputs, words file SHA-256 `a87b5dd4…`. Both sides are
projected onto the profile, and each analysis keeps its profile features in
their own order.

| Comparison | Words exact (ordered) | Analyses equal | Profile values |
| --- | ---: | ---: | ---: |
| binary (JSON ABI) vs camel_tools, NONE | 9,571/9,571 | 217,053/217,053 | 4,499,457 |
| binary (compact ABI) vs camel_tools, NONE | 9,571/9,571 | 217,053/217,053 | 4,499,457 |
| binary vs native text noprob, NONE | 9,571/9,571 | 217,053/217,053 | 4,499,457 |
| binary vs camel_tools, NOAN_PROP | 9,571/9,571 (1 error in both) | 273,703/273,703 | 5,689,107 |
| binary vs camel_tools, ADD_ALL | 9,571/9,571 (2 errors in both) | 1,704,695/1,704,695 | 35,546,977 |
| text (compact ABI, all features) vs camel_tools, NONE | 9,571/9,571 | 217,053/217,053 | 9,740,479 |

- Analyses, their order, the key order and the values are identical. There
  is no frozenset-order caveat, because every profile key comes from the
  stem or the DEFAULT line.
- The compact ABI's `DEDUP` flag drops analyses of a word whose projections
  repeat, leaving 203,681 of 217,053. Sets are unchanged.
- `tests/test_arabic_morph.cpp` (CTest `arabic-morph-contract`) covers:
  - a binary fixture generated from the synthetic database by
    `tools/arabic_morph/gen_bin_fixture.py`, checked against the projected
    camel_tools outputs;
  - compact output equal to the JSON output for both formats;
  - the heap path (`CRISPEMBED_AR_MORPH_MMAP=0`);
  - corrupted and truncated files;
  - version-2 manifests.

  CI regenerates the fixture and fails on a diff.

End to end in Quds Office (`tool/pud_rule_findings.dart`), the Arabic rules
report the same 673 findings on all 1,000 PUD sentences in three setups: the
previous backend (JSON, noprob), the new backend with proofing-bin, and the
new backend with noprob.

## Performance ([performance.json](performance.json))

The machine was an Apple M1 shared with other agents (load average 4–26), so
absolute speeds vary by up to 3×. Before and after were run back to back or
interleaved. The native CLI used `/usr/bin/time -l`.

**Native CLI (9,571 words, NONE).** Session 1 is a single run each at load
6–18. Session 2 gives medians of 5 interleaved runs at load 9–26.

| | Load | RSS after load | RSS after all words | Words/s (session 1) | Words/s (session 2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| text noprob, JSON | 1.08 s | 233 MB | 238 MB (peak 303 MB) | 3,690 | 1,489 |
| binary, JSON | 0.106 s | 13.3 MB | 17.5 MB (peak 17.5 MB) | 6,189 | 2,635 |
| binary, compact | 0.107 s | 13.3 MB | 20.8 MB | 6,742 | 2,858 |

Load includes the SHA-256 of the manifest, `NOTICE`, `LICENSE` and the 12 MB
database, plus full validation.

**Through Quds' Dart API** (`tool/pud_benchmark.dart`, AOT; 7,419 distinct
PUD words):

| | Load | Words/s | RSS after load | RSS after analysing and caching all words |
| --- | ---: | ---: | ---: | ---: |
| before: JSON ABI, noprob | 1.5–2.5 s (median 4.2 s in session 2) | 709–921 (median 429 in session 2) | 230–250 MB | 373–415 MB |
| after: compact ABI, proofing-bin | 0.16–0.26 s (median 0.31 s) | 1,750–2,955 (median 1,802) | 30 MB | 90–100 MB |

- The process starts at 18 MB, so the analyzer adds about 12 MB.
- RSS after analysis is mostly uncollected Dart garbage. With `--verbose_gc`,
  the old space holds about 40 MB at the end, against about 180 MB before.
- Interleaved, the throughput gain is 4.2× (median); in session 1 it is
  2.5–3.2×.

**Web (Wasm worker):**

| | Load | Words/s | Memory |
| --- | ---: | ---: | --- |
| Node, before | 1.37–1.84 s | 1,725–1,785 | 320 MB Wasm heap, 340 MB process growth at load |
| Node, after | 0.13 s | 3,270–3,297 | 16 MB Wasm heap after load, 19 MB after all PUD words (`INITIAL_MEMORY` 16 MB); 41–47 MB process growth at load |
| Chrome 157 headless (Dart web transport), before | 1.37–1.42 s | 916–948 | 320 MB worker heap; page heap 203 MB with all analyses |
| Chrome 157 headless, after | 0.15 s | 3,493–3,523 | 16–19 MB worker heap; page heap 41 MB with all analyses; Chrome's total peak RSS 1.07 GB vs 1.84 GB |

## Publication ([hf-readback.json](hf-readback.json))

`proofing-bin/` was added to
[cstr/camel-morph-msa-v1.0](https://huggingface.co/cstr/camel-morph-msa-v1.0/tree/d5364e04af7a40b0c9044ed127606e585be15272/proofing-bin)
at commit `d5364e04af7a40b0c9044ed127606e585be15272`. Only `proofing-bin/*`
and the card were uploaded (`hf_publish.py upload --only …`).

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 2,064 | `357540af897af5dd2a0f7fc84518cc875c4481e3e566c30ca0a5043a52290f71` |
| `camel_morph_msa_v1.0.proofing.cmorph` | 12,166,992 | `fc9d9109ddb5430da94def715df05b563cb37ffaec508f3514668ded19438909` |
| `NOTICE` | 1,799 | `ce2f9c051786a37e4eb72367e6a6c9ea256b702137d4187bb0bfbda38f860efd` |
| `LICENSE` | 442 | `214965747a90efe3e5268d367192bb077f1101620662cba71d59d902000b2d94` |

An anonymous download of all 15 repository files at that commit into an empty
cache matched every size and SHA-256. That includes the unchanged `full/` and
`noprob/` bundles. The remote tree holds no other files.

`NOTICE` keeps the CC BY 4.0 attribution, citation and provenance, and lists
the changes: derived from noprob, converted, and reduced to the profile.
[manifest-proofing-bin.json](manifest-proofing-bin.json) is the published
manifest.

## Reproduce

```sh
python models/convert-camel-morph-db.py --source camel_morph_msa_v1.0.db --output out/
cmake -S tools/arabic_morph -B build && cmake --build build && ctest --test-dir build
build/crispembed-ar-morph --manifest out/proofing-bin/manifest.json --sha256 357540af… \
    --jsonl words.jsonl --out bin.jsonl [--compact [--dedup]] [--backoff ADD_ALL]
python tools/arabic_morph/camel_reference.py compare --ref ref-noprob.jsonl --native bin.jsonl \
    --profile out/proofing-bin/manifest.json
```

## Open

- Hosted CI on Windows (`MapViewOfFile` path) and Linux; the contract test
  passed locally on macOS only.
- iOS and Android packaging of the standalone target.
- The full-feature text bundle is still parsed into the heap (about 240 MB).
  A binary `full` variant would only pay off for a consumer that needs
  `gloss` or `bw`.
