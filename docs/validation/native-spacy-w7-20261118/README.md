# W7: verified Linux spaCy package and consumers

**Status on 18 November 2026: the local Linux package and installed-consumer gates are complete.** The fresh continuation restored the exact accepted native/model bytes and qualified dependency builds, reproduced the package, and executed both consumers. W0–W6 remain preserved; their consumed language-qualification panels were not rerun.

## Verified results

| Gate | Observed result |
| --- | --- |
| Current package controls | 25 passed |
| Current Python consumer controls | 21 passed; printed protocol byte-identical |
| Current Dart consumer controls | 17 passed, including loader-environment and source-path regressions |
| Exact inputs | 15 native files, five model files and six external dependency identities match |
| Full package reproduction | Two new complete builds are byte-identical and match the original saved hash and length |
| Immutable installation | Atomic installation and full installed-tree verification pass |
| Python operational consumer | Four fresh children pass; 24 panel and 78 warmup/cycle executions; 102 contexts/documents created and freed |
| Resource observations | Maximum sampled post-warmup RSS growth 102,400 bytes against 33,554,432 bytes; no final descriptor growth |
| Actual Quds Dart worker | Eight cases / 75 tokens / 12 sentences plus five worker-control requests pass |
| Preserved source | All 342 sealed logical sources are preserved: 341 direct Git paths plus the documented lockfile mapping |

The Python consumer checks exact annotations across all three enabled processes and all short cycles; its disabled child explicitly refuses context creation. The Dart consumer checks original coordinates at a nonzero UTF-16 origin, stable copied results, same-worker and reopened-worker repeat, unsupported `de-AT`, caller cancellation and awaited idempotent disposal. Both bind execution to the fixed package and actual mapped system dependencies.

## Authoritative evidence

- [Paired completion record](resume-20261118/completion.json) and [precise current status](checkpoint-status.json).
- [Python/package evidence at the saved CrispEmbed result](https://github.com/CrispStrobe/CrispEmbed/tree/482786cce5450e5e9532f5917e2007bbe1c120db/docs/validation/native-spacy-w7-20261118/resume-20261118), including every original Python child report.
- [Current Dart result at the saved Quds checkpoint](https://github.com/CrispStrobe/quds_office_suite/tree/a6143772f048644187080ee6a1a44c070802af15/docs/validation/native-spacy-w7-20261118/resume-20261118/dart-consumer), with the initial result preserved separately.
- [Fresh complete branch/PR inventory](resume-20261118/source-refresh.json), [strict source-mapping audit](resume-20261118/source-refresh-audit.json), and [publication-before-execution record](resume-20261118/pre-native-publication.json).
- [Independent paired evidence audit](https://github.com/CrispStrobe/CrispEmbed/blob/feat/native-spacy-w7-20261118/docs/validation/native-spacy-w7-20261118/resume-20261118/independent-review.json).

All available origin branch and PR heads were fetched first. Both current mains were already included. Two later workspace-handoff commits were integrated before publication; the branch updates retained them. A final refresh found main had advanced through worksheet PR #235. Its 13 incoming commits are now included, preserving all worksheet changes and resolving only history/plan text. The [final reconciliation](resume-20261118/final-main-reconciliation.json) rehashes all 22 frozen native source files and confirms the tested consumer sources remain identical. Office PR #154 overlaps one broad sealed engine path and remains inventoried separately from this consumed W6 candidate.

The final remote reconciliation also preserves the later package/Python audit at `24aba65a` and Dart audit at `ef22a83b`, including their separate refresh inventories. Their exact artifact identities are recorded in the paired completion. These commits add retained-evidence review; the tested consumer, frozen native/model sources and original execution reports are unchanged.

## Corrections and original evidence

Independent review added missing Dart rejection of four loader override variables (including empty values) and an absolute dependency-inspection tool path. The initial installed-worker run then passed. A retained-evidence audit found 14 truncated source labels caused by a trailing directory separator; their hashes mapped unambiguously to the exact executed files. The original report remains unchanged. A small source-path repair, five new controls, a remote source checkpoint and one corrected-source operational run close that defect. All eight saved outputs match the first run after excluding timing.

The earlier workspace exhaustion and later unavailable-workspace probe are historical, preserved in the preceding commits. This fresh workspace completed the pending work. It does not establish that earlier local files were deleted or recover the older missing raw logs. Public Dart reports redact declared host paths and reorder JSON object keys; raw/public digests and the exact transformation are explicit, and annotation/list order is preserved.

## Exact package and scope

The package is 7,808,129 bytes, SHA-256 `4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`. Its manifest is `e043af51e247368931f23fa0d9b931453e174f105b021fe2053e8e3029fd3d8d`. It contains the original library, header and notices plus the original five-file model bundle; the native library was restored, not rebuilt.

This evidence covers the preserved Linux x86-64/AVX2, de-DE, CPU/F32 candidate with six exact external dependency builds. The library has no RPATH/RUNPATH. GLIBC 2.38, GLIBCXX 3.4.29 and CXXABI 1.3.9 are symbol requirements, not broad system qualification. Resource samples and single-run timings are bounded observations. Repeated operational texts add no independent linguistic-quality cases, and caller cancellation does not establish interruption of native computation. No Flutter UI behavior was observed in these CLI runs.

Immutable hosted publication and fresh-cache download readback, portable dependency bundles, other platforms/devices and production activation remain separate. The manual provider stays default off, with no automatic Studio registration, release or app-pin change. No GitHub Actions/status query, wait, rerun or dispatch was used. Checkpoints use `[skip ci]` and non-forced branch updates with verified remote trees.

Use [CONTINUE.md](CONTINUE.md) for the next bounded work and reproducible input commands.
