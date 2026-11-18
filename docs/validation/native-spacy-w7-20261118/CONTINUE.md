# Continue after the completed local W7 gates

The Linux package and both installed-consumer gates completed on 18 November 2026. Read [README.md](README.md), [completion.json](resume-20261118/completion.json) and [checkpoint-status.json](checkpoint-status.json) first. The earlier exhausted-workspace handoff is preserved in history; it no longer describes this continuation's execution state.

## Source and result checkpoints

Both repositories use `feat/native-spacy-w7-20261118`. Fetch the current branch and preserve local differences before changing a worktree.

| Repository | Checked consumer source | Saved operational result | Review |
| --- | --- | --- | --- |
| CrispStrobe/CrispEmbed | `a71c3ce5326346310289db70953f5e300d9c1aaf` | `482786cce5450e5e9532f5917e2007bbe1c120db` | [PR #237](https://github.com/CrispStrobe/CrispEmbed/pull/237) |
| CrispStrobe/quds_office_suite | `71e87486489200f7e259594d2967bba8c0682143` | `a6143772f048644187080ee6a1a44c070802af15` | [PR #153](https://github.com/CrispStrobe/quds_office_suite/pull/153), stacked on W6 / PR #139 |

The source refresh fetched all available origin branches and PR heads. CrispEmbed main `be20e497` (including worksheet PR #235) and Quds main `482e0817` are included; `04e84458` remains the original recorded CrispEmbed base. The later documentation-only workspace probes are preserved as `b847053d` and `225eeb92`. All 342 sealed logical sources still match: 341 direct Git paths and the documented historical lockfile artifact. The final-main reconciliation preserves every incoming worksheet source change and all 22 frozen spaCy source hashes. Office PR #154 remains inventoried separately; do not silently replace the consumed W6 source with unrelated changes.

The final remote reconciliation also preserves the later package/Python audit at `24aba65a` and Dart audit at `ef22a83b`, including their separate refresh inventories. Their exact artifact identities are recorded in the paired completion. These commits add retained-evidence review; the tested consumer, frozen native/model sources and original execution reports are unchanged.

## Next bounded work

1. Define the immutable hosted distribution destination and publication/readback contract. Publish the qualified bytes with notices and external hashes, then verify an actual empty-cache consumer download. A local reproducible package alone is not hosted publication.
2. For another target, declare its OS/CPU/dependency/toolchain contract before execution. Retain any rejected old-pin preflight, and qualify the new target separately. Do not loosen the six frozen dependency identities or substitute a rebuilt library into these results.
3. Preserve the default-off provider and explicit Studio registration boundary. Runtime releases, app pins and production activation remain separate decisions.

Do not rerun the consumed native acceptance or W6 grammar panel to tune a new candidate. The completed operational cases may be reused for a declared engineering change; retain new source/result identities and do not count repeated inputs as new linguistic observations. No GitHub Actions query, wait, rerun or dispatch is needed for this lane.

## Reproduce exact inputs when needed

The original native archive is committed at `93560df6e4e8a592535216ca94b883f5934a915c:docs/validation/native-spacy-20261117/acceptance-candidate-linux-x86_64.tar.gz`. Its 825,745 bytes must hash to `b0cd3ae4d47af442d32a6d2382143c36f8b019470cff7c509f97e4682b67489b`. Restore these bytes; do not rebuild as a substitute.

The official `de_core_news_sm` 3.8.0 wheel is 14,639,490 bytes with SHA-256 `fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7`. Use CPython 3.12.14, the unchanged full hash lock and tracked converter in [CrispEmbed tools/spacy](https://github.com/CrispStrobe/CrispEmbed/blob/feat/native-spacy-w7-20261118/tools/spacy/README.md). All five bundle hashes must match. Model weights remain outside ordinary Git.

The exact package is 7,808,129 bytes, SHA-256 `4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`. Build/install with `tools/spacy/distribution.py` and verify the complete immutable tree. The two fresh full-input builds were compared byte for byte; the unavailable older workspace archive was compared by its retained complete hash and length.

The native library has no RPATH/RUNPATH. PCRE2 and five system runtime dependencies are external and must match the fixed hashes; symbol floors alone do not establish portability. The exact Dart 3.13.7 archive and `crypto 3.0.8`, `ffi 2.2.0`, `typed_data 1.4.0` identities and restored file inventory are in the [Quds Dart controls](https://github.com/CrispStrobe/quds_office_suite/tree/a6143772f048644187080ee6a1a44c070802af15/docs/validation/native-spacy-w7-20261118/resume-20261118/dart-controls).

## Operational commands for a declared subsequent check

The current sources pass 25 package controls, 21 Python consumer controls and 17 Dart controls. Save any changed checked source remotely before native execution. Choose new report filenames so originals cannot be overwritten. Set the variables below to authenticated local inputs and check workspace capacity first.

```sh
python3 -B tools/spacy/distribution_consumer.py \
  --archive "$SPACY_W7_ARCHIVE" \
  --sha256 4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01 \
  --installed "$SPACY_W7_INSTALLED" \
  --protocol docs/validation/native-spacy-w7-20261118/operational-protocol.json \
  --protocol-sha256 60f85df220bb05ff467c6036ce465c841cff3480103720d5139af0644cfd3934 \
  --report new-python-consumer.json

"$SPACY_DART" --disable-dart-dev --packages="$SPACY_PACKAGE_CONFIG" \
  "$SPACY_QUDS/packages/quds_proofing_spacy/tool/consume_distribution.dart" --self-test

CRISPEMBED_SPACY=1 timeout 300 "$SPACY_DART" --disable-dart-dev \
  --packages="$SPACY_PACKAGE_CONFIG" \
  "$SPACY_QUDS/packages/quds_proofing_spacy/tool/consume_distribution.dart" \
  --archive "$SPACY_W7_ARCHIVE" \
  --sha256 4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01 \
  --installed "$SPACY_W7_INSTALLED" \
  --protocol "$SPACY_QUDS/docs/validation/native-spacy-w7-20261118/operational-protocol.json" \
  --protocol-sha256 60f85df220bb05ff467c6036ce465c841cff3480103720d5139af0644cfd3934 \
  --report new-dart-consumer.json
```

Python retains the frozen three-enabled/one-disabled child arrangement, 60 seconds per child, 25 measured ownership cycles after warmup, 32 MiB sampled growth bound and no final descriptor growth. Dart exercises the actual owned worker with the same eight operational case types; it does not duplicate Python's resource-cycle qualification. Retain full outputs and limitations, including original failures and metadata corrections.
