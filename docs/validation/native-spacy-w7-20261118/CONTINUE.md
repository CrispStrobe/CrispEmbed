# Continue W7 from the saved remote checkpoints

Status on 18 November 2026: packaging controls and the real package build/install passed. Both new consumers are saved source; their current execution is pending. The workspace filled, and even a cleanup command could not start because sandbox setup returned ENOSPC. This is an execution blocker, not a completed W7 qualification.

A later read-only workspace probe also failed before execution: the execution service returned `No such file or directory`. Current local mounting, contents and free capacity cannot be verified. This does not establish that files were deleted; use the independently verified remote branches as the recovery source.

## Authoritative source

Both repositories use `feat/native-spacy-w7-20261118`. Fetch the branch before continuing. The source checkpoints below precede this documentation-only handoff:

| Repository | Source checkpoint | Review |
| --- | --- | --- |
| CrispStrobe/CrispEmbed | `d1d9985e143206e8f8e930cf23b947fd71f9c451` | Draft [PR #237](https://github.com/CrispStrobe/CrispEmbed/pull/237) |
| CrispStrobe/quds_office_suite | `2e57784093fd2d1f44fe27edb31077d4163693f2` | Draft [PR #153](https://github.com/CrispStrobe/quds_office_suite/pull/153), stacked on the W6 branch / PR #139 |

The local worktrees may still contain earlier staged and untracked versions. Use fresh isolated checkouts, or preserve and compare those local differences before updating. Do not overwrite an older local tree just because the remote branch is newer.

All available fork branches and PR heads were fetched and inventoried first. CrispEmbed main `04e84458` is included; Quds main `482e0817` is merged at `13747281` with 72 evidence files and no production changes. All 342 sealed Quds source blobs match the candidate. Consult [source-refresh.json](source-refresh.json) and the full compressed inventory before reconciling additional changes.

## Restore inputs without changing the accepted candidate

The original native archive is already in CrispEmbed at `93560df6e4e8a592535216ca94b883f5934a915c:docs/validation/native-spacy-20261117/acceptance-candidate-linux-x86_64.tar.gz`. Its 825,745 bytes must hash to `b0cd3ae4d47af442d32a6d2382143c36f8b019470cff7c509f97e4682b67489b`. Do not rebuild the library as a substitute.

Retain the generated five-file bundle if available. Otherwise reproduce it with the tracked converter and full hash lock using CPython 3.12.14 and the pinned official `de_core_news_sm` 3.8.0 wheel. See CrispEmbed `tools/spacy/README.md` and [runtime-restoration.json](runtime-restoration.json) for all input/output identities, notices and six exact dependency pins. Model weights remain outside ordinary Git.

The real package is 7,808,129 bytes, SHA-256 `4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`; its distribution manifest is `e043af51e247368931f23fa0d9b931453e174f105b021fe2053e8e3029fd3d8d`. The generated capsule has not been published as a hosted distribution. Restore it locally from the pinned inputs if needed.

The library has no RPATH/RUNPATH. PCRE2 and the five other runtime dependencies are external. Its GLIBC 2.38 / GLIBCXX 3.4.29 / CXXABI 1.3.9 symbol floors do not qualify a different system build. If a fresh workspace has different dependency bytes, retain the rejected preflight and explicitly qualify that environment; do not weaken the frozen pins.

## Resume local checks in order

1. Recover sufficient workspace capacity. Remove only identified, reproducible task-owned temporary files after checking they are no longer needed. Do not delete unrelated tasks or weaken access controls. Fetch the saved source and retain any differences from older local files.
2. Run the 21 current Python consumer controls. The earlier 18 passing controls belong to different local source hashes. The saved reconstructed/corrected source has not run:

```sh
python3 -B tests/test_spacy_distribution_consumer.py
python3 -B tools/spacy/distribution_consumer.py --print-protocol > new-protocol.json
cmp new-protocol.json docs/validation/native-spacy-w7-20261118/operational-protocol.json
```

3. Preserve the actual test outcomes and exact source identities remotely before native execution. Record failures and fixes; do not relabel an earlier pass as a pass of changed source.
4. Repeat the real-input package build into a new filename and compare its complete bytes/hash with the first archive. The passing 25 package tests establish fixture determinism; this second full-input comparison is still pending. Reverify the installed immutable tree.
5. Run the first Python operational consumer with the external archive/protocol pins and a new report filename. Set `SPACY_W7_ARCHIVE` and `SPACY_W7_INSTALLED` to authenticated local paths:

```sh
python3 -B tools/spacy/distribution_consumer.py \
  --archive "$SPACY_W7_ARCHIVE" \
  --sha256 4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01 \
  --installed "$SPACY_W7_INSTALLED" \
  --protocol docs/validation/native-spacy-w7-20261118/operational-protocol.json \
  --protocol-sha256 60f85df220bb05ff467c6036ce465c841cff3480103720d5139af0644cfd3934 \
  --report new-python-consumer.json
```

The protocol is already frozen: eight panel case types and nine distinct texts including the cycle, three enabled children and one disabled child, 60 seconds per child, 25 measured ownership cycles after one warmup, at most 32 MiB sampled post-warmup RSS growth, and no final descriptor growth. Expected completed counts are 24 panel executions, 78 cycle/warmup executions and 102 context/document pairs. Retain every first-run outcome and original child report. These are bounded operational checks; they do not extend linguistic quality claims.

6. Restore the authenticated Dart 3.13.7 direct VM and three pinned packages if needed. Their exact archive hashes and the prior successful existing-package import probe are in Quds `dart-consumer-checkpoint.json`. That probe did not compile the new driver. Set `SPACY_DART`, `SPACY_PACKAGE_CONFIG` and `SPACY_QUDS` to the restored executable, external package config and Quds checkout, then compile/run the new driver's model-free self-test:

```sh
"$SPACY_DART" --disable-dart-dev --packages="$SPACY_PACKAGE_CONFIG" \
  "$SPACY_QUDS/packages/quds_proofing_spacy/tool/consume_distribution.dart" --self-test
```

After fixing, checking and remotely saving any source changes, run the same Dart script with the six `--archive`, `--sha256`, `--installed`, `--protocol`, `--protocol-sha256` and `--report` arguments shown for Python, plus an explicit `CRISPEMBED_SPACY=1` environment and an external process timeout. Use a new report filename. Retain source hashes and actual outcomes. This exercises the real owned Quds worker; it is not a Flutter UI observation or a claim that caller cancellation interrupts native compute.

7. Save successful or failed operational results incrementally on both remote branches using non-forced updates and verified readback. Keep both PRs draft until their declared local gates are complete. Do not wait for, query, rerun or dispatch GitHub Actions.

## Preservation limits and remaining work

The source, package build/install receipts, 25-test output, native/model restoration evidence, source inventory and frozen operational protocol are saved remotely. The corrected Python source and Dart draft were saved through the connector after local commands failed. Some earlier raw consumer-preflight and Dart-restoration/review logs could not be reread after ENOSPC; their checkpoint metadata/hashes do not substitute for the unavailable raw files.

The native acceptance and W6 grammar qualification are already consumed and retained; do not rerun or tune on them. W6's paired completion is preserved at CrispEmbed `93560df6` and Quds `20026ff4`. The manual provider remains disabled by default, with no automatic Studio registration. Other platforms/devices, portable dependency bundles, immutable hosted publication/download readback and production activation remain separate work.

See [checkpoint-status.json](checkpoint-status.json) for the precise completed/pending boundary.
