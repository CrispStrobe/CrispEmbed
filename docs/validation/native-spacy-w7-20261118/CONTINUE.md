# Continue W7 after the verified Linux package and consumer gates

The old ENOSPC handoff is superseded. The saved package, current Python consumer
and corrected current Dart consumer have completed their declared local checks.
Start from the remote branch and the [paired completion record](completion-20261118/paired-completion.json);
do not schedule pending runs from the historical source-only checkpoints.

## Exact verified starting points

Both repositories use `feat/native-spacy-w7-20261118`. New documentation commits
can follow these immutable audit checkpoints without changing the tested source.

| Repository | Retained audit checkpoint | Review |
| --- | --- | --- |
| CrispStrobe/CrispEmbed | `24aba65adc3633366fe4b3cf20818ca2965dfefd` | [PR #237](https://github.com/CrispStrobe/CrispEmbed/pull/237) |
| CrispStrobe/quds_office_suite | `ef22a83b6aef4aa4383f14415c42c7aba67f4873` | [PR #153](https://github.com/CrispStrobe/quds_office_suite/pull/153), stacked on W6 / PR #139 |

Fetch all origin branches and available PR heads before new work, inspect newer
commits, and preserve any existing local differences. Use isolated worktrees.
The [refresh receipt](completion-20261118/source-refresh.json) records the current
main integration and why the independent Office PR #154 remains outside this
sealed candidate. Preserve all 342 W6 source identities.

## Verify the existing results without native execution

From the CrispEmbed checkout:

```sh
python3 -B docs/validation/native-spacy-w7-20261118/completion-20261118/audit_retained_python_package.py \
  --repo . --output /path/to/new-python-package-audit.json
```

From the Quds checkout:

```sh
python3 -B docs/validation/native-spacy-w7-20261118/completion-20261118/audit_retained_dart.py \
  . --output /path/to/new-dart-audit.json
```

Use new report paths and preserve failures. These standard-library audit scripts
load neither Dart nor native/model binaries and make no Actions calls. With the
referenced Git objects already available, the audit is local; a partial clone
may fetch missing source objects. They authenticate retained execution/source
identities; they are not new runtime observations. The Python/package audit has 1,031 assertions. The Dart audit
recomputes all eight cases and all 342 sealed source identities.

Completed gates: 25 package controls; two equal full-input builds matching the
original archive hash/length; complete installed-tree verification; 21 current
Python controls and the 102-pair operational run; 17 compiled Dart controls and
the eight-case corrected-source worker run. Preserve first outcomes and
correction history. Rerun affected engineering controls only for an actual source
or environment change. Native acceptance and W6 grammar qualification are
consumed and must not be rerun or tuned on as fresh evidence.

## Restore exact inputs only when the next task needs them

The original native archive remains at CrispEmbed commit
`93560df6e4e8a592535216ca94b883f5934a915c`, path
`docs/validation/native-spacy-20261117/acceptance-candidate-linux-x86_64.tar.gz`.
Its 825,745 bytes hash to
`b0cd3ae4d47af442d32a6d2382143c36f8b019470cff7c509f97e4682b67489b`.
Use these library bytes; rebuilding creates a different candidate.

The model's five files can be reproduced with CPython 3.12.14, the full tracked
hash lock and official `de_core_news_sm` 3.8.0 wheel. Refer to the paired
[CrispEmbed package restoration](https://github.com/CrispStrobe/CrispEmbed/blob/24aba65adc3633366fe4b3cf20818ca2965dfefd/docs/validation/native-spacy-w7-20261118/resume-20261118/package/README.md)
and [runtime restoration](runtime-restoration.json). The package is 7,808,129
bytes with SHA-256
`4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`.
The two restored builds were compared directly; comparison with the unavailable
first-workspace archive is through its retained SHA-256 and byte length.

For an affected Dart check, the authenticated SDK 3.13.7 and three package
archives are fully inventoried in the [saved Dart restoration](https://github.com/CrispStrobe/quds_office_suite/blob/ef22a83b6aef4aa4383f14415c42c7aba67f4873/docs/validation/native-spacy-w7-20261118/resume-20261118/dart-controls/runtime-restoration.json).
Use the direct VM plus external package config; a Flutter SDK is unnecessary
for this developer consumer. The current driver digest is
`cecbc3a4c52dc8880c9eae7add5a925b0c74660e9d1ba1aee7aaae0ffe02cdf6`.

All six runtime dependency hashes remain frozen. Preserve a rejected preflight
if a fresh host differs; qualify the changed environment explicitly. The ELF
symbol floors do not authorize weakening these dependency pins.

## Next bounded distribution lane

1. Declare the exact target, artifact destination and intended immutable version
   before publication. Keep this accepted Linux package and its external digest.
2. Reconstruct the capsule only if needed, verify the full hash and notices, and
   record its immutable artifact identity. Keep model/package weights outside
   ordinary Git. No artifact has yet been hosted by this W7 work.
3. Qualify a fresh consumer download into an empty private cache: authenticate
   archive bytes before parsing, install atomically, verify every payload, and
   run the declared package consumers with new evidence files. Keep the original
   finite operational protocol and budgets; this adds distribution evidence,
   not new linguistic-quality evidence.
4. Treat a portable dependency bundle, rebuilt binary, other OS/architecture or
   mobile/UI/device target as its own declared candidate and qualification.
5. Preserve source, protocol, failures and results incrementally on the same
   remote branches. Use non-forced updates with an expected head and read back
   both commit parents and tree. Keep the paired PRs draft while that broader
   distribution work remains open.

Production activation is a separate decision. The provider remains default off
and Studio has no automatic grammar registration. Caller cancellation does not
establish physical native-compute interruption. Public, path-redacted Dart
receipts are preserved; missing raw originals remain a stated limitation.

Do not query, wait for, rerun or dispatch GitHub Actions. Use `[skip ci]` for
checkpoints. This user instruction governs the lane despite generic repository
CI guidance. There is no current local execution blocker recorded for the
completed Linux gates.
