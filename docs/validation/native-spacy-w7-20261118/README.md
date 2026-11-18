# W7: verified Linux spaCy package and consumers

**Status on 18 November 2026: bounded Linux package reproduction, immutable
installation, Python operational consumption and the current Dart worker
consumer have passed.** Independent retained-evidence audits authenticate those
results. The [paired completion record](completion-20261118/paired-completion.json)
and [current status](checkpoint-status.json) supersede the earlier
workspace-exhaustion handoff. The original execution receipts remain unchanged.

## Completed local gates

| Gate | Verified result |
| --- | --- |
| Package admission and installation controls | 25 passed on the saved packager source |
| Complete real-input reproducibility | Two new archives are byte-identical; both match the original recorded SHA-256 and 7,808,129-byte length |
| Immutable installation | All 20 payload files plus the manifest verified; atomic publication without replacement |
| Current Python controls | 21 passed; emitted protocol equals the frozen bytes |
| Python operational consumer | One disabled and three enabled processes passed; 24 panel executions and 78 warmup/cycle executions; 102 context/document pairs created and freed |
| Python sampled resources | Each enabled process grows by at most 102,400 bytes after warmup against a 32 MiB bound; final descriptors remain 3 to 3 |
| Current Dart compilation/self-test | Pinned Dart 3.13.7; 17 controls passed on the corrected current source |
| Installed Quds worker | Eight operational cases, 75 tokens and 12 sentences; five additional lifecycle/control requests passed |
| Current source identity | 38 Dart source/configuration entries authenticated; all 342 sealed W6 sources unchanged |
| Independent evidence review | Python/package: 1,031 checks, zero failures; Dart source/coordinates/dependencies/sealed-source audit passed |

The [Python/package audit](completion-20261118/python-package-audit.json) and [Dart audit](https://github.com/CrispStrobe/quds_office_suite/blob/ef22a83b6aef4aa4383f14415c42c7aba67f4873/docs/validation/native-spacy-w7-20261118/completion-20261118/dart-audit.json)
recompute report hashes, source identities, original-coordinate reconstruction
and the declared totals from retained evidence. They do not execute a model or
rerun either consumed linguistic qualification. The [package comparison](resume-20261118/package/distribution-byte-comparison.json),
[first Python report](resume-20261118/python-consumer/first.json) and [corrected-source Dart report](https://github.com/CrispStrobe/quds_office_suite/blob/ef22a83b6aef4aa4383f14415c42c7aba67f4873/docs/validation/native-spacy-w7-20261118/resume-20261118/dart-consumer/corrected-source.json)
retain the actual earlier executions.

## Concurrent completion evidence retained

The [concurrent completion](resume-20261118/completion.json),
[strict source-mapping audit](resume-20261118/source-refresh-audit.json) and
[final main reconciliation](resume-20261118/final-main-reconciliation.json)
are retained alongside the reproducible auditors above. The separate paired
review records 615 named checks; it is not added to the 1,031 Python/package
assertions as independent coverage. The native reconciliation preserves all
22 frozen native source files. See the [merge record](completion-20261118/concurrent-reconciliation.json)
for both concurrent branch heads and authenticated receipt identities.

The source count is 342 logical identities: 341 direct Git paths plus the
published W5 mapping from logical `pubspec.lock` to
`docs/validation/native-spacy-w5-20261117/workspace-pubspec.lock`. The latter
is 22,686 bytes with SHA-256
`344190a0224b9e6dcc2161cbfb1be3b7a9569a3a03a809f42801d99f9052e130`.
The current Dart auditor validates that mapping explicitly.

The Dart correction history remains intact: four loader override variables,
including empty values, are rejected before validation; dependency inspection
uses the fixed tool path. The first successful worker report had 14 truncated
source labels. Their hashes mapped to the actual executed files; the label-only
repair added five controls and was published before the corrected-source run.
All eight retained case outputs are identical across that correction.

## Refreshed sources and preservation

The [complete source refresh](completion-20261118/source-refresh.json) reconciles
245 CrispEmbed branches and 191 Quds branches, 217 and 154 fetched PR heads,
respectively, and all PRs updated since 17 November. Only four branch heads
changed since the prior inventory: both W7 heads, CrispEmbed main, and the
independent Quds Office lane.

CrispEmbed main `be20e497` (merged worksheet PR #235) is integrated into W7 at
`d18efcaf`. That integration commit had one conflict: equivalent PR #236
closure text in PLAN; both
facts were retained. The accepted spaCy archive, converter/consumer sources and
prior evidence are unchanged. Quds main `482e0817` was already included. Office
PR #154 is inventoried and remains separate from the sealed W6 candidate.

The package retains the original 15 native/header/notice files and five model
files. Its external SHA-256 is
`4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`;
its manifest SHA-256 is
`e043af51e247368931f23fa0d9b931453e174f105b021fe2053e8e3029fd3d8d`.
The frozen protocol remains
`60f85df220bb05ff467c6036ce465c841cff3480103720d5139af0644cfd3934`.
Generated package/model binaries are reconstructible from pinned inputs and
remain outside ordinary Git. No hosted capsule has been published.

## Scope and evidence limits

Qualification is for the preserved Linux x86-64/AVX2, de-DE, CPU/F32 development
binary and six exact external dependency hashes. The library has no
RPATH/RUNPATH. GLIBC 2.38, GLIBCXX 3.4.29 and CXXABI 1.3.9 are inferred symbol
floors, not qualification of another host or a rebuilt library.

The Python panel has eight case types and nine distinct texts including its
cycle. Its repeated engineering checks add no independent linguistic cases.
RSS/descriptor observations are finite post-free samples, not a general leak
proof. Dart checks source coordinates, copied annotations, unsupported locale,
caller cancellation, disposal and reopen through the real owned worker. It does
not replicate Python's resource cycles, observe Flutter UI, or prove physical
native-compute interruption.

The public Dart reports and logs are available and rehashed. Their unredacted
raw counterparts are not in this fresh checkout; recorded raw digests do not
replace those bytes. Separate repeat results and in-memory `sentenceEnd` fields
were not serialized, so the corresponding outcomes remain source-bound harness
assertions. Some earlier raw pre-exhaustion receipts were never recovered; that
historical limitation remains explicit.

## Next work

Follow [CONTINUE.md](CONTINUE.md) for exact source checkpoints, reproducible
read-only audit commands and the remaining distribution gates. Hosted immutable
publication with independent empty-cache download/readback, portable dependency
bundles, other platforms/devices and production activation remain open. The
manual grammar provider remains disabled by default, without automatic Studio
registration. This completion publishes no runtime release and changes no app pin.

Both continuation PRs remain draft for the broader W7 distribution work. Remote
checkpoints use non-forced expected-head updates and verified commit/tree
readback. No GitHub Actions query, wait, rerun or dispatch was performed.
