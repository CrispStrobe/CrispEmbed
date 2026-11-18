# W7: immutable Linux spaCy packaging

Status: package controls passed; current consumer execution is blocked by
workspace exhaustion, 18 November 2026. This continuation starts from the completed W6 checkpoint `93560df6` and
paired Quds checkpoint `20026ff4`. The accepted native library, model bundle,
original native acceptance, and W6 grammar qualification remain unchanged.

The [source inventory](source-refresh.json) records every fetched fork branch
and PR head, current main, the Quds documentation-only merge, and unchanged
Git blob identities for all 342 sealed Quds source files. Historical Office
evidence remains at its recorded original branches. The worksheet and GECToR
lanes retain their independent ownership.

## Saved status

| Work | Observed outcome |
| --- | --- |
| Source refresh and relevant main integration | Complete; all 342 sealed Quds source blobs preserved |
| Package controls | 25 passed on the saved packager source |
| Real package build and atomic installation | Passed; 20 payload files plus authenticated manifest |
| Second complete real-input build comparison | Pending |
| Corrected Python consumer | Source reviewed and saved; current 21 controls and first native run pending |
| Dart installed-package consumer | Source reviewed and saved; compilation, self-test and worker run pending |

Local execution is blocked by filesystem exhaustion during sandbox startup. Even task-owned cleanup commands could no longer launch. Approximately 440 MB of reproducible task files had already been removed before the later exhaustion. GitHub connector saves and readback remained available.

Follow [CONTINUE.md](CONTINUE.md) and [checkpoint-status.json](checkpoint-status.json) from a workspace with capacity. The latest remote source can be newer than the local checkouts. Package/model binaries were not uploaded to ordinary Git, and some raw local preflight/SDK logs could not be reread after exhaustion; these preservation limits are explicit in the checkpoints.

## Bounded work

Create a deterministic package of the 15 original native/header/notice files
and five original model files. Authenticate the package against an external
SHA-256 before reading its archive, admit only the fixed manifest and payload,
and install atomically into a private content-addressed cache. The installed
library retains its original bytes and immediate SHA-256 parent directory for
the existing Quds factory. Package tooling is a developer tool, with no Python
runtime added to the application.

The unchanged Linux x86-64/AVX2 library depends on host PCRE2 and C/C++ system
libraries. It has no RPATH/RUNPATH. Actual ELF symbol requirements reach
GLIBC 2.38, GLIBCXX 3.4.29 and CXXABI 1.3.9. These are compatibility floors,
not qualification of arbitrary distributions. The first operational consumer
must match all six dependency hashes from W6, then verify the libraries
actually mapped into its process. See [runtime restoration](runtime-restoration.json).

## Predeclared operational checks

The [operational protocol](operational-protocol.json) fixes eight authored
input shapes, three fresh enabled processes, one disabled-gate process, and
25 context/document ownership cycles after one warmup per enabled process.
It fixes a 60-second child timeout, a 32 MiB post-warmup RSS growth bound,
and no growth in final open descriptors. These are bounded engineering checks,
not additional independent language cases or proof of general absence of leaks.

Local package controls must reject changed hashes, missing and unexpected
members, duplicate names, traversal, links, invalid manifests, oversized
payloads, corrupt existing installs and unsafe publication. A second build
must reproduce identical package bytes. The actual Dart worker consumer must
use the installed paths and current source without altering the application
or either consumed linguistic panel.

Results will retain failures and source hashes. No GitHub Actions/status query,
wait, rerun or dispatch is required. Checkpoints use `[skip ci]` and exact
remote-tree verification.

## Remaining distribution scope

This step qualifies the preserved Linux development binary. New portable
builds and dependency bundles need their own source/toolchain and parity
evidence. macOS, Windows, Android, iOS and browser native execution are
unqualified. Immutable hosted model/native publication and consumer download
readback remain separate from a reproducible local package; model weights stay
outside ordinary Git. Production activation remains a separate decision.
