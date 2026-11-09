# PDF Craft package evidence — 9 November 2026

Package source bytes are pinned to `d4d508c460c6e172fd1d7d4afc6b9c78652ad607`.
Every recorded Python and Flutter package source hash was checked against that
revision before these receipts were retained. These local checks were explicitly
authorized by the user. No package source or version was changed, and no package
was published.

## Python archive readback

[The Python receipt](python/receipt.json) records **26 checks, zero failures**
using Python 3.12.14, setuptools 84.2.1 and wheel 0.48.0. The actual native library
contains 12,586,264 bytes and has SHA-256
`0f1034b037ac2d3b09d8d80b5b956b53dc9c17f9ef77e871192acb1e56d9833a`.

The checkout wheel, source archive and wheel rebuilt from the isolated source
archive each preserve that exact binary and the canonical `NOTICE.marker`,
`NOTICE.teleocr` and Apache 2.0 license text. The isolated rebuild has no repository
parent notices. Missing each of the three notice files and an empty Apache text
produce the expected explicit rejection. The original native binary and package
sources remain unchanged. Exact native, notice, package and source hashes are in
the receipt; [command excerpts](python/command-excerpts.log) retain archive entries
and rejection diagnostics.

The runner copies the binary as opaque bytes and never loads it. These checks do
not establish installed-wheel runtime behavior, cross-platform builds, model
quality or reproducible whole-archive bytes. Native and client runtime evidence is
recorded separately.

## Flutter package selection

[The Flutter receipt](flutter/receipt.json) records **15 scoped checks, zero
failures** with Dart 3.13.7. The ordinary `dart pub publish --dry-run` fails because
the Flutter SDK is unavailable. The explicitly limited
`dart pub publish --dry-run --skip-validation` succeeds without dependency
resolution or upload; a verbose repeat records 41 selected files in 59 stream
members. All nine root, iOS and macOS notice files are selected, and their unchanged
source bytes match the canonical texts exactly. The staged package matches the
pinned checkout bytes. [Command excerpts](flutter/dry-run-excerpts.log) preserve
the ordinary failure, explicit skipped-validation warning and archive member list.

Dart reports an 80 KB compressed archive stream but does not retain that stream
in this mode. **No archive-byte readback is claimed.** The receipt explicitly sets
`archive_byte_readback` and `full_flutter_validation` to `false`. Flutter SDK,
CocoaPods/Xcode and actual platform builds remain outside this evidence.

## Retention

Only sanitized receipts and compact command excerpts are retained here. Original
and sanitized full-attempt log hashes remain in the receipts for traceability.
Archives, binaries, private host paths and credentials are excluded.
