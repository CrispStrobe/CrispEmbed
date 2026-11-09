# Actual v0.17.15 compatibility — 9 November 2026

The current Python and Dart Marker wrappers pass **34 assertions each, 68 total**
against the immutable v0.17.15 Linux CPU release. All six new artifact/lifecycle
symbols are absent. Calling each method reports the missing symbol, preserves
all four existing renders byte for byte, and allows another exact legacy PDF
conversion. No wrapper or native source fix was needed.

The [release archive](https://github.com/CrispStrobe/CrispEmbed/releases/download/v0.17.15/crispembed-linux-x86_64.tar.gz)
was checked against the size and SHA-256 recorded in
[`tests/results/releases/v0.17.15.json`](../../../releases/v0.17.15.json) before
extracting or loading it. Its source is
`d28ca2ea43a95934d673d1a54177fdf721614ade`. The passing probe uses wrapper source
`891ea3d3632c6a3c3155e25b34705da6a02ec807`; the wrapper bytes are identical to the
initial probe's source. Exact archive, runtime library, PDFium, wrapper and
fixture hashes appear in [summary.json](summary.json),
[release-verification.json](release-verification.json) and the full attempt
receipts.

This check uses real PDFium 151.0.7891.0 from pypdfium2 5.10.1 and the owned
`simpledigital.pdf` fixture. It uses no model weights and does not rebuild the
old native library, alter release assets or change dependency pins. It closes
the old-library client compatibility gate for this case; it does not qualify
model inference or general document quality.

## Preserved initial failure

[Attempt 1](attempt-1.json) stopped in both probes because the probe required the
complete heading as one substring in raw HTML. The existing renderer correctly
outputs the heading as three adjacent `strong` elements. The actual
[HTML output](initial-render.html), failed logs and original probe sources are
retained. The corrected probes inspect visible HTML text and parsed JSON/chunk
`text` fields. Archive, native library, PDFium, input and wrapper bytes are
unchanged between attempts. [Attempt 2](attempt-2.json) passes every command and
all 68 assertions. This was a probe expectation correction, not a runtime fix.

## Checked methods and errors

Both wrappers were checked for `save_extraction`/`saveExtraction`,
`open_extraction`/`openExtraction`, `start`, `step`, `cancel` and `progress`.
Python reports `RuntimeError` with the missing-symbol `AttributeError` as its
cause. Dart reports an `ArgumentError` naming the failed lookup. These expected
errors are retained in the passing receipt. Eleven legacy Marker exports were
also verified before conversion.

## Reproduction inputs

The portable corrected probes are in `probes/attempt-2/`. The Python probe
accepts `--python-source`, `--library`, `--library-sha256`, `--pdfium` and `--pdf`.
The Dart probe accepts the native library, PDFium library and PDF paths in that
order; compile it as package `marker_legacy_test` with the current shipped
`flutter/crispembed/lib/src/marker.dart` copied into `lib/marker.dart` and
`ffi: 2.2.0`. [dart-pubspec.lock](dart-pubspec.lock) is the byte-identical resolved
`pubspec.lock`, stored under an evidence-specific name to avoid repository
ignore rules. Python/Dart versions and Python dependency versions are recorded.

Use the verified release directory for `LD_LIBRARY_PATH`, set
`CRISPEMBED_MARKER=1`, and leave TeleOCR/model consent disabled. The probes used
`CRISPEMBED_MARKER_ARTIFACTS=1` and `CRISPEMBED_MARKER_INCREMENTAL=1`, ensuring
new-method failures are absent-symbol failures rather than closed feature gates.
Private path prefixes in logs and Dart's lookup messages are replaced with
placeholders. No library, executable, archive or model is included in this
receipt directory.
