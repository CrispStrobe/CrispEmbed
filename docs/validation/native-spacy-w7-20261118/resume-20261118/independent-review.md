# Independent review of W7 operational closure

**The bounded Python and corrected Dart operational results pass.** This review rehashed retained evidence, inspected authenticated source, reconstructed counts and source coordinates, and preserved the initial provenance defect. No tests, model inference or consumed panels were rerun by the reviewer.

## Python first operational result

All four raw child files match their stored digests and the aggregated report. The disabled arm loads the library but explicitly refuses a context and performs no inference. Three enabled arms each create and free 34 contexts and documents. Recomputed denominators are **24 panel calls + 78 cycle/warmup calls = 102 ownership pairs**, covering nine distinct texts including empty input and the cycle text. There are 99 nonempty analyze inputs; empty cases are kept in the ownership denominator.

Original byte/scalar/UTF-16 coordinates, token partitions, sentence spans and dependency graphs validate for every retained full output. Full panel/warmup outputs agree across processes; all 75 measured-cycle hashes match independently rehashed warmups. Each enabled process has 102,400 bytes maximum sampled RSS growth under the unchanged 32 MiB budget, one sampled thread, and three initial/final descriptors. This is sampled bounded behavior, not a leak proof or device benchmark.

## Dart first result and repaired source

The first Dart run completes eight panel cases, **75 tokens and 12 sentences**, plus five control requests. Its exposed annotations/sentences and original coordinates exactly match Python, including the nonzero ten-unit UTF-16 origin. All six dependency identities and all 38 recorded source/configuration hashes verify. The public report exactly reproduces the declared metadata-only redaction of the retained original; both raw and public hashes were recomputed.

The original driver contained one provenance defect: a trailing slash caused source-path slicing to drop the first relative-path character from all 14 Quds library labels. The JSON audit preserves a unique reported-label to actual-path mapping and verifies all corresponding launched-source hashes. The original report remains unchanged.

Published source `71e87486489200f7e259594d2967bba8c0682143` repairs that metadata helper and adds five separator/boundary regressions. Its 17 model-free controls pass. A single separately named current-source worker run passes in approximately 2.49 seconds. All 38 labels now name actual source/configuration entries, all 37 non-driver hashes are unchanged, and all 14 Quds library blobs match the published commit. Every saved case payload is exactly equal to the original after excluding elapsed time. Repeating these eight operational inputs adds no new linguistic cases.

The authenticated driver asserts request-snapshot equality, same/reopened-worker repeat equality, awaited idempotent disposal, stable copied results, immutable token collections, explicit unsupported locale and caller cancellation. Cancellation demonstrates caller settlement and awaited ownership cleanup; physical compute start or prompt interruption is not observed. Worker queue/text limits are statically bound; this small operational panel does not independently stress every capacity limit.

## Preserved boundaries

All **342 sealed W6 logical source identities** remain preserved: **341 direct Git paths plus one previously documented lockfile mapping**. Root `pubspec.lock` is deliberately ignored; its exact bytes are retained at `docs/validation/native-spacy-w5-20261117/workspace-pubspec.lock`. The mapped artifact's full SHA-256 matches the seal, original W5 publication and current tree. Every direct path must explicitly exist in both trees before blob equality is accepted; missing/missing is not proof. Eighteen materialized direct files were additionally rehashed, together with the mapped lock artifact. [The source-refresh amendment](source-refresh-audit.json) retains the original receipt unchanged and lists every verified direct Git blob plus all three full-byte lockfile readbacks.

The frozen operational protocol, accepted native/model payloads, immutable installed tree and two complete rebuilt package bytes remain exact. Neither consumed qualification panel was executed again.

The detailed JSON distinguishes raw-data checks from executed source-bound assertions. Python retains full panel/warmup annotations and measured-cycle hashes/resources. Dart omits request snapshots and separate repeated/disposal snapshots from its JSON, limiting independent recomputation of those assertions; its checked source and recorded passing outcomes support them. Dart retains mapped dependency paths and byte hashes, while Python also checks mapped inode/device identity.

No further actionable finding remains in this bounded closure. Other platforms/devices, Flutter typing/UI behavior, hosted immutable publication/download readback and production activation remain outside this result. Raw Dart report/log originals remain outside ordinary Git with exact identities and a reproducible public redaction. No GitHub Actions endpoint was queried, awaited, rerun or dispatched.
