# LightOn4B bounded native profiling

This separate source candidate instruments only the diagnostic document runner.
It reuses `CRISPEMBED_QWEN35_PROFILE=1`, existing thread-local counters and scoped
phase timing; production engine files and active Q8/Q4 producer sources remain
unchanged. Unset/empty/0 retains inactive timing. No speed gain is claimed.

The source hypotheses are repeated weight conversion in `project_cpu`, repeated
backend/context/graph creation, and970 streamed256-row head tiles per248320-row
vocabulary. The154-token prefill traverses the decoder in ten chunks of at most16,
so immutable matrix conversion can repeat before any generated token. During
single-token decode, every layer again enters the same uncached projections.
The existing converted-weight cache admits F32/F16 only, and the4B blocks do not
pass it. Quantized retention is a separate memory/arithmetic decision.

The runner adds phase-specific preprocessing, vision, prefill, cached decoder
and vocabulary-projection totals. Existing projection counters separate setup,
materialization, graph construction, compute and cleanup; matrix counters record
materialized source/F32 bytes. The head's outer `selection` timer includes raw
stream read/dequantization; nested tile timers see temporary F32 tiles, so their
matrix records must not be mislabeled as the original quantized head traffic.
Timing excludes diagnostic array writes at these coarse boundaries. Teacher and
greedy continuations aggregate under decode and their counts must be reported.

A later paired protocol must use one immutable candidate/image/prompt/thread
count and compare gate0/1 in fresh processes, with identical output IDs/text,
all logits/stage hashes and stops before accepting observer neutrality. Preserve
GNU-time user/system/wall/peak-RSS and source/model hashes, then report median
paired timings and instrumentation overhead. Counter phases overlap by design;
do not sum nested timers as wall time. Current smoke GNU-time is end-to-end with
array I/O, not a decoder-only measurement. After a measured bottleneck, prepare
one independently gated candidate (for example reusable graph/backend or bounded
quantized execution) and run the same paired outputs/resources. No full-panel
replacement or speed-based quality promotion is authorized by profiling alone.

## Existing smoke measurements, independently reviewed

The small `native-document.json` members were retrieved by bounded ZIP ranges
and CRC-checked, without downloading110MB stage archives. Both use source
fd150b98f488a6884a6e8d8127e7dfb263cab237,4 threads,154 prompt tokens,
140 image tokens, one teacher position and two generated tokens:

| Recipe/run | GNU wall seconds | User/system seconds | Peak RSS bytes |
| --- | ---: | ---: | ---: |
| Q4/41999584163 |133.35 |129.64/42.75 |4516601856 |
| Q8/41999580781 |127.69 |116.85/50.78 |6316945408 |

The [resource receipt](../tests/results/qwen35-ocr/lightonocr3-4b-smoke-resources.json)
binds original small-report hashes. These separate-host end-to-end observations
include file loading and stage output; they establish no phase bottleneck or
paired optimization improvement. Both capped smoke outputs remain distinct from
complete EOS/document quality.

Hosted model-free build42002026366 at4f2f4f679 passes actual diagnostic-runner
compilation, nine native-context controls and unchanged streamed projection
controls. The [build receipt](../tests/results/qwen35-ocr/lightonocr3-4b-profile-build.json)
retains executable hashes. This is compilation/control evidence; actual
profile0/1 output-neutrality/resource pairing remains pending.
