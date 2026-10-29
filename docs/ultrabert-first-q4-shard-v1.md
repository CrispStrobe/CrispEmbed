# UltraBERT conservative Q4 model first, reviewed shard0 second

This is a separately closed caller layered over immutable018/37/08 sources.
The [completed F32 freeze](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834743025)
and [independent review](../tests/results/ultrabert/retained-freeze-review.json)
are mandatory immutable prerequisites. The old72-file producer, all original
references, saved F32 streams, runtime018 and old guards are unchanged. There
are no new original or F32 forwards and no whole-panel invocation.

The [new workflow](../.github/workflows/ultrabert-first-q4-shard-v1.yml) has exactly
two choices and one job, with six-hour/two-CPU,14GiB available RAM/24GiB disk
admission, root LLAMAFILE ON/native OFF and the same numerical packages as the
successful freeze. It preserves small reports, payload and exact binaries
remotely90 days, including partial failure journals. Use a dispatch ref matching
the reviewed `source_ref`; current source/service and both old/new closures are
checked before any model access. No automatic matrix or shard1–7/aggregate path
exists in this caller.

1. **Model conversion, zero forwards.** Authenticate the whole frozen payload
   ZIP132,645,918 bytes/SHA and every77-member manifest entry, exact49,048-byte
   freeze report and reviewed source. The converter then uses the exact pinned
   original checkpoint/metadata, reproduces609,437,696-byte F32 SHA
   `89478b9cbbad2fd1121be4031d760f1529c989fedf23b6b19251ba2c87422566`,
   and runs the unchanged018 C quantizer with the existing
   `q4k-last-four-ffn` overrides. Inventory must retain218 tensor names/shapes,
   twelve genuine Q4_K FFN gate/up/down tensors in layers20–23 and206 F32 tensors
   with unchanged critical payload bytes. The new report records the actual
   model whole SHA/bytes, inventory, quantizer/source/build identities and freeze
   receipt. No native model call occurs here.
2. **Only reviewed shard0.** Model artifact/report/whole-model SHA and explicit
   shard0 activation start **pending**. They cannot be filled from a latest
   artifact or inferred from recipe names. Root review of the conversion must
   freeze them before any quantized forward. The selected rows are exactly
   `range(0,5483,8)`:686 original query/document IDs, text and expected token
   streams. Both raw and normalized public calls use the old source/body and
   exact four default-off gates. Unset, empty and literal0 quant-gate rejections
   precede gate1. Every finite768-F32 raw/normalized pair is fsynced before its
   input/model/library/comparator/source-bound journal entry commits. A reviewed
   partial artifact may resume only exact authenticated committed rows; corrupt,
   torn, reordered, mismatched or extra rows reject. No healthy row is silently
   recomputed. The five-hour in-process boundary and2GiB output reserve remain.

The conversion body is byte-identical to08b. The shard body changes only explicit
new-controller resume identity and the extra approved whole-model SHA check;
math, inputs, gates, limits and journal helpers remain identical. The actual
source/library/comparator and controller identities are bound separately.

Actual Ref and independent raw/normalized metrics retain unchanged0.002 absolute
and0.99999 row-cosine flags. Quantized drift is reported separately; structural
job completion is not a strict-vector pass. Shard0 is not all5,483 rows or all300
rankings and cannot qualify Q4 retrieval, release, a model publication or a
default. Existing full300 ranking gates remain≤0.001 NDCG/MRR/Recall degradation
and≥0.99 top1 agreement for a later separately reviewed complete panel.

The frozen018 quantizer emits legacy file-type12 metadata. This caller records
that fact and makes no standard-header publication claim. It cannot relabel Q8
as Q4, activate Q8, switch to a newer runtime or rewrite old references. Q8 and
standard-metadata admission need their own explicit source/protocol. Source
preparation uses only stdlib checks locally; all payloads, builds and inference
remain hosted. No dispatch is part of this preparation.
