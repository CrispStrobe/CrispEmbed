# PCS frozen failure attribution — 18 November 2026

Production PCS, original driver, corpus, reference and acceptance are unchanged.
Main's failure remains open; no fix commit, fullstop or green-main claim exists.

| Run | Actual result |
| --- | --- |
| [44272906599](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44272906599) | Compiled native original/diagnostic OFF/ON: all four dumps and decoded bytes exact; Intel positive67/67 and6/6 |
| [44273793050](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44273793050) | Compiled AVX2 exact failure: casing66/67, decoded5/6; original ONNX observer-neutral and all original heads match frozen reference |
| [44274761438](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44274761438) | Same-weight F32 matrix alternative rejected: casing66/67, decoded5/6;13 stage pairs retained |
| [44276100741](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44276100741) | Initial independent replay stopped before inference on an omitted singleton embedding axis |
| [44276378964](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44276378964) | All214 stored tensors mapped into original graph; original collector also produces casing66/67, decoded5/6; execution success is not quality success |

[Native control](frozen-native-control-20261118.json),
[AVX2 failure](frozen-avx2-failure-20261118.json),
[rejected F32 control](rejected-f32-matrix-control-20261118.json), and
[original-graph stored-weight blocker](original-graph-stored-weight-blocker-20261118.json)
retain exact sources/model/hash/ISA, all16 row37 logits, alignment, conditioning,
artifact identities and numerical limits. The independent replay's original
execution status was success despite collecting quality negatives; the corrected
report-only verifier returns1 on those unchanged results. No model rerun is needed.

The immutable stored-weight set itself is quality-negative independently of PCS
runtime. The exact offending tensor, conversion operation and conversion commit
are not proven. No artifact repair/regeneration or new owner is authorized by this
handoff. Artifact/model remediation requires separate scope; existing raw receipts
support bounded review. Preserve every positive and negative boundary.


The [final checkpoint handoff](../../../docs/pcs-final-handoff.md) records the
separately authorized operator continuation and its narrower qualification limits.
Its missing casing-projection payloads do not change this canonical stored-weight
replay result, and its scoped QK contribution is not a PCS repair. No repeated
execution or audit is needed to preserve these existing receipts.
