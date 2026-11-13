# Physical RoPE fixed panel: audited diagnostic — 12 November 2026

The [completed independent raw audit](../tests/results/qwen35-ocr/lighton4b-physical-rope-fixed-raw-audit.json), SHA-256 `d0bf3b45fa69e8197452a9846bd7f77dfddda28d93549a0c8a063e12aed68a28`, binds fixed producer `8001a29fab5abc3481e601168e10ca4383d10060ad7e4356a9110eb345c0b56a` and its completed outer receipt `e49ec255fddbb22faf8770b339fe0023d9b490437c778d7271d6d1e1c60e2091`. Runtime candidate `a8a22749ab0b757ae1f317b6fa5b84ffd148b939`, tested controls `855471a734c3188fcf6e83b93eeaac7ba6f101a51728ff838b5b04e6babf8d2e`, controller `a6164205620c71427a39c97ab386591406c4ca73` and [protocol](../tools/physical-rope-fixed-panel/protocol.json) remain exact. Auditor correction source is `d256247c99cce5ab9e9b81d1363c5b3d1c91ab6b`.

All six arms ran in the declared book-first order, including the complete capped-negative book OFF. Only physical RoPE changes OFF→ON; fused SwiGLU remains ON. Source references, provisional labels/scorer, cap1536/context2048, threads4, pixels, input seals and physical checkpoint are unchanged. A negative decoded result does not skip a later arm.

| Case | OFF → ON tokens / stop | OFF → ON content CER / WER | OFF → ON source floor |
|---|---|---|---|
| Two-column book | 1536/noEOS → 654/EOS | 1.2089902717 / 1.75 → 0.1764508554 / 0.2625899281 | false → true |
| Historical receipt | 465/EOS → 468/EOS | 0.03125 / 0.125 → 0.0325520833 / 0.1319444444 | true → false |
| Inventory table | 248/EOS → 248/EOS | 0 / 0 → 0 / 0 | true → true |

All three ON arms complete, but only two meet their unchanged provisional source floors. The receipt regression remains a blocker to broader quality claims. The book completion does not establish expert ground truth or portfolio qualification. All experimental gates remain default-off; no alias, release or application pin changes.

The hosted artifact-only auditor rehashed all 578 current producer-owned artifacts totaling 4,812,973,943 bytes and read every one of 3,619 archived vocabulary rows for SHA/finiteness/greedy argmax. It independently checked original source references, image bytes, exact seals, all historical OFF stage/row/text/stop identities, full decoded scorer outputs/floors, separate character/word edit counts and all 96 paired native layers. It also rehashed all 1,399 artifacts and 849 raw rows of the reused observed six-arm producer, all 2,249 historical fused baseline rows, and 84 other controls artifacts including the exact tested inference executable/library bundle. These 6,717 rows were read remotely; no raw vectors were downloaded locally for this audit. Every recomputed RMS/magnitude matches the producer exactly, within the unchanged eight-double-ULP portability allowance. Artifact-only analysis took296.518 seconds; this is not inference-performance evidence.

The earliest native OFF/ON hidden difference is layer3 in every fixed case. Source first-score errors retain the same mixed result:

| Case | OFF → ON RMS | OFF → ON maximum absolute error |
|---|---:|---:|
| Book | 0.1803049329 → 6.6325689539e-7 | 0.6972784996 → 3.8146972656e-6 |
| Receipt | 0.1221114823 → 0.2213878489 | 0.9809701443 → 1.3122177124 |
| Inventory | 0.1145709977 → 3.4307818525e-7 | 0.9309492111 → 1.9073486328e-6 |

These paired native boundaries are not original source hidden states. The remaining receipt difference requires separate attribution; the book component proof cannot explain it by itself. The earlier observed table/form regressions against original BF16 outputs remain binding.

The [initial failed artifact-audit receipt](../tests/results/qwen35-ocr/lighton4b-physical-rope-fixed-raw-audit-initial-failure.json) and [outer failure journal](../tests/results/qwen35-ocr/lighton4b-physical-rope-fixed-raw-audit-initial-bootstrap.json) remain separate and immutable. The first attempt stopped while hashing a controls outer-bootstrap snapshot captured before that wrapper finalized. The corrected artifact-only retry admits exactly the recorded312-byte intermediate hash and the retained355-byte final hash; it does not widen array, numerical, stop or quality gates and repeats no model/source forward.

Three provenance limitations remain explicit:

- Historical fused controls lost the original candidate `source.log` bytes when the worker wrote the controller revision. The linked prior audit journals retain that limitation; no missing original log is claimed rehashed.
- The separate fixed-panel metadata-preflight binary was not retained. Its recorded hash, exact candidate/source-file pins, retained CMake profile and actual sealed outputs are checked; its unavailable executable bytes are not independently rehashed. The retained tested inference bundle is independently rehashed.
- The controls intermediate outer-bootstrap snapshot was not retained. Its immutable recorded hash remains; the exact finalized wrapper is independently rehashed and journaled. All other controls artifacts keep strict size/hash checks.

The public receipt contains only counts, metrics and hashes. Book recognized text, token sequences and full model/raw reports stay private and remote. The audit performs no inference, model acquisition, conversion, build or source capture.
