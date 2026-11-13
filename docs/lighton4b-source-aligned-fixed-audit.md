# Independent source-aligned fixed artifact audit

The [safe audit receipt](../tests/results/lighton4b/receipt-layer0-operators/aligned-fixed-independent-audit.json) passes for candidate `f8bffd1217fa41a958047d88dafb782a4b1cd989`. This was an artifact-only hosted CPU audit: no model/reference forward, native build or decoded rerun. The six fixed arms retain receipt OFF/ON, book OFF/ON, inventory OFF/ON order, original inputs, labels, seals, scorer, caps and provisional floors. Only the separately gated source-prefill planner differs between paired arms; RoPE and SwiGLU remain ON.

| Case | OFF generated IDs / edits | ON generated IDs / edits | ON source IDs/text | ON floor |
| --- | --- | --- | --- | --- |
| Receipt | 468 / 25 character, 19 word | 465 / 24 character, 18 word | Exact | Pass |
| Book | 654 / 526 character, 146 word | 654 / 526 character, 146 word | Exact | Pass |
| Inventory | 248 / 0 character, 0 word | 248 / 0 character, 0 word | Exact | Pass |

All six arms stop at EOS. Receipt first-score RMS against source falls from `0.22138784887772267` to `6.231868788194192e-7`. Its paired hidden arrays first differ at layer0; book and inventory remain identical through all32 paired layers. Source exactness is fidelity to the frozen quantized reference, not proof of broader gold quality.

The audit independently verifies actual sizes and SHA256 for all581 current artifacts (3,937,315,279 bytes), all1,467 observed artifacts, all69 source-control artifacts, the tested inference bundle, retained quantized model and separately built metadata-preflight executable/provenance. It checks2,737 new fixed vocabulary rows,1,370 prior-baseline rows and916 reused observed rows:5,023 complete248,320-F32 rows. Checks include finiteness, exact greedy argmax, row hashes, whole archives and exact archive census, stop/EOS contracts, OFF baseline closures and source/input/seal authentication. Frozen scoring and independent character/word edit recomputation preserve all floors. First-score and96 paired-layer metrics are recomputed; RMS discrepancies are0 double ULP under the predeclared8-ULP bound. The public safe projection is checked against the authenticated payloads and recomputed metrics.

| Identity | SHA256 |
| --- | --- |
| Full private fixed producer receipt | `91944ec7fcf2df0b23463c78fb6d80bc1ff0de616528da979051fbb15f1b35e3` |
| Fixed safe projection | `c9cf3f7bbd4fc5db34be9de505b25209f7e996fda1f36879c3af1d820f18a263` |
| Fixed outer receipt | `85eadb24afa32781749ee34f85f46b597cce256e34dbe28d95e6c30befda4bff` |
| Independent safe audit | `33a09075b0c614108f2392c6a1ec8c37da8543026ebceb8d73cf70856c5a5c02` |
| Independent audit bootstrap source | `54bae4a617a6b8b2e5c6b31b255b14a0bfc2959fd2a71eaa75ed096d31f753a3` |
| Independent audit outer receipt | `2a3a929a3958056e4c3904f807279c7ec069fee2008e3bd85eb3f520935d7037` |

The auditor is [source-bound tooling](../tools/source-aligned-fixed-audit/audit.py), with frozen [protocol](../tools/source-aligned-fixed-audit/protocol.json) and eight small admission/corruption controls. Its implementation was pinned at `c5ca828c9c428d2a0fb4a03ea6601fc9df9e7710`; the hosted audit completed in327.71seconds. That elapsed time is not inference-performance evidence.

The audit rehashes source-control payloads but does **not** independently recompute their56 source component comparisons. Reused observed source metrics are tied to the independently reviewed immutable small receipt; this audit does **not** reread the original observed-source responses. It reads the three original fixed source first-score vectors and source response records, **not** complete original source decode score inventories. Large arrays, model bytes, book text and token sequences remain remote; only safe JSON reports were collected.

The [earlier receipt attribution](lighton4b-receipt-layer0-attribution.md), prior receipt468/floor failure, capped book baseline, setup failures and original observed BF16 table348/97 and form189/46 edit regressions remain preserved. Passing all three unchanged quantized-source floors does not qualify broad decoded quality, change defaults, create a runtime release, update the app consumer or establish a speed win.
