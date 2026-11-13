# Independent aligned fixed-panel artifact audit

This hosted CPU auditor is prepared for the completed source-aligned fixed
producer. It makes no model, source-reference, decoder, build or network calls.
It remains blocked until the caller supplies the exact completed private
producer, terminal wrapper and safe-projection report hashes. Incomplete or
reordered arms fail admission. The declared order is receipt OFF/ON, book
OFF/ON, inventory OFF/ON; original labels, scorers, input seals, 1536 caps and
floors remain frozen.

Required retained inputs are the new fixed producer, successful aligned
controls, aligned observed producer, old RoPE fixed producer and the three
original fixed reference outputs. Existing public source supplies the frozen
old raw-audit receipt and reviewed observed safe evidence. No new source capture
or model acquisition is needed. All payload reads occur remotely; only the
safe audit JSON is collected.

The auditor:

- Rehashes every declared new fixed artifact, all 69 control artifacts and the
  observed inventory, including the retained quantized model and inference
  executable/libraries. It independently verifies the retained metadata probe,
  source pins and CMake profile.
- Checks all 916 reused observed vocabulary rows and all 1370 old RoPEON baseline
  rows for actual finite full-vocabulary argmax, whole archive/member hashes and
  exact ZIP census. The new six-arm row total is measured, not hardcoded.
- Recomputes all new raw rows, EOS/cap semantics, source/input/position seals,
  OFF IDs/text/rows/stages/summary closure and the single aligned-gate change.
  ON hidden layers may change; vision and decoder inputs must remain identical.
- Recomputes the original source scores, every fixed score/floor and separate
  bit-parallel edit counts; checks source ID/text equality without emitting the
  sequences. Paired 32-layer errors and first-score magnitudes use the existing
  eight-double-ULP RMS portability bound. Maximum errors, SHA identities, greedy
  argmaxes, stops and quality floors remain exact.
- Crosschecks the producer-side safe projection. A negative quality result is
  retained; it is never converted into a passing floor or omitted case.

The observed source-screen metrics are bound to the independently reviewed
public safe receipt; the auditor rehashes actual observed payloads and rows but
does not reread its original observed-source response captures. The earlier
56 source component comparisons are required by their immutable control receipt
and their payload inventory is rehashed; those source comparisons are not
recomputed. Original fixed references have their first score vectors checked,
not their entire historical decoded score inventories.

CLI: `audit.py --receipt … --receipt-sha … --bootstrap … --bootstrap-sha …
--safe … --safe-sha … --out …`. The exact inputs are bound only after producer
completion. The deadline is 2400 seconds; rows are processed one vocabulary
vector at a time. No dispatch is part of this tool. Small controls:
`python3 tools/source-aligned-fixed-audit/test_controls.py`.
