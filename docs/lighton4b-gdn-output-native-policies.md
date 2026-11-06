# Default-off LightOn4B GDN output policies

The native owner can admit two exact named24-output policies in addition to the
preserved plainQ4, plainQ8, mixed-attention and32-Q6 policies. Conflicting profile
gates, unknown markers, wrong tensor types, changed tied heads and incorrect
source/selection/count/delta metadata reject before allocating model payloads.

- `lighton4b-q4k-gdn-out-q8-v1` requires `CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8=1`.
  It retains the verified24-Q8-output donor recipe and exact original metadata.
- `lighton4b-q4k-gdn-out-q6-v1` requires both `CRISPEMBED_LIGHTON4B_Q6_K=1` and
  `CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6=1`. Its [exact manifest](lighton4b-gdn-out-q6-v1.json)
  selects recurrent GDN output weights only, directly from original BF16. The
  remaining224 matrices are Q4; the tied embedding/head stays Q8. It is a custom
  policy, separate from the preserved upstream-style32-Q6 recipe.

The integrated owner, metadata helpers, preflight and two C++ tests are byte-exact
copies of tested source[`10d3fad88`](https://github.com/CrispStrobe/CrispEmbed/tree/10d3fad88f59c787dee855a564bfd8fed57962d0).
[Model-free42218056668](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42218056668)
passes700 new Q6-output checks,112 Q8-output checks and713 existing32-Q6 checks,
plus native/mixed/BF16 controls and actual native/preflight builds. The new Q6
fixture executes actual grouped output projection for prefill and cached decode,
compares independently dequantized weights, retains serialized bytes and tests
explicit-off/profile conflicts. Tests are `test-lighton4b-gdn-out-q6-recipe` (426
text tensor headers generated from the manifest) and `test-lighton4b-gdn-out-recipe`.
The separate source also passes actual723-payload admission and pinned same-worker
Q4/Q8-output first-score controls in
[42219710049](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42219710049).

The [reproduction tools](lighton4b-q6-conversion-and-calibration.md) now include the
streaming source converter and closed experimental harnesses; their helper is
excluded from default builds. Production artifacts gain no reference probe,
model alias, default promotion or release. Q6 matches Q8-output full calibration
IDs/text/EOS on both cases; the table still differs from original BF16, and the
observed logo remains `A`. All six strict first-score failures and Arabic
probability regression remain preserved. Evidence is maintained separately in
PR157; no heldout-quality qualification follows.
