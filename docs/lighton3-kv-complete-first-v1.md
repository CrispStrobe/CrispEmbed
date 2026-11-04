# LightOn3 cache A/B: preserve ten calls, complete two

[Completion41901879295](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41901879295)
**passes**. It reuses ten authenticated native results and executes only the two
missing F16 free processes, with zero original calls. The
[reviewed result](../tests/results/lightonocr3-1b/cache-ab-v1/completed-first-review-41901879295.json)
records4 rows/12 total calls, both formats' exact original text and IDs, preserved
old-cache negatives and unchanged strict/cap flags. This is one case, not a
full-panel/default/publication decision.

## Actual correction and evidence

The first [cache run41897136488](../tests/results/lightonocr3-1b/cache-ab-v1/initial-prefill-classification-failure-41897136488.json)
completed ten native calls, then misclassified `generated_ids.f32` as a prefill
activation. That callback contains postdecode argmax IDs, which should change
when the defective path is corrected. This result remains a failed historical
run. Its original protocol/driver/owned-source guards are not edited or repinned.

The separate [completion protocol](lighton3-kv-complete-first-v1.json) corrects
only the phase classification: prefill excludes generated IDs and logits after
step0. Every other prefill file, including step0 logits, must remain bit-exact.
Actual output IDs/text, both free arms and every full-vocabulary forced logit
still pass the unchanged original comparison; no decoded-output guard is waived.

The original ten executions and their receipts are copied without rewriting.
Both KV0 tensor inventories are independently rehashed against the successful
stored replay proofs, including their postdecode arrays. The known French F16
negative remains negative. Only F16/KV1 autonomous/plain directories may be new.

| Format | Step6 max error, F16 cache | Step6 max error, F32 cache | Corrected minimum final cosine |
|---|---:|---:|---:|
| F32 |0.0129647255|0.0001430511|0.999999999939971|
| F16 |0.0133957267|0.0001006126|0.9999999999249204|

The original2918-versus3377 logit margin is0.0024375916. Old-cache F16 reverses it
to-0.0006361008, choosing French; F32-cache F16 restores0.0024681091 and the exact
English original. Both candidate free outputs match, with strict28/38 stage
failures and the original128-token cap still retained. A selected-margin probe
accompanies complete comparisons; it is not a standalone qualification gate.

## Native and operator identity

The controller is source `fc8ce8d2de1d9bc89790c0e6892a52fd51a2c1cf`;
the native checkout/build remains exactly
`3e82a12058cfca8869f4aa8ead250acf2b708934`. Its executable is reproduced byte for
byte, SHA256 `7d4b027feb14b51261b36efbb3efc6cea73b20ee81aa8bd3bbf85976fea4dfdd`,
before the two calls. Existing comparer requirements for identical model,
binary and native-source identity remain enforced across all three arms.
No receipt is relabelled with the controller source.

The workflow copies its committed controller/protocol outside the checkout,
then checks out the frozen native source at the original working/build paths.
The controller verifies its copied bytes against the operator's Git blobs and
all native owned/protected objects against the frozen checkout. A binary mismatch
must stop before inference. The diagnostic observer executable is preserved only
in the raw evidence artifact; it is excluded from production releases.

Small artifact13359865825 is503653 bytes, SHA256
`e194a312cb84c17cf7a702cd585e2d202fbca45eb451ff80198a168140f1d6fb`.
Raw artifact13358642574 is1462821808 bytes, SHA256
`58e3b03b2022046e611146836e6c53ed8ce98cfc0bb5e3dd1c342c0c6513e6b1`.
Both retain90d. The small ZIP, counters,4 unique rows, source/binary proof,
prefill controls and corrected decoded outputs were independently reviewed.

## Remaining five cases

[The source-closed remaining protocol](lighton3-kv-remaining-v1.json) and
`tools/ci-heavy/lighton3_kv_remaining.py` require this exact positive first proof
before acquisition or native calls. Actual
[41905258340](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41905258340)
at558825720 is running. It imports these4 rows/12 calls and executes60 new
processes for the other5 cases. No first-case repetition or new original capture.

This first-only completion selector is `tools/ci-heavy/lighton3_kv_complete_first.py`,
empty args/pip, ubuntu-latest/portable. It must not be rerun without a justified
new concern. Query the remaining run instead. Its final24-row/72-call result,
all strict/cap/decoded flags and exact old-path byte controls are required before
any subsequent default decision. Both correction gates remain default-off.
