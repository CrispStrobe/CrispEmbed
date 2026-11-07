# Physical GDN source operators: unqualified decoded negative

**Keep `CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS` default-off. Do not ship or promote this path.** The gated implementation remains available for diagnosis; exact source-fed operator controls do not imply decoded parity.

The [full table/form A/B evidence](https://github.com/CrispStrobe/CrispEmbed/blob/e112858b6/docs/lighton4b-gdn-full-ab-results.md) preserves actual source `f6a1e60d3`, runtime `cd50e9e69`, and receipt SHA256 `40b20127fe0a0ffbbbc227f40a847a31a5ac6c0c8c231edff24aa17ce1c6cdc5`. All four independent greedy arms reach EOS. OFF reproduces all282 table/167 form raw rows, all38 text anchors and immutable vision/input bytes before ON; only the GDN flag changes.

Table worsens from349 to364/607 character edits and97 to104/106 word edits against the original BF16 text; actual upstream remains348/607 and97/106. Form stays exactly167 tokens and identical raw text, with175/656 and45/90 edits. First-score RMS against upstream improves on both cases, which does not override the table decoded regression. Table child wall time is210.16→234.78 seconds; form143.72→173.25. These are diagnostic timings, not a speed claim.

Global Q4 repack remains off. No vendor traits, public loading fence, model alias, app pin or release default changes. Held-out cases were not rerun; their historical-receipt failure remains binding. The paired native stages first differ at layer0; this is not an upstream hidden-stage attribution. Large raw rows and the exact binary bundle remain remote; selected small outputs and an independent edit-count/hash review are public in the linked evidence.
