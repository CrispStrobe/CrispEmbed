# LightOn4B composed table and Arabic: report review

The completed observed-case worker report has SHA-256
`f69d23ffe6d65d9abf6c7c5c660723613ce84cc901383c5151e17d8f8dc5fe64`.
Its executor source is `a03412aeb3b4df9c8a5a9f093d9ab699b9d41258`, using
the already tested runtime producer `b98567cfd46aaf594f33076aaab4ff8e4610edbb`
and form receipt `aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b`.
The physical candidate SHA-256 is
`843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.
The worker reports all cases authenticated before weights, the frozen gate
profile, no runtime rebuild and no new upstream or original forward.

Both observed cases reproduced their frozen source-feature baseline and then
matched that baseline in the pixels-only composed arm, including merged feature
bytes, decoded output, complete greedy rows and the recorded stage controls.
Both arms reached EOS: 282 tokens for the table and six for the Arabic line.
The derived [receipt](../tests/results/lighton4b-composed-calibration-review-2026-11-07.json)
preserves decoded text, token IDs, reported metrics, runtime bundle hashes and
all 897 artifact hashes. This review read the small worker report and checked its
internal consistency; it did **not** download or independently rehash the raw
stage or logit payloads. The receipt is a filtered derivation, not the original
worker report, whose hash is recorded above.

| Observed case | Composed versus source-feature | Versus original BF16 output | Frozen upstream floor |
|---|---|---|---|
| Simple table | Merged features byte-exact; 282 generated IDs, text, greedy rows and recorded stages exact | 349/607 character edits; 97/106 word edits | **Missed:** upstream 348/607 characters, 97/106 words |
| Arabic printed line | Merged features byte-exact; six generated IDs, text, greedy rows and recorded stages exact | 0/12 character edits; 0/2 word edits | Met; upstream also exact |

For context, the table first-score comparison with actual upstream has RMS
0.188629 and maximum absolute difference 1.061775 (upstream maximum absolute
20.541553); Arabic has RMS 0.109236 and maximum absolute difference 0.691092
(upstream maximum absolute 18.174143). These are not same-weight attribution or
quality proof. The original BF16 model output is a reference, not adjudicated
human ground truth. This is a two-case observed panel, not held-out coverage;
retain the failed table floor and both default-off gates. Independent payload
rehashing, representative held-out comparisons and broader qualification remain
before any promotion.
