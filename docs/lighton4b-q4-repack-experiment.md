# Physical Q4 repack: unqualified, keep disabled

`CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK` remains **default-off and unqualified**.
Do not enable it as a quality fix or use it for a release/model alias.

Same-input operator and lifecycle controls passed at source `5f6dfb9f0`: all30
real-input runtime OFF/ON outputs match their corresponding sealed raw/preferred
operator outputs bytewise, and the bounded cache/main-library controls pass. This did not establish decoded
parity. The complete same-feature table comparison at source
`f079dcec64a799a23d2bdcc25c0da0fee6c49d8c` is a measured regression:

| Measure | Gate OFF | Gate ON |
|---|---:|---:|
| Generated IDs / completion |282 / EOS|295 / EOS|
| Character edits vs original BF16 (607 codepoints) |349|365|
| Word edits vs original BF16 (106 words) |97|104|
| Character / word edits vs actual upstream |9 /7|19 /17|
| First-score RMS vs actual upstream |0.188629|0.202156|
| Child wall seconds |178.576|181.195|
| Observed peak RSS bytes |2,958,786,560|4,632,313,856|
| Additional quant cache entries / bytes |0 /0|216 /1,662,861,312|
| Converted F32 weight entries / bytes |0 /0|0 /0|

OFF matched all282 frozen raw score rows,38 prefill stages, IDs, text and EOS
before ON started. Weights, retained upstream features, prompt, F16 attention,
Q6 head, context2048, cap512 and all other gates stayed fixed. The actual
upstream reference has348/607 character edits and97/106 word edits against the
same BF16 reference. Timing is OFF-first/ON-second with different token counts,
loading/prefill/score-write costs; it is not a balanced speed claim.

First measured native OFF/ON hidden drift is layer0 (RMS3.26613e-5,
maximum0.000956059), growing to layer31 RMS0.0762301/maximum3.13368. Actual
upstream text hidden stages were not captured, so this does not locate its
earliest internal divergence. It is a source-feature-only diagnostic with one
teacher row and full independent greedy output, not composed-vision or held-out
qualification.

[Full raw evidence and independent review](https://github.com/CrispStrobe/CrispEmbed/blob/1d48ac73c/docs/lighton4b-q4-repack-operator-results.md)
retain the initial adapter compile failure, successful controls, immutable
weights/source/receipt hashes, raw text/IDs, cache accounting and resource limits.
The model receipt SHA256 is
`5acbdab91bf45752d06faf712856cb0eab82ab116657fc43e590166db4e03d48`.

Keep the alternative for diagnosis. Prioritize the actual composed-vision
preprocessing/stage discrepancy; require source-stage evidence for further text
changes and all observed/composed/held-out gates before any promotion. Global
vendor traits, public format fencing, defaults and application release pins are
unchanged.
