# Retained Q8_K workspace causal replay

[Run40780061239](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40780061239)
completed the frozen protocol at
`9991426b9e3296def3457340a989e841a46a6383`. The
[independent review receipt](../tests/results/packaging/wasm-q8k-cross-v1.json)
records the exact service identities, all33 small-artifact member hashes,
18 original byte anchors and44 actual `crispembed-diff::Ref` comparisons.
The job succeeded because the intervention protocol completed and retained its
negative comparisons. It does not qualify the whole model.

The replay uses the retained native and browser GELU inputs to the generic
BERT/MiniLM layer3 down projection, `blk.3.ffn_down.weight`. Its unchanged
IQ4_XS coefficients are tensor type23, GGML shape `[1536,384]`,313344 bytes,
SHA256 `998ab6add261d232b8bc6fe0fb97b4bc0cba2a7eda5c79130ef20e9f0d3135f5`.
The selected activation conversion is Q8_K/type15. Each retained workspace is
21024 bytes for12 tokens. This is the standard GELU FFN in
[crispembed.cpp](../src/crispembed.cpp), not the PosFormer OCR engine.

All18 normal/direct output anchors reproduce their original bytes. Regenerated
workspace hashes also match the saved captures. All eight outputs agree
byte-for-byte between native one and two threads; repeats, invalid-input
rejection, unchanged failure outputs and recovery pass. The browser reports
cross-origin isolation, shared memory/pthreads and no page errors. Actual native
and WASM compilation profiles were checked against the frozen profiles.

The44 Ref comparisons agree with the independent hosted NumPy metrics:
26 pass and18 reject, with12 byte-exact comparisons. All44 minimum row cosines
exceed0.99999; all18 rejections exceed the unchanged0.002 absolute-error gate.
The table compares each intervention with that runtime's native-input workspace
baseline. Integer fields travel with their matching block sums to retain valid
Q8_K workspaces.

| Admitted workspace | Native maximum / RMS error | Browser maximum / RMS error | Gate |
| --- | --- | --- | --- |
| Entire browser-input workspace | 0.003310918808 /0.000279435246 | 0.003311038017 /0.000279435720 | Reject |
| Native scales, browser integers and matching sums | 0.003192067146 /0.000280372194 | 0.003192067146 /0.000280371643 | Reject |
| Browser scales, native integers and matching sums | 0.000418305397 /0.000019474981 | 0.000418186188 /0.000019473678 | Pass |

For every identical admitted workspace, native versus browser output errors are
at most9.5367431640625e-7. Changing the validated floating-input metadata while
retaining a workspace leaves its output byte-exact: the replay's direct-workspace
arm consumes the workspace for arithmetic. Thus the workspace intervention
reproduces the natural projection jump inside either runtime.

Across72 activation blocks,38 F32 scale bit patterns differ; signed-integer
hashes and matching-sum hashes differ in two blocks, token5/blocks3 and5
(zero-based). The small report identifies these blocks but does not expose the
changed signed-byte positions or their count within each block. Both pinned
generic and x86 IQ4_XS dot implementations read the scale and signed integers;
they do not read Q8_K block sums. These results do not attribute an independent
effect to the sums.

This establishes local activation-quantization amplification for the fixed
matrix and two retained natural inputs. It does not identify the unique upstream
origin of their small floating-point drift. The earlier
[same-input F32 replay](wasm-f32-operators-result-v1.md) bounds local discrepancies
at its captured inputs; small participating-operation differences can still
propagate across activation-quantization thresholds. No model forward, original
rerun, production modification or new golden reference occurred here. The
earlier whole-model negative and all its gates remain in force.

Only the52420-byte review ZIP was downloaded for independent review
(SHA256 `6b905f156a2bc7b65b79fbd48af9e6c7d63406262a5b9102789be4168cfb2e0c`).
The6428177-byte raw artifact12859228520 remains remote, SHA256
`e7f834648d39a45a85e15700438aa459241575a5bab2431c038315efa0f376b6`,
expiring26 January2027. Independent review checked source, report identities,
anchor cross-links, metric agreement, magnitudes and saved channel0 values;
it did not recompute full raw-array metrics locally. The receipt retains all44
comparisons and the original producer report hash.
