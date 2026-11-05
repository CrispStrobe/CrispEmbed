# Predeclared LightOn4B mixed-precision diagnostic

[Protocolv1](lighton4b-mixed-calibration-v1.json) freezes an architecture-only
candidate before any new panel outputs. Every decoder attention/linear-mixer
matrix uses the retained actual Q8 payload; FFN matrices retain plain Q4_K.
The existing shared Q8 embedding/tied head and F32 vision/scalars remain exact.
This is named `lighton4b-q4k-mixers-q8-v1`, never Q4_K_M. No layer is selected
from the previously observed form, late-layer error peaks or heldout outcomes.

Calibration uses CC0 simple-table and Arabic-line images. Heldout uses separately
frozen CC0 receipt and letter images with declared deterministic resize rules.
All source hashes, public license/source paths, prompt, cap256, EOS, expected
processor geometry, selectors and resource guards are frozen before execution.
A hosted model-free preparation run first emits rendered PNG/RGB hashes. Those
must be independently reviewed and committed before any model forward. Official
processor input hashes/grid checks precede model loading. Original references
use BF16; no F32 original is required.

The old form, both plain candidates and their running full captures remain
unchanged. New panel comparisons require every original-position score plus
independent decoded IDs/text/EOS. Calibration may reject this fixed candidate;
it may not tune v1, its thresholds or heldout cases. Another recipe requires a
new predeclared protocol and fresh heldout data. This small panel cannot qualify
representative task quality, model publication, defaults or an application pin.

Actual [model-free input freeze42004846562](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42004846562)
at `98e4b7d71` prepares all four cases without a model. Independent review
rehashes every PNG and rawRGB output, preserving the exact producer manifest.
The [frozen inputs](../tests/regression/lighton4b-mixed-v1/frozen-inputs.json)
and small opaque RGB PNGs are committed before any new reference/candidate
forward. Model workers decode these exact PNGs and authenticate RGB SHA,
expected grid and token count before model load; they do not rerender/rescale
source images. RawRGB and large future arrays remain remote.

The [packing contract](lighton4b-mixed-packing-v1.json) selects152 attention/
linear-mixer Q8 matrices,96 FFN Q4_K matrices, the already-shared Q8 embedding/
tiedhead and474 protectedF32 tensors. Exact authenticated source payloads are
copied with8MiB streaming chunks; no weights are dequantized, requantized or fit.
Candidate payload is4,673,030,144 bytes,652,083,200 bytes larger than plainQ4_K.
The canonical base file-type14 (`MOSTLY_Q4_K_S`) remains, with explicit custom
`qwen35_ocr.mixed_quant_recipe`; it is never relabeled Q4_K_M15. A separate
runtime opt-in and exact physical type policy are required.

Before transfers, hosted packaging runs synthetic byte-preservation/tamper
controls and checks load, RAM, swap and24GiB scratch (plus12GiB output when on
a separate filesystem). It verifies both full immutable input hashes and every
selected tensor hash, then reopens and rehashes all723 written payloads. Source
models stay in hosted scratch; only the new candidate and small audit are
retained remotely. This is artifact feasibility, not decoded parity proof.
