# Atomic LightOn4B tensor ownership

The standalone `map_lighton4b_model` component combines the exact PR145 text
owner (`882b8dded`) with unchanged PR144 vision source/header (`84e19504e`).
It requires exactly723 borrowed tensor headers:426 text and297 vision. The
caller retains all tensor/backend lifetimes. The mapper never reads payloads,
opens a GGUF file, verifies a checkpoint identity or admits format5 recognition.
All four gates must be enabled: `CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER`,
`CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER`, `CRISPEMBED_LIGHTON4B_VISION_OWNER`
and `CRISPEMBED_QWEN35_OCR`. Every gate stays default off.

The atomic wrapper rejects global pointer aliases, noncontiguous layouts,
unknown prefixes and partial inventories before returning either component.
The component mappers require exact names/shapes; protected norms, biases and
recurrent parameters stay F32. Supported matrix headers can be F32/F16. No
quantized profile is admitted. Failure clears both returned components.

Hosted [41973787198](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41973787198)
at `8ea8f4d03f7bbdf909986548dc9df2d5de7ab213` passes5281 metadata-only controls.
The fixture derives independently from all723 entries in the frozen original
header, with only the converter's declared Conv3D-to-matrix flattening. Controls
include all723 entries individually missing, four-axis drift, unsupported type
and bad strides; all197 protected vision vectors reject F16. Further checks
exercise unset/empty/zero for each gate, same-shape and cross-component aliases,
unknown prefixes, a stored tied-head alias and the32/24-layer returned geometry.
Separate protocol controls also preserve17 GDN,6 frozen replay and13 larger-input
checks. The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-model-owner.json)
verifies every recorded source digest, schema hash/census, inherited component
identity and independently recomputes5281 controls from the test structure.

Initial [41973459077](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41973459077)
compiled but failed its test expectation for literal `false`. The shared helper
only treats unset, empty and literal `0` as off; the test was corrected to that
existing contract. No gate semantics or historical receipt was changed.

This is borrowed metadata ownership evidence, without weight allocation or
inference. Full payload-backed admission/owner lifetime, original/native stage
arrays with magnitudes/errors, complete logits, tokenizer/preprocessing, decoded
outputs and EOS remain open. Grouped execution and the separately gated streamed
projection still require composed integration. Uploaded conversion does not
qualify this native path. No model-support claim, alias, release or default flip.
