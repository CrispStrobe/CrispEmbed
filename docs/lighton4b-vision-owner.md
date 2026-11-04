# LightOn4B vision ownership component

The explicit `map_lighton4b_vision` API maps the converter's `v.*` names into
`vision_weights`, requiring exactly 297 vision tensors: nine patch/position/merger
parameters and 24 blocks of twelve parameters. Text tensors can coexist in the
same map; they are not validated by this component. Both
`CRISPEMBED_QWEN35_OCR=1` and `CRISPEMBED_LIGHTON4B_VISION_OWNER=1` are required.
The old `map_small_vision` path is unchanged.

The source geometry is LightOnOCR-3-4B at
[`b71010f095dc3735de7ee6b9969bff8f8f50b39a`](https://huggingface.co/lightonai/LightOnOCR-3-4B/tree/b71010f095dc3735de7ee6b9969bff8f8f50b39a):
width1024, 16 heads, FFN4096, output2560, 24 layers. The returned configuration
carries these dimensions into existing `vision_cpu`; the mapper does not execute
inference, load a model or establish checkpoint identity. The caller must retain
the GGUF weight owner for the lifetime of every returned pointer and must apply
the separately pinned model/profile admission checks before inference.

Validation checks every shape and permits only F32/F16 tensors. The flattened
patch projection is `[1024,1536]` in exported row/column order; an unflattened
Conv3D source tensor is rejected. Missing/extra vision names, DeepStack entries,
integer/quantized types and nonunit trailing dimensions fail closed, leaving
empty weights and the legacy default configuration. No payload access occurs.

`test-lighton4b-vision-owner` uses metadata-only headers, checks every tensor's
shape/type/missing rejection and F16 acceptance, plus both gate and extra-name
controls. It runs in ordinary hosted model-free CI. This is ownership evidence
only: all723 tensor validation, loaded-payload provenance, original/native vision
arrays and magnitudes/errors, full logits, decoded text/EOS and tokenizer parity
remain required. Format5 production admission stays closed; no alias, precision
qualification or release follows from this component.

The initial ordinary Build
[41926266003](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41926266003)
at `0c1452d48` failed on an audit-copy duplicate symbol and the standalone test's
iOS target placement. Corrected `fea7fc72f` renames the audit-copy symbol and
places the target alongside existing non-iOS validation executables.
Full Build dispatch
[41927088394](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41927088394)
has no mode input: dispatch selects the full matrix. Its actual
[Linux job](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41927088394/job/126914390596)
compiles the target and runs it successfully, printing `PASS LightOn4B vision
ownership, 297 tensors and per-tensor malformed controls`. This proves the
metadata-only ownership controls, not model inference. The full dispatch also passes Windows/macOS, iOS, all3 Android legs, sibling,
Flutter bundle and the aggregate Build result.

Contract [41928007006](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41928007006)
subsequently rejected historical vision source drift. The additive mapper now
lives in `src/lighton4b_vision_owner.cpp`; `src/qwen35_vision.cpp` is restored
byte-for-byte to the main implementation. Historical receipts and source-pin
chains are unchanged. Fresh hosted contracts/builds must validate this source
placement correction; the earlier passing matrix remains source-specific.
