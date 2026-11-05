# LightOn4B image composition

The standalone component in `src/lighton4b_image_compose.h` composes the existing
Qwen image processor, configurable vision engine and multimodal positions with
4B's1024-wide24-layer/16-head vision and2560-wide decoder embedding geometry.
It reuses the PR146 owner configuration without editing historical runtime source
closures. Caller-owned payload lifetime and source admission remain separate.

The pinned source is `lightonai/LightOnOCR-3-4B` revision
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`; processor specifications are retained
in `tests/fixtures/lightonocr3-4b-source/processor_config.json` and
`preprocessor_config.json`. The current native1024-patch bound remains; original
max-pixel metadata is not a qualification of all original image sizes.

All paths require `CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE=1` and
`CRISPEMBED_QWEN35_OCR=1`; payload-backed vision additionally requires the owner
gate. New functions clear output on failure. Image-prefill splicing preserves
text embeddings, consumes exactly one image span and returns its three-axis
positions and RoPE delta. The caller must obtain embeddings from the admitted
native tokenizer and embedding table. No reference vectors are loaded.

The model-free hosted driver `tools/ci-heavy/lighton4b_image_compose.py` checks
actual resize/white normalization, every2560-wide splice value, explicit expected
positions, gate isolation and malformed/nonfinite inputs. It does not execute
model inference. Actual processor arrays, full vision stages and errors,
quantized decoder composition, complete logits and decoded EOS output remain
required before Q8/Q4 parity. Format5 recognition stays closed.
