# Source-compatible physical vision prototype

Two default-off gates add a source-compatible vision path for the experimental
LightOnOCR3-4B physical-weight owner:

- `CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE=1`: fixed22-bit Pillow bicubic RGB
  resize and source normalization `(pixel/255-.5)/.5`, retaining the frontend's
  existing geometry admission.
- `CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION=1`: source F16 matrices with F32
  parameters, states and residuals. This requires the resize gate and verified
  physical ownership, OCR, image composition and vision-owner gates.

Source model arithmetic is derived from pinned llama.cpp
`6c01da6278ab6c69ed67d9124806b50d2b299ba7`. All198 sourceF32 parameters remain
F32;99 native matrices are rounded to their100 sourceF16 tensors, including the
temporal patch split. Explicit F16 im2col and paired SGEMM preserve source
operand precision without changing vendor traits. GGML operators retain their
source semantics, including internal GELU lookup precision. The runtime acquires
no Torch or MKL dependency. The current GGUF keeps protected vision storageF32;
text weights retain their experimental quantized recipe.

Execution is restricted to the existing fixed Linux x86_64 AVX2/FMA/F16C private
ABI profile, exact nativeGGML `3a27b6c665b191cb230be367db5c52d81dd3e2ac`,
four threads and still-image P16/T2/M2 geometry of at most1024 patches. Both the
build profile and hardware capability must pass. Requested unsupported F16
execution fails closed; main-library/WASM builds retain stubs. Gate-off and
nonphysical owners retain their existing behavior. The public factory continues
to reject experimentalformat6 and no application model alias is added.

[Retained qualification evidence](https://github.com/CrispStrobe/CrispEmbed/blob/efc8e24c5626076a22e977aff1523b16b124ddb9/docs/lighton4b-projector-attribution.md)
records model-free resize/synthetic/gate controls and the actual observed-form
component at sourceb59. All28 live upstream visual arrays match byte-for-byte,
and29 legacy anchors remain exact. This is one-image component evidence;
The complete composed form now emits167tokens/EOS and reproduces every retained
source-feature decode row. Character edits against the frozen BF16 output
reference fall259→175/656 and word edits47→45/90, versus actual upstream189/46.
This meets the observed form floor; broader reference checks remain required. Both gates remain off and the candidate remains unqualified. No
runtime release or quality promotion follows from this component result.

Reference tools and test executables are excluded from default builds and
production artifacts. The adapter map-gate setup failure is preserved separately;
its corrected metadata-only admission now precedes all weight transfer.

The [runtime pin amendment](lighton4b-source-vision-runtime-amendment-20261107.json)
records the tested driver transition while preserving all historical protocols.
