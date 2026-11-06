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
The processor config names `Qwen2VLImageProcessor`, while the preprocessor config
names `Qwen2VLImageProcessorFast`. This component reuses native torchvision-style
uint8 bicubic. Metadata alone does not resolve slow/fast selection. The actual retained
original capture and pixel replay below resolve that selection for the frozen
single-document case.

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

Hosted [41996964796](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41996964796)
passes34 controls at9de9011c2. Expanded
[41997341624](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41997341624)
passes45 controls at0d3ff8da6, including independently authored channel/time/merge
coordinates and the document's452×317→560-patch geometry. The small downloaded
receipts were independently reviewed and retained in
`tests/results/qwen35-ocr/lightonocr3-4b-image-compose.json`. These controls execute
preprocessing and synthetic image-token splicing, without model inference.

## Actual retained-original preprocessing replay

[Hosted41999219551](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41999219551)
at5f0ca990fa405d6430187488a873f3a9e0423289 authenticates the original BF16
capture at07543dcc80ebcecad644815d9397be2570a51dd4, its saved five input arrays,
the frozen `tests/regression/images/cc0/simple_form.png` bytes and decoded RGB,
then executes the real native processor. No checkpoint or model forward is used,
and the original is not rerun. The observed original processor is
`Qwen2VLImageProcessor`, backend `torchvision`, not an inference from metadata.

All860,160 F32 pixel values in[560,1536] match bit exactly: maximum/mean/RMS
absolute error0, cosine1, both absolute maxima1 and L2 magnitude831.4008051587742.
Native grid[1,20,28] and45 component controls also pass. The
[reviewed receipt](../tests/results/qwen35-ocr/lightonocr3-4b-preprocess-replay.json)
binds all source/image/input hashes and the small independently reviewed report.
The large arrays remain remote. Earliest numerical divergence for this image
lies downstream of preprocessing; vision/logits/decoded Q8/Q4 parity remains open.
Initial41998232116 and41998761159 failed provider dependency imports before any
input transfer or native execution; both unchanged negative receipts are retained.
The corrected job pinsKaggle2.2.2 and installs its undeclared`six` import.
