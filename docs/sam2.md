# SAM 2.1 image segmentation

Meta's [SAM 2.1](https://github.com/facebookresearch/sam2) (Apache-2.0) in image mode: one image, then
point and box prompts, masks out. The video-memory parts of SAM 2 are not implemented. Engine:
`src/sam2.cpp` (Hiera encoder with windowed and global attention and query pooling, FPN neck, prompt
encoder, two-way mask decoder); tested with Hiera-tiny (31M parameters).

## Use

```bash
# CLI: one mask (0/255 image on stdout); registry name or a GGUF path
crispembed --sam2 photo.png --sam2-model sam2.1-hiera-tiny \
    --sam2-box 453,288,1305,1132 --sam2-point 888,884 --sam2-point 878,116:0 > mask.png

# HTTP: crispembed-server --sam2-model sam2.1-hiera-tiny
curl -X POST localhost:8080/sam2/segment -d '{"image": "photo.png", "points": [[888, 884, 1]], "box": [453, 288, 1305, 1132]}'
```

Points and boxes are in image pixels; label 1 is object, 0 background. Without `--sam2-multimask`
(`"multimask": true`) the result is SAM's single mask, or its best alternative when the single mask is
unstable (SAM 2's dynamic fallback); with it, the three alternatives with their predicted IoU.

C API (`crispembed.h`): `crispembed_sam2_init / _set_image / _set_image_f32 / _predict / _process /
_free_masks / _free`. `_predict` returns the raw 4 x 256 x 256 logits and 4 scores for prompts in the
1024 frame (box corners first, labels 2 and 3), for callers that do their own selection. Bindings:
`CrispSam2` in Python, Rust (`crispembed`) and Dart. The engine also builds alone as the library
`crispembed-sam2` (`-DCRISPEMBED_SAM2_SHARED=ON` for a shared one), which is what an application that
only segments links.

Gates: `SAM2_FORCE_CPU=1` keeps everything on the CPU backend; `CRISPEMBED_SAM2_BENCH=1` prints
encoder and decoder times.

## Convert and verify

```bash
python models/convert-sam2-to-gguf.py --checkpoint sam2.1_hiera_tiny.pt --output sam2.1-hiera-tiny-f32.gguf
build/crispembed-quantize sam2.1-hiera-tiny-f32.gguf sam2.1-hiera-tiny-f16.gguf f16
python tools/dump_sam2_reference.py --sam2-source sam2/ --checkpoint sam2.1_hiera_tiny.pt \
    --image photo.png --box 453,288,1305,1132 --point 888,884 --point 878,116:0 --output ref.gguf
build/test-sam2-diff sam2.1-hiera-tiny-f16.gguf ref.gguf   # non-zero exit on any failed stage
```

The reference must come from PyTorch on the **CPU**: PyTorch 2.7 on Apple MPS computes the strided
query `max_pool2d` of the Hiera encoder wrongly (a contiguous copy fixes it), so MPS outputs differ
from the model's arithmetic from the first stage change on.

Parity on a 1749 x 1155 turntable photo, box and five points, CPU backend (`test-sam2-diff`: cosine
per stage, IoU of the four masks thresholded at 0):

| GGUF | Size | Encoder stages (cos) | Mask logits (cos) | Mask IoU (4 tokens) | Verdict |
| --- | --- | --- | --- | --- | --- |
| F32 | 125 MB | 1.000000 (max abs 2.6e-5) | 1.000000 | 1.000, 1.000, 1.000, 1.000 | reference |
| F16 | 63 MB | >= 0.999999 | 1.000000 | 1.000, 1.000, 1.000, 1.000 | lossless, recommended |
| Q8_0 | 34 MB | 0.9966 to 0.9999 | 0.99998 | 0.9992, 0.9996, 0.9895, 0.9986 | below the 0.999 gate |
| Q4_K | 18 MB | 0.78 to 0.99 | 0.998 | 0.994, 0.996, 0.922, 0.993 | not usable |

Hiera's narrow early stages and its global-attention blocks make the 8-bit activation quantization of
ggml's Q8_0 matrix products show more than usual; F16 is the size to ship.
