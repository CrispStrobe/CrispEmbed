# Real-ESRGAN Super-Resolution

Whole-image super-resolution based on Real-ESRGAN's SRVGGNetCompact architecture. Upscales images 4x with ~620K parameters.

## Architecture

Single-path VGG-style design: 17 Conv3x3+PReLU layers + PixelShuffle x4 upsampling.

The body consists of repeated Conv3x3 -> PReLU blocks operating at the input resolution, followed by a final PixelShuffle x4 sub-pixel convolution that produces the high-resolution output. No skip connections or attention — pure feed-forward simplicity optimized for real-time inference.

## Model

- **Source**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (realesr-general-x4v3)
- **License**: BSD-3-Clause
- **Parameters**: ~620K (SRVGGNetCompact)
- **GGUF**: `/mnt/storage/gguf-models/esrgan-x4-f32.gguf`
- **Converter**: `models/convert-esrgan-to-gguf.py`

## Parity

Reference vs C++ engine: cosine similarity = 1.000000 (exact float parity on F32 GGUF).

## C API

```c
#include "crispembed.h"

void * ctx = crispembed_esrgan_sr_init("esrgan-x4.gguf", 4);
int scale = crispembed_esrgan_sr_scale(ctx);  // 4

uint8_t * out = NULL;
int ow, oh;
crispembed_esrgan_sr_process(ctx, rgb_pixels, w, h, 0, 0, &out, &ow, &oh);
// out is w*scale x h*scale x 3 RGB
crispembed_esrgan_sr_free_image(out);
crispembed_esrgan_sr_free(ctx);
```

## CLI

```bash
crispembed --esrgan-model esrgan-x4.gguf --esrgan-sr input.png > upscaled.ppm
```

## Python

```python
from crispembed import CrispEsrganSr

sr = CrispEsrganSr("esrgan-x4.gguf")
print(sr.scale)  # 4
out, ow, oh = sr.process(pixels, width, height)
```

## Rust

```rust
use crispembed::CrispEsrganSr;

let mut sr = CrispEsrganSr::new("esrgan-x4.gguf", 4).unwrap();
let (pixels, w, h) = sr.process(&rgb, width, height).unwrap();
```

## Dart / Flutter

```dart
final sr = CrispEsrganSr('esrgan-x4.gguf');
final result = sr.process(pixels, width, height);
print('${result.width}x${result.height}');
sr.dispose();
```

## Server

```bash
crispembed-server --esrgan-model esrgan-x4.gguf
# POST /esrgan/sr  {"image": "/path/to/image.png"}
```
