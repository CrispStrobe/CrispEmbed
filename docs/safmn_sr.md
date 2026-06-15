# SAFMN Super-Resolution

Lightweight whole-image super-resolution based on Spatially-Adaptive Feature Modulation. Upscales images 2x or 4x with only ~228K parameters.

## Architecture

Single-path design: Conv3x3 -> 8 AttBlocks -> Conv3x3 + PixelShuffle.

Each AttBlock contains:
1. **SAFM** (Spatially-Adaptive Feature Modulation) -- multi-scale spatial attention via progressive downsampling + channel mixing
2. **CCM** (Convolutional Channel Mixer) -- Conv3x3 + GELU + Conv3x3 feed-forward

Final PixelShuffle x4 (or x2) sub-pixel upsampling produces the high-resolution output.

## Model

- **Source**: [sunny2109/SAFMN](https://github.com/sunny2109/SAFMN) (ICCV 2023)
- **License**: Apache-2.0
- **Parameters**: ~228K (extremely lightweight)
- **GGUF**: `/mnt/storage/gguf-models/safmn-sr-x4-f16.gguf`
- **Converter**: `models/convert-safmn-to-gguf.py`

## Parity

Reference vs C++ engine: cosine similarity = 1.000000 (exact float parity on F16 GGUF).

## C API

```c
#include "crispembed.h"

void * ctx = crispembed_safmn_sr_init("safmn-sr-x4.gguf", 4);
int scale = crispembed_safmn_sr_scale(ctx);  // 4

uint8_t * out = NULL;
int ow, oh;
crispembed_safmn_sr_process(ctx, rgb_pixels, w, h, 0, 0, &out, &ow, &oh);
// out is w*scale x h*scale x 3 RGB
crispembed_safmn_sr_free_image(out);
crispembed_safmn_sr_free(ctx);
```

## CLI

```bash
crispembed --safmn-model safmn-sr-x4.gguf --safmn-sr input.png > upscaled.ppm
```

## Python

```python
from crispembed import CrispSafmnSr

sr = CrispSafmnSr("safmn-sr-x4.gguf")
print(sr.scale)  # 4
out, ow, oh = sr.process(pixels, width, height)
```

## Rust

```rust
use crispembed::CrispSafmnSr;

let mut sr = CrispSafmnSr::new("safmn-sr-x4.gguf", 4).unwrap();
let (pixels, w, h) = sr.process(&rgb, width, height).unwrap();
```

## Dart / Flutter

```dart
final sr = CrispSafmnSr('safmn-sr-x4.gguf');
final result = sr.process(pixels, width, height);
print('${result.width}x${result.height}');
sr.dispose();
```

## Server

```bash
crispembed-server --safmn-model safmn-sr-x4.gguf
# POST /safmn/sr  {"image": "/path/to/image.png"}
```
