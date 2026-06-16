# SCUNet Image Denoising

## Architecture

SCUNet (Swin-Conv-UNet) is a hybrid image denoising model combining Swin Transformer blocks (shifted window multi-head self-attention) with residual convolutional blocks in a U-Net encoder-decoder structure.

- **Backbone**: U-Net with 4 encoder + 4 decoder stages
- **Blocks**: Each stage alternates Swin Transformer blocks (window attention + shifted window attention) with residual Conv blocks (Conv-LeakyReLU-Conv)
- **Parameters**: ~18M
- **GGUF size**: 69 MB (F32)
- **Input/Output**: Same resolution (denoising, not super-resolution)
- **Training data**: SIDD (Smartphone Image Denoising Dataset) for real-world noise
- **Source**: [cszn/SCUNet](https://github.com/cszn/SCUNet) (CVPR 2022)
- **License**: Apache-2.0

## Parity

CrispEmbed C++ engine vs PyTorch reference:

```
cos = 1.000000
```

## API

### C

```c
#include "crispembed.h"

void * ctx = crispembed_scunet_init("scunet-color-f32.gguf", 4);

// Denoise RGB image (output same size as input)
uint8_t * out = NULL;
int rc = crispembed_scunet_process(ctx, rgb_pixels, width, height, &out);
// out is [height * width * 3] uint8 RGB
crispembed_scunet_free_image(out);
crispembed_scunet_free(ctx);
```

### CLI

```bash
crispembed --scunet-model scunet-color-f32.gguf --scunet-denoise noisy.png > clean.ppm
```

### Server

```bash
crispembed-server --scunet-model scunet-color-f32.gguf

curl -X POST http://localhost:8080/scunet/denoise \
  -H 'Content-Type: application/json' \
  -d '{"image": "noisy.png"}'
# Returns: {"image": "<base64 RGB>", "width": W, "height": H, "ms": ...}
```

### Python

```python
from crispembed import CrispScunet

dn = CrispScunet("scunet-color-f32.gguf")
result = dn.process(pixels, width, height)  # numpy (H, W, 3) uint8
```

### Rust

```rust
use crispembed::CrispScunet;

let mut dn = CrispScunet::new("scunet-color-f32.gguf", 4).unwrap();
let (pixels, w, h) = dn.process(&rgb_data, width, height).unwrap();
```

### Flutter / Dart

```dart
final dn = CrispScunet('scunet-color-f32.gguf');
final result = dn.process(pixels, width, height);  // Uint8List
dn.dispose();
```

## Model download

```bash
crispembed --list-models | grep scunet
crispembed --download scunet-color
```

Or manually:
```
https://huggingface.co/cstr/scunet-GGUF/resolve/main/scunet-color-f32.gguf
```
