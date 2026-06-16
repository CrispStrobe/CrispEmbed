# InstructIR — All-in-One Image Restoration

**InstructIR** is a text-guided all-in-one image restoration model from
[mv-lab/InstructIR](https://github.com/mv-lab/InstructIR) (ECCV 2024).

## Architecture

- **Backbone:** NAFNet U-Net with Instruction-Conditioned Blocks (ICB)
- **Parameters:** ~16M
- **License:** MIT
- **Input/Output:** RGB uint8 (same resolution)

Pre-computed prompt embeddings for each task are baked into the GGUF file,
so no text encoder is needed at runtime.

## Tasks

| ID | Task | Description |
|----|------|-------------|
| 0 | `denoise` | Gaussian noise removal |
| 1 | `deblur` | Motion / defocus deblurring |
| 2 | `dehaze` | Haze and fog removal |
| 3 | `derain` | Rain streak removal |
| 4 | `super_resolution` | Single-image super-resolution |
| 5 | `low_light` | Low-light enhancement |
| 6 | `enhance` | General image enhancement |

## Usage

### CLI

```bash
crispembed --instructir-model instructir-f16.gguf \
           --instructir noisy.png \
           --instructir-task 0 > denoised.ppm
```

### Server

```bash
crispembed-server --instructir-model instructir-f16.gguf

curl -X POST http://localhost:8080/instructir/restore \
  -d '{"image": "noisy.png", "task": 0}'
```

### Python

```python
from crispembed import CrispInstructIR

ir = CrispInstructIR("instructir-f16.gguf")
out = ir.process(pixels, width, height, task=0)
```

### Rust

```rust
let mut ir = CrispInstructIR::new("instructir-f16.gguf", 4).unwrap();
let (pixels, w, h) = ir.process(&input, width, height, 0).unwrap();
```

### Flutter / Dart

```dart
final ir = CrispInstructIR('instructir-f16.gguf');
final out = ir.process(pixels, width, height, task: 0);
ir.dispose();
```

## C API

```c
void * ctx = crispembed_instructir_init("instructir-f16.gguf", 4);
int n = crispembed_instructir_n_tasks(ctx);  // 7

uint8_t * out = NULL;
int rc = crispembed_instructir_process(ctx, 0, input, w, h, &out);
// out is [h * w * 3] RGB uint8
crispembed_instructir_free_image(out);
crispembed_instructir_free(ctx);
```

## Model download

```bash
crispembed --list-models | grep instructir
crispembed --download instructir
```

Or manually from HuggingFace:
<https://huggingface.co/cstr/InstructIR-GGUF>

## Reference

Marcos V. Conde, Gregor Geigle, Radu Timofte.
*High-Quality Image Restoration Following Human Instructions.*
ECCV 2024. [arXiv:2401.16468](https://arxiv.org/abs/2401.16468)
