# Pix2Struct -- Document Understanding (Image-to-Text)

Pix2Struct is a pretrained image-to-text model from Google for visually-situated
language understanding. It renders input images as variable-resolution sequences
of patches, eliminating the need for fixed-resolution preprocessing, and decodes
with a standard T5 text decoder.

## Architecture

- **Encoder**: Variable-resolution Vision Transformer (ViT). Instead of
  resizing all images to a fixed resolution, Pix2Struct scales images to
  extract a variable number of 16x16 patches (up to a maximum sequence
  length of 2048 patches). This preserves aspect ratios and fine details
  critical for document/chart understanding.
- **Decoder**: Standard T5 text decoder with causal attention.
- **Parameters**: 282M (base), 1.3B (large)
- **Pretraining**: Screenshot Parsing -- predict simplified HTML from
  masked webpage screenshots. This single self-supervised objective teaches
  layout understanding, text recognition, and structure extraction.
- **License**: Apache-2.0
- **Source**: [google/pix2struct-base](https://huggingface.co/google/pix2struct-base)

## Parity

Cosine similarity between reference PyTorch output and CrispEmbed GGUF
engine output for encoder and decoder hidden states: **cos = 1.000000**.

## Fine-Tuned Variants

Pix2Struct has 17 fine-tuned variants covering diverse visual language tasks:

| Variant | Task | Dataset |
|---------|------|---------|
| pix2struct-textcaps | Image captioning with OCR | TextCaps |
| pix2struct-screen2words | Screen summarization | Screen2Words |
| pix2struct-widget-captioning | UI widget captioning | Widget Captioning |
| pix2struct-docvqa | Document visual QA | DocVQA |
| pix2struct-infographicvqa | Infographic visual QA | InfographicVQA |
| pix2struct-chartqa | Chart visual QA | ChartQA |
| pix2struct-ocrvqa | OCR visual QA | OCR-VQA |
| pix2struct-ai2d | Science diagram QA | AI2D |
| pix2struct-refexp | Referring expression | RefExp |
| pix2struct-tab-finetuned | Table understanding | Various |

All variants share the same base architecture; only decoder weights differ.

## Usage

### CLI

```bash
# Explicit model path
crispembed -m pix2struct-base-q8_0.gguf --pix2struct document.png

# Auto-download from registry
crispembed -m pix2struct-base --pix2struct chart.png

# With max token limit
crispembed -m pix2struct-base --pix2struct document.png --pix2struct-max-tokens 512
```

### HTTP Server

```bash
# Start server
crispembed-server --pix2struct pix2struct-base-q8_0.gguf

# Generate text
curl -X POST http://localhost:8080/pix2struct/generate \
  -H "Content-Type: application/json" \
  -d '{"image": "/path/to/document.png", "max_tokens": 256}'
# → {"text": "...", "ms": 1234.5}
```

### Python

```python
from crispembed import CrispPix2Struct

p2s = CrispPix2Struct("pix2struct-base-q8_0.gguf")
text = p2s.generate("document.png")
print(text)
```

### Rust

```rust
use crispembed::CrispPix2Struct;

let mut p2s = CrispPix2Struct::new("pix2struct-base-q8_0.gguf", 0).unwrap();
let text = p2s.generate(&image_bytes, width, height, 256);
println!("{:?}", text);
```

### Dart / Flutter

```dart
final p2s = CrispPix2Struct('pix2struct-base-q8_0.gguf');
final text = p2s.generate(imageBytes, 800, 600, maxTokens: 256);
print(text);
p2s.dispose();
```

## GGUF Quantization

| Format | Parity (cos) | Notes |
|--------|-------------|-------|
| F16    | 1.000000    | Full precision |
| Q8_0   | 1.000000    | Recommended |
| Q4_K   | 0.999+      | Smallest, minor quality loss |
