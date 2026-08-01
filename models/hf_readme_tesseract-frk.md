---
license: apache-2.0
language:
- de
- frk
tags:
- ocr
- fraktur
- tesseract
- gguf
- crispembed
pipeline_tag: image-to-text
---

# Tesseract German Fraktur — GGUF

Native CrispEmbed GGUF port of the Apache-2.0 Tesseract `frk` LSTM language
model. It recognizes German Fraktur line images, including long-s (`ſ`) and
historical characters in the upstream output alphabet.

Source: [tesseract-ocr/tessdata `frk.traineddata`](https://github.com/tesseract-ocr/tessdata/blob/main/frk.traineddata)

- Source SHA-256: `7cd1b541e9d3884b9546a7d292c4f349cdb45ff646b4b43166dfc5099a8ad1a1`
- Source training flags: `65` (`int_mode=true`), preserved in both GGUF files
- VGSL: `[1,48,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx384O1c1]`
- Input height: 48 pixels
- Output alphabet: 100 Unicode tokens / 99 CTC output codes
- Parameters: 933,763
- Architecture: convolution + max-pool + four recurrent LSTM stages + CTC projection

## Files

| File | Precision | Size | Use |
|---|---:|---:|---|
| `tesseract-frk-f32.gguf` | F32 | 3.6 MB | Reference and maximum fidelity |
| `tesseract-frk-q8_0.gguf` | Mixed F32/Q8_0 | 1.1 MB | Recommended deployment variant |

The Q8 variant keeps `output.weight` and `output.bias` at F32 because they are
the sensitive CTC decision boundary. Recurrent matrices are Q8_0; the input
convolution and smaller matrices remain F32. This is the same precision-first
policy used for other CrispEmbed OCR GGUFs.

## Usage

Pass a single-line grayscale crop to the native Tesseract LSTM path:

```bash
crispembed -m tesseract-frk-q8_0.gguf --ocr line.png
```

For full pages, run a text detector/line segmenter first and recognize each
deskewed crop independently. This model is a line recognizer, not a page
layout or text-detection model.

## Validation

The native implementation was compared with a pure-Python reference generated
from the same `frk.traineddata` and the same 200×80 grayscale crop:

| Variant | Worst stage cosine | Logits cosine | Result |
|---|---:|---:|---|
| F32 | 1.000000 | 1.000000 | Pass |
| Q8_0 mixed precision | 0.999873 | 0.999945 | Pass |

The reference activation dump is retained outside Git at
`$CRISPEMBED_GGUF_DIR/tesseract-frk-ref-line.gguf`.

## License

Apache-2.0, following the upstream Tesseract language data. Preserve the
upstream source URL, checksum, and license when redistributing this GGUF.
