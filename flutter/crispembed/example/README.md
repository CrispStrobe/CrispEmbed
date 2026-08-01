# crispembed examples

`crispembed` runs lightweight text-embedding and OCR/OMR inference on-device via
ggml. See `feature_parity.dart` in this directory for an end-to-end parity
harness; the essentials:

## Text embeddings

```dart
import 'package:crispembed/crispembed.dart';

void main() {
  // Needs the native libcrispembed at runtime (see the package README).
  final model = CrispEmbed('/path/to/model.gguf');
  try {
    final embedding = model.encode('A short document');
    print('dim = ${embedding.length}');
  } finally {
    model.dispose();
  }
}
```

## What else is here

`crispembed` also exposes sparse (BGE-M3 / SPLADE) and ColBERT multi-vector
embeddings, cross-encoder reranking, and the OCR/OMR engines
(`math_ocr`, `omr`, `hmer_ocr`, `bttr_ocr`, `text_detect`, `layout`,
`scan_cleanup`) — all imported from `package:crispembed/crispembed.dart`.
