# crispembed

Flutter/Dart FFI bindings for
[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed), an on-device
ggml-based inference library for embeddings, OCR, layout analysis, scan
cleanup and related document-processing tasks.

The package exposes the Dart API. It expects the native `crispembed`
library to be supplied by your application or build pipeline:

- Linux and Android: `libcrispembed.so`
- macOS: `libcrispembed.dylib`
- Windows: `crispembed.dll`
- iOS: `Libs/libcrispembed-static.a` linked into the app

The platform plugin files contain the expected locations for prebuilt
libraries. Model GGUF files are loaded at runtime by the native library.

## Usage

```dart
import 'package:crispembed/crispembed.dart';

final model = CrispEmbed('/path/to/model.gguf');
final embedding = model.encode('A short document');
model.dispose();
```

The public API also includes wrappers for sparse and ColBERT embeddings,
reranking, face pipelines, OCR engines, OMR, layout detection, scan cleanup,
PDF DPI analysis and image restoration models, depending on which symbols are
available in the native library you bundle.

## License

MIT. See `LICENSE`.
