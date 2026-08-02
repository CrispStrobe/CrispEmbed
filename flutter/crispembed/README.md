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

## Intended purpose & acceptable use

CrispEmbed is a **software component**, not a finished AI system: it returns
vectors and text, and holds no identity, database, or decision threshold. Every
consequential decision happens in your app, and the responsibility for it is
yours. Full terms:
**[POLICY.md](https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md)**.

**Do not use this package** for untargeted scraping of facial images to build a
face database (EU AI Act Art. 5(1)(e)), emotion inference in the workplace or
education (Art. 5(1)(f)), biometric categorisation to deduce race, political
opinion, religion, sex life or sexual orientation (Art. 5(1)(g)), real-time
remote biometric identification in public spaces for law enforcement
(Art. 5(1)(h)), social scoring or predictive policing (Art. 5(1)(c),(d)), or
generation of non-consensual intimate imagery or CSAM. These prohibitions have
applied since 2 February 2025 and are not waived by the Art. 2(12) open-source
exemption.

**Face recognition.** `acceptBiometricUse()` exists because a face template is
special-category personal data under GDPR Art. 9 — you generally need an
Art. 9(2) condition (usually explicit consent) plus an Art. 6 lawful basis, and
a DPIA is likely mandatory. Loading a recognition model fails without that
acknowledgement. Detection alone is not gated: a bounding box is not a
template. There is deliberately no gallery, enrolment, or 1:N search primitive
— building one makes an Annex III high-risk system with its own obligations.

**OCR output is a probabilistic reconstruction, not a copy**, and VLM engines
can hallucinate plausible text. Do not make it the sole basis for a decision
about a person without human review of the source document.

## License

MIT. See `LICENSE`. This covers the code only — auto-downloaded models keep
their upstream licences, some non-commercial.
