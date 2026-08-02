# CrispEmbed — Python wheel

Lightweight ggml-based text embedding inference. **23+ verified models**,
~9× faster than FastEmbed (ONNX) on the standard MiniLM-L6 benchmark.

Supports BERT, XLM-R, MPNet, NomicBERT, ModernBERT, DeBERTa-v2, Qwen3,
Gemma3, and BidirLM-Omni — including SPLADE sparse, ColBERT multi-vector,
and cross-encoder reranking through the same shared library.

This package bundles the native `libcrispembed.{so,dylib,dll}` plus its
ggml backend siblings, so a plain `pip install crispembed` works without
needing CMake on the user's machine.

## Quick start

```python
from crispembed import CrispEmbed

# Auto-downloads the GGUF from huggingface.co/cstr/<model>-GGUF on first use.
ce = CrispEmbed("all-MiniLM-L6-v2")
vec = ce.encode("Hello world")          # numpy float32 [384]
batch = ce.encode(["hello", "world"])   # numpy float32 [2, 384]
```

## Multimodal (BidirLM-Omni)

```python
ce = CrispEmbed("bidirlm-omni-2.5b")
text_vec  = ce.encode("a cat on a mat")
image_vec = ce.encode_image("cat.jpg")     # needs `pip install crispembed[image]`
# text_vec @ image_vec.T is a meaningful cross-modal similarity score.
```

## Building from source

Wheels are built by CI from the parent `CrispEmbed` repo via CMake. To
build locally, see the parent project's `README.md` and `build-macos.sh`
/ release.yml. The Python package's `pyproject.toml` here is just a thin
packaging shim around the prebuilt native libraries.

## Intended purpose & acceptable use

CrispEmbed is a **software component**, not a finished AI system: it returns
vectors and text, holds no identity, no database, and no decision threshold.
Every consequential decision happens in your code, and the responsibility for
it is yours.

Full terms are in **[POLICY.md](https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md)**,
which also ships in this package as `crispembed/POLICY.md`. The short version:

**Do not use this package to build or operate** — untargeted scraping of facial
images to build a face database (EU AI Act Art. 5(1)(e)); emotion inference in
the workplace or education (Art. 5(1)(f)); biometric categorisation to deduce
race, political opinion, religion, sex life or sexual orientation
(Art. 5(1)(g)); real-time remote biometric identification in public spaces for
law enforcement (Art. 5(1)(h)); social scoring or predictive policing
(Art. 5(1)(c),(d)); or generation of non-consensual intimate imagery or CSAM.
These prohibitions have applied since 2 February 2025 and are **not** waived by
the Art. 2(12) open-source exemption.

**Face recognition.** `crispembed.accept_biometric_use()` exists because a face
template is special-category personal data under GDPR Art. 9 — you generally
need an Art. 9(2) condition (usually explicit consent) plus an Art. 6 lawful
basis, and a DPIA is likely mandatory. Loading a recognition model fails
without that acknowledgement (or `CRISPEMBED_ACCEPT_BIOMETRIC=1`). Detection
alone is not gated: a bounding box is not a template. Building a gallery and
searching it (1:N identification) is an Annex III high-risk system with its own
provider and deployer obligations — this package deliberately ships no gallery,
enrolment, or 1:N search primitive.

**OCR output is a probabilistic reconstruction, not a copy.** VLM-based engines
can hallucinate plausible text. Do not use it as the sole basis for a decision
about a person without human review of the source document.

**Model licences are not the package licence.** Auto-downloaded GGUFs keep their
upstream licence, some non-commercial. Run `crispembed --list-models` for the
per-model tags.

## License

MIT. See `LICENSE`. This covers the code only — see the model licence note above.
