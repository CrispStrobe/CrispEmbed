# CrispEmbed OCR — WASM browser demo

Fully client-side OCR in the browser. No server, no upload — the GGUF model
is fetched once (from Hugging Face or any CORS-enabled host) and inference
runs in WebAssembly.

**Live demo:** https://crispstrobe.github.io/CrispEmbed/
(deployed by `.github/workflows/deploy-pages.yml` on every push to `main`
that touches the WASM build)

## Modes

| Mode | Models | Default URLs |
|------|--------|--------------|
| Single model | pix2tex (math formula → LaTeX), TrOCR, PARSeq, HMER, BTTR | [`cstr/pix2tex-mfr-gguf`](https://huggingface.co/cstr/pix2tex-mfr-gguf) `pix2tex-mfr-q4_k.gguf` (17 MB) |
| Pipeline (Det+Rec) | DBNet detection + TrOCR recognition | [`cstr/dbnet-ic15-GGUF`](https://huggingface.co/cstr/dbnet-ic15-GGUF) `dbnet-ic15-q4_k.gguf` (7 MB) + [`cstr/trocr-small-printed-GGUF`](https://huggingface.co/cstr/trocr-small-printed-GGUF) `trocr-small-printed-q4_k.gguf` (45 MB) |
| Scan cleanup | none (classical deskew/binarize/denoise) | — |

Note: the Det+Rec pipeline is functional but slow in the single-threaded
WASM build (recognition is a ViT decode per detected region — expect minutes
on a full page). The single-model and cleanup modes run in seconds.

## Run locally

```bash
# 1. Build the WASM module (needs Emscripten; `brew install emscripten` or emsdk)
./build-wasm.sh

# 2. Copy the artifacts next to index.html
cd examples/wasm-ocr
cp ../../build-wasm/crispembed_ocr.{js,wasm} .
cp ../../wasm/crispembed-ocr.js .

# 3. Serve (plain static server; --coi only needed for --threads builds)
python serve.py
# open http://localhost:8080
```

## JS API (one-shot, memory-safe)

```js
const ocr = await CrispEmbedOCRWrapper.create({
  modelUrl: 'https://huggingface.co/cstr/pix2tex-mfr-gguf/resolve/main/pix2tex-mfr-q4_k.gguf',
  onProgress: (p) => console.log(`${(p * 100) | 0}%`),
});
const { text, confidence } = await ocr.recognize(imageOrCanvasOrBlobOrUrl);
ocr.dispose();
```

See `wasm/crispembed-ocr.js` for the pipeline (`CrispEmbedOCRPipeline`),
scan cleanup (`CrispEmbedScanCleanup`), text detection, and layout detection
wrappers.

## Tests

- `tests/test_wasm_ocr_live.js`, `tests/test_wasm_ocr_wrapper.js` — node
  unit/smoke tests against the real compiled module (run in CI).
- `tests/wasm-browser/e2e.test.js` — headless-Chromium end-to-end test of
  THIS page: loads a real model through the UI, OCRs a fixture image, and
  asserts the output equals the native CLI's ground truth (run in CI).
  Set `WASM_E2E_PIPELINE=1` to also exercise the Det+Rec pipeline.
