# OCR backend and graph matrix

This is an implementation claim matrix, not a claim that every model artifact
is currently cached. “Full graph” means the inference-critical forward is
constructed as ggml graphs and dispatched through the selected backend. A GPU
build alone does not promote a CPU implementation.

| Engine family | Current path | Metal/CUDA today | Next action |
|---|---|---:|---|
| PP-OCRv6 detector/recognizer | detector opt-in full graph with CPU geometry accept-gate; tiny/small/medium recognizers have gated full stem/backbone/SVTR graphs with CPU fallback and multi-fixture gold references. **Recognizer correctness settled 2026-08-02** against the upstream PaddleOCR source (stem ReLU, neck local residual, skip after the blocks + norm, aspect-preserving width, space class): CER 0.0031 on the 20-fixture ground-truth corpus, the best arm measured. Reachable as `--ocr-engine ppocrv6` (C-ABI id 16) | Partial | shape-keyed recognizer graph (the fixed 48x320 graph is refused for wider crops and falls back to CPU), detector geometry parity, quiet-box latency |
| PP-LCNet orientation | CPU reference plus opt-in backend graph with automatic CPU fallback; Metal graph passes repeated execution and full pipeline smoke, but one numerical outlier remains and graph is slower than CPU; per-crop scheduler reallocation is retained because mixed CPU/Metal depthwise reuse is unreliable on pre-tensor Apple GPUs | Partial | Metal SE/depthwise numerical parity and reuse/perf optimization |
| DBNet + TrOCR | DBNet CPU preprocessing plus graph-backed recognition. The short-side resize no longer enlarges a page that already resolves its text (`max_upscale`/`upscale_floor`), which was 4.7x of the detector cost and also hurt accuracy | Partial | O11.3 detector/warp audit |
| Tesseract-LSTM | DBNet/crop pipeline plus recognizer path; CER 0.0290 vs system Tesseract 0.0256 on the ground-truth corpus | Partial | measure crop batching; WER gap is spacing/grouping, not recognition |
| EasyOCR | DBNet detection plus EasyOCR CRNN recognition via `easyocr_pipeline`; reachable as `--ocr-engine easyocr` (C-ABI id 17); CER 0.0808 vs Python EasyOCR 0.0769 | Partial | CRAFT detector parity; WER/spacing |
| PARSeq | ggml encoder/decoder graph | Yes, when backend enabled | residency/perf gate |
| Surya | CPU detector layers plus ggml graph stages | Partial | detector graph audit |
| GOT/GLM/Qwen/InternVL/DeepSeek VLMs | GPU-scheduled vision/decoder graphs with engine-specific CPU boundaries: GOT/DeepSeek retain CPU window partition; Qwen retains CPU spatial merge/position work; GLM retains optional scalar merger; InternVL has host-side pixel-unshuffle/merge; all schedulers keep a CPU fallback | Partial | O11.4 vision-neck audit and per-engine residency/perf gates |
| Unlimited-OCR | GPU-scheduled SAM/LLM graphs, with CPU spatial merge and opt-in CPU neck/MoE fallbacks (`UOCR_SAM_CONV_CPU`, `UOCR_MOE_CPU`); scheduler retains CPU fallback | Partial | O11.4 split residency gate and end-to-end timing |
| SmolDocling | CPU scalar preprocessing plus CPU ggml graphs for SigLIP/LLM | No | evaluate backend port |
| PP-FormulaNet / MixTeX | CPU CNN/transformer sections; FormulaNet encoder graph and MixTeX batched linear graph are CPU-scheduled | No | O11.4 graph audit |
| HMER / BTTR / PosFormer | CPU CNN sections plus CPU-scheduled ggml encoder graphs and decoder | No | evaluate backend residency port |
| SMT / SMT++ / Polyphonic-TrOMR / Transcoda | ggml graph inference with backend selection | Yes, when backend enabled | O11.6 perf gate |
| Classical cleanup, SR, dewarp | varies; several dedicated CPU paths | Partial | O11.3 benchmark before porting |

The default local diagnostic build used for CPU parity has `GGML_METAL=OFF`
and `GGML_CUDA=OFF`. A separate macOS build with `GGML_METAL=ON` now passes
the backend graph smoke on Apple M1, both in auto mode (`MTL0`) and with an
explicit `test-backend-smoke metal` request. This validates backend/device
selection only; it does not promote any OCR row to “Full graph” without an
engine-specific residency and parity test. Explicit requests fail when the
requested GPU is unavailable instead of silently passing on CPU.
