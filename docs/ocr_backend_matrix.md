# OCR backend and graph matrix

This is an implementation claim matrix, not a claim that every model artifact
is currently cached. “Full graph” means the inference-critical forward is
constructed as ggml graphs and dispatched through the selected backend. A GPU
build alone does not promote a CPU implementation.

| Engine family | Current path | Metal/CUDA today | Next action |
|---|---|---:|---|
| PP-OCRv6 detector/recognizer | detector opt-in full graph; tiny recognizer logits graph; small/medium recognizers CPU SVTR path; CPU accept-gate reference | Partial | O11.1/O11.6 parity and tier completion |
| PP-LCNet orientation | CPU reference plus opt-in backend graph with CPU fallback; Metal graph passes repeated execution and full pipeline smoke, but one numerical outlier remains and graph is slower than CPU; per-crop scheduler reallocation is retained because mixed CPU/Metal depthwise reuse is unreliable on pre-tensor Apple GPUs | Partial | O11.2 parity and reuse/perf gate |
| DBNet + TrOCR | DBNet CPU preprocessing plus graph-backed recognition | Partial | O11.3 detector/warp audit |
| Tesseract-LSTM | DBNet/crop pipeline plus recognizer path | Partial | measure crop batching |
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
