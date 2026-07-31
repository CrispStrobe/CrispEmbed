# OCR backend and graph matrix

This is an implementation claim matrix, not a claim that every model artifact
is currently cached. “Full graph” means the inference-critical forward is
constructed as ggml graphs and dispatched through the selected backend. A GPU
build alone does not promote a CPU implementation.

| Engine family | Current path | Metal/CUDA today | Next action |
|---|---|---:|---|
| PP-OCRv6 detector/recognizer | custom CPU conv/linear | No | O11.1 graph port |
| PP-LCNet orientation | custom CPU conv/linear | No | O11.2 graph port |
| DBNet + TrOCR | DBNet CPU preprocessing plus graph-backed recognition | Partial | O11.3 detector/warp audit |
| Tesseract-LSTM | DBNet/crop pipeline plus recognizer path | Partial | measure crop batching |
| PARSeq | ggml encoder/decoder graph | Yes, when backend enabled | residency/perf gate |
| Surya | CPU detector layers plus ggml graph stages | Partial | detector graph audit |
| GOT/GLM/Qwen/InternVL/DeepSeek VLMs | graph decoder; some vision neck/preprocess CPU | Partial | O11.4 vision-neck audit |
| Unlimited-OCR | mixed CPU neck and ggml graphs | Partial | O11.4 split residency gate |
| SmolDocling | CPU backend graph path | No | evaluate backend port |
| PP-FormulaNet / MixTeX | CPU CNN/encoder sections plus graph sections | Partial | O11.4 graph audit |
| HMER / BTTR / PosFormer | CPU CNN sections plus graph decoder | Partial | O11.4 graph audit |
| SMT / SMT++ / Polyphonic-TrOMR / Transcoda | ggml graph inference with backend selection | Yes, when backend enabled | O11.6 perf gate |
| Classical cleanup, SR, dewarp | varies; several dedicated CPU paths | Partial | O11.3 benchmark before porting |

The local diagnostic build used for CPU parity currently has `GGML_METAL=OFF`
and `GGML_CUDA=OFF`. CI’s macOS release configuration enables Metal, but each
row above still requires an engine-specific residency test before it may be
described as GPU-accelerated.
