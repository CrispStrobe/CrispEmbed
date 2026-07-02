# crispembed-imatrix-quant (Kaggle)

Per-model importance-matrix (imatrix) quantization for CrispEmbed — the C1
rollout. Runs **one model per invocation** so you can loop:

```
run --model lfm2-embed   # calibrate → quant+imatrix → A/B → upload → rm
run --model bge-m3
run --model jina-v5-nano
...
```

Each run (see `imatrix_quant.py`):
1. builds `crispembed-cli` + `crispembed-quantize` from `origin/main` (C1 is merged),
2. downloads the HF source and converts it to an f16 GGUF,
3. calibration pass with `CRISPEMBED_IMATRIX_OUT` over `calib_corpus.txt`
   (falls back to a builtin corpus),
4. for each quant spec `q8_0` (no imatrix), `q4_k` (+imatrix), `iq4_xs` (+imatrix):
   quantize → **A/B cosine vs the f16 gold** on `eval_corpus.txt` → upload → `rm`,
5. uploads the f16 + `.imatrix`, then removes everything locally.

Local validation reference (jina-v5-nano, cos vs f16 gold):
`q4_k` 0.9455→0.9569 (+imatrix); `iq4_xs` 0.9584→0.9648 (+imatrix, 172.7 MB).

## Kaggle regime (same as crispembed-quant-upload)
Uses `kaggle_harness` (kh): `init_progress` (JSONL progress pushed to HF),
`resolve_hf_token` (env → Kaggle Secret → **dataset** `hf_token.txt`),
`install_build_toolchain` + ccache warmed from the `crispasr-ccache` dataset, and
`build_heartbeat` (30 s) around every long step so the kernel never idles out.

Runs under the **chr1str** account (CrispEmbed convention; it owns both datasets —
cross-account attach is blocked, see kaggle_usage.md #13). Attaches **both**
(in `kernel-metadata.json`): `chr1str/crispasr-hf-token` (token) and
`chr1str/crispasr-ccache` (ccache seed — warms the shared ggml-cuda build).
CPU build by default (`-DGGML_CUDA=OFF`) — a CUDA build compiles ggml-cuda's
~254 template TUs (~15 min) these small embedders never use. `enable_gpu:true`
stays ONLY because Kaggle CPU workers get no internet (usage #3); the GPU
provides internet, not the build. Set `CRISP_GPU=1` for large models (BidirLM). `kaggle_harness.py` is bundled (also cloned from CrispASR at runtime).

Push (chr1str is the active CLI account):
```
cd tools/kaggle/crispembed-imatrix-quant && python -m kaggle kernels push -p .
```
Pushing runs it immediately. Edit `MODEL` before each push (one model per run).

## Setup
- Pick the model: edit the `MODEL` constant at the top of `imatrix_quant.py`
  (or set the `MODEL` env var), then `kaggle kernels push`. One model per run.
- HF token comes from the `crispasr-hf-token` dataset (no secret needed).
- Override corpora by dropping `calib_corpus.txt` / `eval_corpus.txt` next to the
  script. **Use text resembling the target embedding domain** — imatrix quality
  scales with calibration relevance.

## Per-model notes (verify before running)
- `lfm2-embed`, `bge-m3`, `e5-large`: converter args confirmed.
- `jina-v5-nano`: confirm the exact HF repo id (`hf_src`) and whether a LoRA
  adapter merge is needed (`convert-decoder-embed-to-gguf.py --lora-adapter …`).
- **BidirLM-Omni**: not yet wired — multimodal (vision+audio), no single-file
  converter in `models/`. Add its conversion path to `MODELS` before running.

The reusable local A/B harness is `tools/imatrix_ab.py` (same metric, runs
against a local f16 GGUF).
