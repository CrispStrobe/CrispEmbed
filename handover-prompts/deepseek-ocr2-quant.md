# Handover: DeepSeek-OCR-2 GGUF Reconversion + Quantization

## Current State

The F16 GGUF on HuggingFace (`cstr/deepseek-ocr2-crispembed-GGUF`) has a
**broken tokenizer** — `tokenizer.ggml.merges` is stored as a nested array
(array-of-arrays, etype=9) instead of array-of-strings. This causes
`crispembed-quantize` to fail with:
```
gguf_init_from_file_ptr: key 'tokenizer.ggml.merges' has invalid GGUF type 9
```

The converter code (`models/convert-deepseek-ocr2-to-gguf.py`) has been
fixed to flatten merge pairs to strings, but the GGUF was never reconverted.

## What needs to happen

1. Download `deepseek-ai/DeepSeek-OCR-2` safetensors from HuggingFace
2. Reconvert to F16 GGUF with the fixed converter
3. Quantize F16 → Q8_0 + Q4_K
4. Upload all three (F16, Q8_0, Q4_K) to `cstr/deepseek-ocr2-crispembed-GGUF`

## Why Kaggle is needed

The model is 3.4B params (~6.4 GB F16). The 8GB VPS cannot load it for
quantization. Kaggle P100 has 30GB RAM.

## Kaggle kernel

Location: `tools/kaggle/crispembed-quant-upload/`

**v6 is currently running** at:
https://www.kaggle.com/code/chr1str/crispembed-deepseek-quant

### CRITICAL: Kaggle harness rules (DO NOT DEVIATE)

1. **ALWAYS read a working kernel template first** — use
   `tools/kaggle/qwen2vl-convert/qwen2vl_convert.py` as the gold standard.

2. **Mandatory boilerplate** (copy verbatim from template):
   ```python
   CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
   _CRISPASR_DIR = WORK / "CrispASR"
   if not _CRISPASR_DIR.exists():
       try:
           subprocess.check_call(["git", "clone", "--depth", "1",
               CRISPASR_URL, str(_CRISPASR_DIR)])
           sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
       except Exception:
           pass
   if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
       sys.path.insert(0, str(Path(__file__).resolve().parent))
   import kaggle_harness as kh
   kh.init_progress()
   hf_token = kh.resolve_hf_token()
   ```

3. **ALWAYS bundle `kaggle_harness.py`** in the push directory (copy from
   `/mnt/volume1/CrispASR/tools/kaggle/kaggle_harness.py`).

4. **NEVER use `huggingface-cli`** — it's deprecated on Kaggle. Use
   `hf_hub_download()` or `snapshot_download()` from Python.

5. **Use `chr1str` account** with `chr1str/crispasr-hf-token` dataset.

6. **Slug must match title** in `kernel-metadata.json`.

7. **Build inside the repo** — `BUILD = REPO / "build"`, not `WORK / "build"`.

8. **Use harness build helpers**: `kh.install_build_toolchain()`,
   `kh.cache_and_link_flags()`, `kh.safe_build_jobs()`.

9. **Wrap ALL long ops** in `kh.build_heartbeat("label")`.

10. **Check converter CLI args** by running `--help` before writing the call.

## Converter CLI

```bash
python convert-deepseek-ocr2-to-gguf.py \
    --model <path/to/model-00001-of-000001.safetensors> \
    --config <path/to/config.json> \
    --tokenizer <path/to/tokenizer.json> \
    --output deepseek-ocr2-f16.gguf \
    --fp16
```

Note: `--fp16` flag (not `--dtype f16`). The `--model` arg takes the
safetensors file path, not the directory.

## Failures so far (avoid repeating)

| Version | Error | Root cause |
|---------|-------|------------|
| v1 | `kh.resolve_tokens()` AttributeError | Wrong function name (correct: `kh.resolve_hf_token()`) |
| v2 | `huggingface-cli` deprecated | Used CLI instead of Python API |
| v3 | `../tests/test_adair_diff.cpp` not found | Build dir outside repo (fixed in CMakeLists.txt) |
| v4 | `deepseek-ai/DeepSeek-OCR2` 404 | Wrong repo ID (correct: `DeepSeek-OCR-2` with hyphen) |
| v5 | `--config --tokenizer required` | Passed `--dtype f16` instead of `--fp16`, missing `--config`/`--tokenizer` |

## If v6 fails

Check the converter's `--help` output and the exact file layout in the
snapshot directory. The model may have multiple safetensors shards or
a different config structure. Always `ls` the downloaded directory first.

## Files

- `tools/kaggle/crispembed-quant-upload/quant_upload.py` — the kernel script
- `tools/kaggle/crispembed-quant-upload/kaggle_harness.py` — bundled fallback
- `tools/kaggle/crispembed-quant-upload/kernel-metadata.json` — Kaggle config
- `models/convert-deepseek-ocr2-to-gguf.py` — the converter (has the merge fix)
