# crispembed-imatrix-rerank-f7 (Kaggle, chr1s4)

**F7b for the rerankers.** F7 (`68033e8d` on main) named the runtime's
pre-merged BERT QKV weight and taught the quantizer the alias, so every imatrix
artifact published before it collected q/k/v statistics under ggml's auto
`leaf_N` and quantized attention with **no importance**. The embedding roster
was re-done by `crispembed-imatrix-t19` (F7b); this kernel does the seven
published cross-encoders.

Thin driver — all the logic is `../crispembed-imatrix-quant/imatrix_quant.py`,
executed FROM THE CLONE (a Kaggle *script* kernel ships only its `code_file`,
so the calibration corpora must travel with the git clone; `kaggle_usage.md`
#26/#19).

## Measured starting state (published `.imatrix` files, read with `GGUFReader`)

| model | keys | `leaf_N` | verdict |
|---|--:|--:|---|
| ms-marco-MiniLM-L-6-v2 | 24 | 6 | defect (q/k/v uncovered) |
| ms-marco-MiniLM-L-12-v2 | 48 | 12 | defect |
| bge-reranker-v2-m3 | 96 | 24 | defect |
| jina-reranker-v2-base-multilingual | 48 | 12 | defect |
| mxbai-rerank-base-v1 | 72 | 12 | defect (only `attn_v` missing; DeBERTa also projects q/k over the relative-position embeddings, which named those two) |
| mxbai-rerank-xsmall-v1 | 72 | 12 | defect (same shape) |
| bge-reranker-base | 72 | 0 | **already clean — no-change CONTROL** |

`leaf_N` equals `n_layer` in every defective file. `bge-reranker-base` escaped
because its GGUF ships **F16** q/k/v and `src/crispembed.cpp` skips the
pre-merge for non-F32 weights, so each projection kept its own name.

## ms-marco: the base file is pinned explicitly

Both ms-marco repos carry the superseded `<name>.gguf` (F32, 1-layer head — the
G7c defect) *and* the corrected `<name>-g7c.gguf`. `pick_base_gguf` prefers the
exact `<name>.gguf`, so the driver pins `base_file=<name>-g7c.gguf`. The g7c
artifacts are **ollama-mode-named** (`blk.N.attn_q.weight`); an imatrix
collected on a `--crisp` conversion renames the encoder and matches 0 tensors.
The proof either way is the quantizer's own stdout — the pipeline now parses
`N quantized, M kept, K with imatrix` and **raises** when an `-imatrix` arm
reads `K == 0`, instead of uploading a mislabeled file.

## Rerank calibration needs no surgery

The shared pipeline already has first-class cross-encoder support:
`MODE[<model>] == "rerank"` routes calibration through **14 `(query, [docs])`
pairs** on the `--rerank` path (the collector fires on that graph exactly like
the embed path), and the A/B metric is mean **Kendall-tau over 30 eval pairs ×
6 docs** vs the full-precision gold, with mean `|dscore|` as tiebreaker. Text
corpora (`calib_corpus.jsonl`) are used only by the embed modes.

Added for this run (in `imatrix_quant.py`, so t19 gets them too):
`base_file`/`prefix` overrides, the quantizer-coverage check above, an imatrix
`leaf_N` digest appended to every A/B summary, and a raw-logit block per arm
(a reranker's absolute scale is what tau cannot see — G7c shipped a broken head
that still ranked but scored ±0.2 instead of ±11).

## Naming (G3 / F7b precedent — new names only)

Published SHAs stay valid; `examples/cli/model_hashes.h` pins several of these
files. Nothing is overwritten and the registry is untouched.

```
<prefix>-f7.imatrix                <prefix>-f7-imatrix-ab.txt
<prefix>-q4_k-imatrix-f7.gguf      <prefix>-iq4_xs-f7.gguf
# ms-marco composes with the G7c correction suffix:
<prefix>-g7c-f7.imatrix            <prefix>-g7c-f7-imatrix-ab.txt
<prefix>-q4_k-imatrix-g7c-f7.gguf  <prefix>-iq4_xs-g7c-f7.gguf
```

## Spot-check

After the batch, the driver re-downloads the **just-uploaded**
`ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf` and scores a 6-doc Berlin
probe against **ONNX Runtime** on `Xenova/ms-marco-MiniLM-L-6-v2` (the local
miniconda torch mis-executes BERT-class forwards — never use it as reference).
Result is printed and uploaded as `…-g7c-f7-spotcheck.txt`. Non-fatal by
design: the artifacts are already published by then.

## Push

```bash
cd tools/kaggle/crispembed-imatrix-rerank-f7
KAGGLE_API_TOKEN=<chr1s4 token> python -c \
  "from kaggle import KaggleApi; a=KaggleApi(); a.authenticate(); print(a.kernels_push('.'))"
```

Runs under **chr1s4** and attaches the chr1s4 copies of both datasets
(cross-account attach is rejected, `kaggle_usage.md` #13). `enable_gpu` is true
only because Kaggle CPU workers get no internet (#3); the build is CPU-only.
If a run is already live, `yes | kaggle kernels delete <slug>` first — a
re-push STACKS sessions (#25).
