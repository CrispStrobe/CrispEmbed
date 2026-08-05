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

## Result (kernel v1, 2026-08-05, ~40 min, all 7 models, 0 failures)

**Coverage — the defect is gone everywhere.** Every re-collected `.imatrix` has
`leaf_N=0`, and the quantizer's own stdout confirms the tensors took importance
(`imatrix vectors loaded` → `N with imatrix`; the gap is the F7 alias expanding
one merged QKV vector into three per-layer weights):

| model | vectors loaded | `N with imatrix` / quantized | before |
|---|--:|--:|--:|
| ms-marco-MiniLM-L-6-v2 | 36 | **36** / 38 | 18 |
| ms-marco-MiniLM-L-12-v2 | 72 | **72** / 74 | 36 |
| mxbai-rerank-xsmall-v1 | 72 | **72** / 74 | 60 |
| mxbai-rerank-base-v1 | 72 | **72** / 74 | 60 |
| bge-reranker-base (control) | 72 | **72** / 74 | 72 |
| jina-reranker-v2-base-ml | 48 | **72** / 74 | 36 |
| bge-reranker-v2-m3 | 96 | **144** / 146 | 72 |

**A/B, mean Kendall-tau (n=30 pairs × 6 docs) and mean `|dscore|` vs the
full-precision gold**, old published run → this run:

| model | q4_k (no imx) | q4_k+imx old | q4_k+imx new | iq4_xs old | iq4_xs new |
|---|--:|--:|--:|--:|--:|
| ms-marco-L-6 † | .9689/.145 | .9333/.0069 | .9644/.130 | .9467/.0095 | .9644/.138 |
| ms-marco-L-12 † | .9556/.169 | .9289/.0076 | .9600/.166 | .9156/.0090 | .9556/.165 |
| mxbai-xsmall | .7289/.038 | .6533/.0315 | .6978/.0268 | .7422/.0317 | .6978/.0302 |
| mxbai-base | .7911/.102 | .7156/.0580 | .7644/.0737 | .7556/.0501 | .7378/.0498 |
| bge-reranker-base | .9511/.446 | .9511/.3165 | .9511/.3165 | .9333/.3694 | .9333/.3694 |
| jina-reranker-v2 | .9289/.134 | .9422/.1032 | .9422/**.0792** | .9378/.0979 | .9467/**.0903** |
| bge-reranker-v2-m3 | .9244/.400 | .9244/.3053 | **.9556**/**.2245** | .8933/.4101 | **.9467**/**.2546** |

† ms-marco `dscore` is NOT comparable across the G7c boundary: the old run
scored the broken 1-layer head (±0.2 range), this one the corrected 2-layer
head (±11). Tau is.

`bge-reranker-base` reproduces its old numbers to 4 dp on all four arms — the
expected control result (its imatrix was already clean), and it doubles as
proof the pipeline is deterministic, so the other deltas are real.

Clear wins: **bge-reranker-v2-m3** (tau .9244→.9556, dscore −26 %) and
**jina** (dscore −23 %). Both ms-marco models improve on tau. **mxbai is the
one soft spot** — see below.

**Spot-check** (`ms-marco-MiniLM-L-6-v2-g7c-f7-spotcheck.txt`), uploaded
artifact re-downloaded from HF and scored against ONNX Runtime on
`Xenova/ms-marco-MiniLM-L-6-v2`:

```
ONNX ref                       [0] +8.846 [1] -10.886 [2] +7.401 [3] -11.225 [4] -5.200 [5] -4.944
…-g7c.gguf (f16 base)          max|delta|=0.0003  score_range=20.071  HF-scale OK
…-q4_k-imatrix-g7c-f7.gguf     max|delta|=0.0994  score_range=20.091  HF-scale OK
…-iq4_xs-g7c-f7.gguf           max|delta|=0.3495  score_range=19.785  HF-scale OK
```

### Recorded, not fixed: mxbai q/k importance has the wrong provenance

The mxbai coverage digest shows `blk.N.attn_q.weight` and `blk.N.attn_k.weight`
collected **separately** from `enc.N.attn.qkv_merged.weight`. That is DeBERTa-v2
disentangled attention: `src/crispembed.cpp` applies `q_w`/`k_w` a second time to
the relative-position embeddings (`Pk = mul_mat(k_w, P)`, `Pq = mul_mat(q_w,
P_p2c)`), and those matmuls name the weights. `tools/quantize.cpp` prefers a
direct name match over the alias, so post-F7 mxbai's q and k take importance
collected **only over rel-position inputs** (T×T rows, dominating any merge)
while v takes the correct hidden-state vector. This is the likely cause of
mxbai being the only family where a tail arm regresses (iq4_xs tau .7422→.6978
xsmall, .7556→.7378 base). Options if pursued: prefer the merged alias for
DeBERTa, or accumulate both into one vector. Needs its own A/B — not touched here.

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
