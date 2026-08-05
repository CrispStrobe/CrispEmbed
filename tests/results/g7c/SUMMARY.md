# G7c (expanded) — ms-marco rerankers: missing BertPooler stage (2026-08-05)

## Defect

HF `BertForSequenceClassification` scores `classifier(tanh(pooler(CLS)))`.
The shipped `cstr/ms-marco-MiniLM-L-{6,12}-v2-GGUF` artifacts (f16 and every
quant) carried only the 1-layer `classifier.{weight,bias}` — the checkpoint's
`bert.pooler.dense.*` was never converted — so the native runtime scored
`dot(CLS, w) + b`. The LEARNINGS 2026-07-03 claim that ms-marco runs the
2-layer tanh path did NOT hold for the shipped artifacts (1-layer head,
verified by reading the GGUF tensor lists).

Measured impact (L-6, 1 query x 6 docs, reference = ONNX Runtime on
`Xenova/ms-marco-MiniLM-L-6-v2` — a faithful export of the HF model):

| doc | ONNX ref | shipped q8_0 |
|---|--:|--:|
| Berlin population | 8.820 | 0.078 |
| Berlin metro | 8.565 | 0.072 |
| NYC population | -5.016 | -0.142 |
| Berlin Wall | -9.409 | -0.066 |
| Paris | -11.244 | -0.164 |
| Cats | -11.224 | -0.269 |

Score calibration destroyed (±0.2 instead of ±11 — thresholding/sigmoid use
broken); tail ranking reordered (two discordant pairs, Kendall-tau 0.733 on
this set). Top-2 happened to survive here.

**Reference caveat:** the local miniconda torch could NOT serve as the
blueprint — `AutoModelForSequenceClassification` forwards produced NaN /
bus-errors / nonsense orderings on this box (fresh re-download did not help).
All parity numbers here are against ONNX Runtime (CPU EP), which produces the
model's known-good behavior.

## Fix (converter-only; no runtime change)

`models/convert-bert-to-gguf.py`: for a BERT-family (non-DeBERTa) reranker
with a 1-layer classifier and a checkpoint pooler, emit the head as the
runtime's already-verified 2-layer path — `classifier.dense` = pooler.dense
-> tanh -> `classifier.out_proj` = classifier — which is HF-exact
(same structure the XLM-R rerankers use). Also: suppress the pooler emit for
2-layer (Roberta-head) rerankers (HF never runs the backbone pooler there;
emitting it would make the runtime apply a gelu pooler HF does not), and
record the truthful `bert.pooler_act` (`tanh` for BertPooler; DeBERTa keeps
`config.pooler_hidden_act`). DeBERTa ContextPooler path (mxbai) unchanged.
Non-reranker embedding conversions keep the historical pooler emit.

## Acceptance

Re-converted both models (ollama-mode naming, same as the shipped artifacts,
so the published `.imatrix` files match — 18 (L-6) / 36 (L-12) quantized
tensors take importance; a `--crisp` conversion renames the encoder and gets
0 imatrix matches, which is how this was caught).

**f16 vs ONNX reference (6 pairs each):** L-6 max |delta| 0.0006, L-12 max
|delta| 0.0009, rankings identical. The crisp-mode and ollama-mode f16 score
identically (mode is score-neutral).

**Quants (2 queries; full logs `quant_scores.txt`):** q8_0 within ~0.007 of
f16 both models; q4_k / q4_k+imatrix / iq4_xs within ~0.4 with only deep-tail
near-tie swaps (both tail docs at approximately -11.2, gap < 0.05) — the known
4-bit tail behavior; q8_0 stays the default. q4_k+imatrix now differs from
plain q4_k (imatrix actually applied).

**Shipping (G3 precedent — new names, old files kept so released binaries'
pins keep working):** 10 artifacts uploaded as `*-g7c.gguf` (f16 + 4 quants
per model) to the two cstr repos, README notes added. Registry aliases
re-pointed and `model_hashes.h` re-pinned (4 pins). End-to-end verified with
the rebuilt worktree binary: fresh `--cache-dir` download of
`ms-marco-MiniLM-L-6-v2` (q8_0-g7c) and `ms-marco-MiniLM-L-12-v2-iq4xs`
(iq4_xs-g7c) — SHA-256 pass, `classifier head loaded (reranker=2-layer)`,
HF-scale scores.

## Recorded, not fixed

- The DeBERTa ContextPooler path applies **tanh-approx** GELU where HF `gelu`
  is erf-exact (same class as the granite projector finding) — mxbai-only,
  needs its own A/B before touching.
- The ms-marco `.imatrix` files remain pre-F7 (no attention q/k/v coverage);
  a t19-style re-collection is the standing F7b leftover for rerankers.
- jina/bge (2-layer Roberta heads) are structurally correct as shipped; their
  checkpoints' backbone poolers were never emitted (their GGUFs have no
  pooler tensors — that is the correct state).
- The local miniconda torch environment mis-executes BERT-class forwards
  (NaN/bus error/garbage) — do not use it as a parity reference; use ONNX
  Runtime or a remote box.
