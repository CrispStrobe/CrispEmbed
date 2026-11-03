# Josev Q4_K: final-output cosine and the sensitivity sweep

Goal: the smallest Q4_K-class Josev GGUF whose final output clears the owner's
Q4_K bar ([precision-acceptance.md](precision-acceptance.md), > 0.94; Q8_0 > 0.97),
minimum over the four frozen cases, against the frozen original reference
(`tests/results/josev2b/original-reference.json`, v5 capture; no original
forward reruns).

## What "final output" means for Josev

Josev answers typed questions; its head emits, per field, a small logit vector
(2 or 3 options) whose softmax is the typed probability vector that the answer
serializer rounds into the JSON response. Two cosines are reported per case:

- **Judged: head-logit cosine** — cosine of the concatenated per-field logits of
  the case (the vector that produces the typed response).
- **Diagnostic: final-norm cosine** — global cosine of the backbone final-norm
  hidden state (seq x 2048). This is what the v4 table quoted (0.9367–0.9606);
  it is stricter and the head is fairly robust to it.

Cosines come from the actual comparator's per-stage moments (original/native L2
and RMS of the difference); the F32 control gives 1 - 3e-12. Head-logit vectors
have only 4–7 elements, so the judged cosine is a weak discriminator on its own:
it is always reported next to the final-norm cosine, the probability error,
option argmax agreement and complete typed-JSON equality.

Every recipe keeps the 292 critical tensors F32 (token embedding, norms, the
30 M-parameter head, non-projection tensors) and stores the 150 decoder
projections as real Q8_0/Q4_K (the C quantizer rejects any fallback; every
projection width is a multiple of 256). The protected F32 part is 2,079 MiB, so
the floor for any recipe is high: all-Q8_0 is 3,462 MiB, all-Q4_K 2,808 MiB.

Build: `tests/josev-q4k` forces `GGML_NATIVE=OFF` (portable, runner-independent);
lscpu is in each receipt. Runtime: the Josev composed/packed path has only an
explicit CPU backend, so there is no CUDA arm to run (see below).

## Earlier runs rescored (no new inference)

[`q4k-final-output-rescore.json`](../tests/results/josev2b/q4k-final-output-rescore.json)
rescored every earlier quantized Josev panel with the same method:

| Run | Recipe | MiB | Head-logit cos min | Final-norm cos min | Typed JSON exact |
|---|---|---|---|---|---|
| v4 (Kaggle CPU) | 150 Q4_K | 2,808 | 0.99300 | 0.93673 | 0/4 |
| 40787077302 | A: MLP Q4_K, attention F16 | 3,446 | 0.99680 | 0.97015 | 0/4 |
| 40787077302 | B: gate/up Q4_K, down F32, attention F16 | 4,436 | 0.99917 | 0.98000 | 0/4 |
| 40787077302 | C: gate/up L20–23 Q4_K, rest F16/F32 | 5,126 | 0.99999 | 0.99830 | 1/4 |
| 40838785499 | 150 Q8_0 | 3,462 | 0.99995 | 0.99944 | 1/4 |

Recipes A/B/C store the non-Q4 projections as F16, so they are as large as or
larger than all-Q8_0; they are not useful Q4_K artifacts.

## Sweep (hosted CPU, run 41615751516, AMD EPYC 9V74, portable)

Base: all 150 projections Q8_0; each row lowers one group to Q4_K. Receipt:
[`q4k-sweep-41615751516.json`](../tests/results/josev2b/q4k-sweep-41615751516.json).
F32 control: head-logit and final-norm cosine 1.0, 4/4 typed exact.

| Recipe | Q4_K/Q8_0/F32 tensors | MiB | vs Q8_0 | Head-logit cos min | Final-norm cos min | Max prob. error | Typed exact | Argmax |
|---|---|---|---|---|---|---|---|---|
| q8-all | 0/150/292 | 3,462 | — | 0.99995 | 0.99944 | 0.0059 | 1/4 | 4/4 |
| attn-q4 | 78/72/292 | 3,240 | -222 | 0.99801 | 0.96622 | 0.0500 | 0/4 | 4/4 |
| gate-up-q4 | 48/102/292 | 3,174 | -288 | 0.99927 | 0.97969 | 0.0130 | 0/4 | 4/4 |
| down-q4 | 24/126/292 | 3,318 | -144 | 0.99908 | 0.98809 | 0.0427 | 0/4 | 4/4 |
| mlp-q4 | 72/78/292 | 3,030 | -432 | 0.99705 | 0.96992 | 0.0191 | 0/4 | 4/4 |
| layers 0–5 Q4 | 37/113/292 | 3,297 | -165 | 0.99272 | 0.96250 | 0.0942 | 0/4 | 4/4 |
| layers 6–11 Q4 | 38/112/292 | 3,300 | -162 | 0.99443 | 0.98516 | 0.0677 | 0/4 | 4/4 |
| layers 12–17 Q4 | 37/113/292 | 3,297 | -165 | 0.99936 | 0.98970 | 0.0170 | 0/4 | 4/4 |
| layers 18–23 Q4 | 38/112/292 | 3,300 | -162 | 0.99987 | 0.98828 | 0.0046 | 0/4 | 4/4 |
| all-q4 (= v4) | 150/0/292 | 2,808 | -654 | 0.99300 | **0.93673** | 0.0378 | 0/4 | 4/4 |
| all-q4, down Q8_0 | 126/24/292 | 2,952 | -510 | 0.99739 | 0.94911 | 0.0358 | 0/4 | 4/4 |
| all-q4, layers 20–23 Q8_0 | 125/25/292 | 2,917 | -545 | 0.99233 | 0.94349 | 0.0380 | 0/4 | 4/4 |
| all-q4, layer 23 Q8_0 | 143/7/292 | 2,833 | -629 | 0.99294 | 0.93999 | 0.0370 | 0/4 | 4/4 |

The all-q4 row reproduces v4 to all printed digits on a different CPU and
runner (v4 ran on Kaggle CPU), as expected from the portable build.

Sensitivity: the early layers (0–5, then 6–11) cost the most head-logit
cosine; attention and `down_proj` cost the most final-norm cosine per byte.
Worst intermediate stage for the Q4 recipes is `backbone/layer/23` (min row
cosine 0.56–0.76, the attention-sink row) or the scalar `head/residual_scorer`
row, whose 1-element rows report -1 when a near-zero residual score flips sign.
The decisions do not change: option argmax is 4/4 in every recipe.

## Result

- **Q8_0** (`q8-all`, 3,462 MiB): head-logit 0.99995, final-norm 0.99944 —
  clears the Q8_0 bar (> 0.97) on both metrics.
- **Smallest recipe that clears the Q4_K bar on the judged head-logit
  metric:** plain all-Q4_K (`all-q4`, the v4 storage, 2,808 MiB, -654 MiB vs
  Q8_0): 0.99300. It fails the stricter final-norm metric (0.93673).
- **Smallest recipe that clears 0.94 on both metrics:** `all-q4-l20-23-q8`
  (2,917 MiB, 0.99233 / 0.94349), with a thin final-norm margin.
- **Recommended policy: `josev_q4k_down8_v1`** (`all-q4-down-q8`, 2,952 MiB,
  -510 MiB vs Q8_0): 0.99739 head-logit / 0.94911 final-norm, every projection
  Q4_K except the 24 MLP `down_proj` at Q8_0.

None of the Q4_K recipes is token/text exact on the typed response: complete
typed JSON is 0/4 because the four-decimal probabilities and scores move (worst
probability error 0.036–0.094); option choices are 4/4 everywhere. The all-Q8_0
file is 1/4 exact. The strict 0.002 / 0.99999 stage budgets remain failed
diagnostics for every quantized recipe and are not relaxed.

## Policies

`CRISPEMBED_QUANTIZE_JOSEV_POLICY=<name>` with the base type `q4_k`
(`tools/quantize.cpp`):

| Policy | Recipe | Contents |
|---|---|---|
| `josev_q4k_down8_v1` | `all-q4-down-q8` | all projections Q4_K except `mlp.down_proj` Q8_0; everything else F32 |
| `josev_q4k_mlp_v1` | `mlp-q4` | MLP Q4_K, attention/linear-attention projections Q8_0; everything else F32 |

The policy refuses any other architecture or base type, selects the
`crispembed-josev-packed-mixed-v2` profile and the standard file-type label
(the runtime admits header 14 with real Q4_K storage), and inserts its rules
before any `--tensor-type`. `tests/test_josev_quant_policy.py` checks that
each rule table resolves all 442 tensors to the measured recipe; the sweep
re-quantizes with the policy and compares bytes (`mlp-q4`: byte identical,
sha256 `8e65bcfd…`; `josev_q4k_down8_v1`: byte identical, sha256 `adc7fa47…`).

Confirmation run 41635417771 (AMD EPYC 9V45, portable,
[receipt](../tests/results/josev2b/q4k-sweep-41635417771-down8-confirm.json))
rebuilt and re-quantized `all-q4-down-q8` on a different CPU model: same GGUF
sha256 and identical cosines to every digit (0.9973891795 / 0.9491092304).

```sh
CRISPEMBED_QUANTIZE_JOSEV_POLICY=josev_q4k_down8_v1 \
  crispembed-quantize josev2b-systemone-f32.gguf josev2b-systemone-q4_k-down8.gguf q4_k
```

## CUDA

Not run. `josev2b_composed.cpp` loads the packed GGUF onto an explicit
`ggml_backend_cpu_init()` backend and the qwen35 backbone path is CPU-only;
there is no CUDA path for the quantized Josev model to measure. Adding one is
runtime work outside this lane.

## Publication

[`josev2b-q4k-publication-v1.json`](josev2b-q4k-publication-v1.json) prepares a
cosine-gated `Q4_K_MIX` recipe (`josev_q4k_down8_v1`). It is not published;
`tools/ci-heavy/josev_publish.py` currently uploads only the F32 file and would
need a quantized branch before dispatch.

## Reproduce

Push a change to `docs/josev2b-q4k-sweep-request.json` on a `perf/q4k-josev-*`
branch, or dispatch `.github/workflows/josev-q4k-sweep.yml` with `recipes=`
(it needs an Actions token for the frozen original-reference artifact, which
`heavy-cpu.yml` does not provide). Initial rejections are kept:
`q4k-sweep-initial-filetype-rejection.json` (41588071746, an inherited exact
file-type check; the source F32 has no `general.file_type`) and
`q4k-sweep-initial-counts-rejection.json` (41590988284, a result-dict key clash).
