# MoEdern router operator audit v1

This is a separately declared, model-free CPU experiment. It cannot qualify a
model, remove the current exact-cutoff-tie rejection, establish old wheel byte
identity, select a production tie policy, or replace any frozen reference.
Dispatch requires source review and an immediate provider capacity check.

The reviewed full-corpus F32 runs rejected27M document2931832 and115M documents
14530534/16058322. Their reports retained no failing layer/token, router input,
logit/probability bits, selected expert IDs or margins. Router weights already
remain F32. Storage precision, propagated hidden/logit/softmax rounding and tie
selection are distinct questions; this experiment addresses only operators on
new authored inputs. No checkpoint, tokenizer, corpus, original model invocation,
native model forward or historical raw arrays are accessed.

The three model variants pin the same original modeling source SHA256
`528afffbd33210845ed521b12af5569f9b6d0b0eb3f176beebb3efd83cee6668`.
At the declared512-token bound,27M and115M evaluation use `_inferenz_dicht`:
router linear, softmax, `topk(2, dim=-1)`, then selected probabilities divided by
`sum + 1e-9`. Both original full-corpus receipts record Torch2.8.0+cpu. Their
installers pinned version/index, but did not authenticate the wheel or compiler
and C++ standard library. The new exact CPU wheel lock is a new environment
profile; it does not prove historical binary identity.

[PyTorch2.8 CPU TopKImpl](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TopKImpl.h)
uses `nth_element` for eight experts/top two. Its comparator examines values,
without an index tie key. The selected sorted prefix is `[begin, begin+k-1)`.
[CPU dispatch](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cpu/SortingKernel.cpp)
uses float accumulation for F32 input. [The documented contract](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)
does not guarantee stable tied indices. The small C++ adaptation retains the
upstream BSD notice and is test-only. Compiler/stdlib-dependent tied results
must be checked against the actual installed Torch; matching is never assumed.

The JSON protocol freezes exact finite F32 input bits before any output. Cases
include strict gaps, selected-pair ties with strict cutoff, rank2/rank3 ties,
all-equal values, signed zero and one-ULP differences. Every Torch input runs
contiguous, column-strided, row-strided and transposed, at one/two threads, three
repeats. All original operator observations finish before C++ operators execute.
Selection on identical probabilities and softmax-plus-selection on identical
logits remain separate. A third arm feeds Torch's exact softmax output into both
C++ selectors, isolating selection from softmax drift.

The native control uses actual pinned a06 ggml CPU softmax/argsort/gather/divide.
The source-derived candidate uses the exact E8/K2 branch, without inventing an
index ordering. Indices, selected-value bits and renormalized-weight bits are
checked separately. Lossless GGUF adapters compare probabilities, selected
values, weights and the eight-expert scattered route weights through actual
`crispembed_diff::Ref` and independent NumPy metrics. The unchanged0.002 absolute
and0.99999 row-cosine gates remain explicit. Reports include absolute/RMS/mean
errors, L2/min/max magnitudes, input/probability bits, indices and cutoff margins.
Route-weight comparisons expose different experts even when selected values tie.
Positive, numeric-negative, wrong-shape and malformed/nonfinite-packet controls
must execute. Numeric differences remain complete structured observations and
exit1; structural failures exit2. No candidate becomes production-qualified.

The hosted runner authenticates runtime Git objects, every new source, exact
wheel archive hashes and installed versions before numerical imports. It records
Torch Git/config/CPU capability, loaded Torch/libstdc++ hashes, compiler version,
compile commands, cache, binary hashes, install proofs and per-command logs.
The explicit CPU build uses LLAMAFILE ON, native ISA OFF, GPU/BLAS backends OFF,
and contraction OFF in the two test translation units. No root build/API source
or historical guard changes. Both reports and tiny authored arrays/adapters/
binaries survive failures for90 days in GHA; Kaggle keeps them in working output.

Use the standalone [workflow](../.github/workflows/moedern-router-topk-operator-v1.yml)
or render the [Kaggle CPU adapter](../tools/kaggle/moedern-router-topk-v1/kernel.py)
with one reviewed full source SHA. The adapter follows existing network-enabled
CPU numerical workers, uses a real Kaggle batch environment, and never spoofs
GitHub variables. If network delivery is unavailable, it fails; no alternate
credential or unverified payload fallback exists. Kaggle output requires external
explicit-version Get/Status/every-List-page/script authentication before public
acceptance. Environment admission alone is explicitly not service verification.

A later, separately authorized three-document capture must retain the actual
native layer/token coordinates, masks/scoring eligibility, normalized hidden,
F32 router weight identity, logits/probabilities and expert/cutoff bits before the
unchanged rejection. Same-input Torch operator replay could then test cause.
This audit neither reruns those documents nor establishes their cause.
