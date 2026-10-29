# MoEdern retained real-input router operator replay

This [closed protocol](moedern-router-real-v1.json) follows the completed
[actual v4 capture](moedern-first-row-actual-v4-result.md). At document24726600,
position58/token29084, layer1 selects original experts2/4 versus native2/6.
Probabilities differ near a nonzero1e-7 cutoff margin, although normalized top
weights match. Routers are already F32. This is component attribution, with no
precision change, new model/reference inference, new gold or tie-policy change.

Only the two actual307-row router matrices, logits, probabilities, selected
indices/weights and original operator metadata are admitted. Original router
inputs have directly captured shape `[307,128]`. Native normalization is captured
as `[1,307,128]`; the source's existing flat vector is the byte-identical
`[307,128]` component view. No original tensor is flattened or changed to fit.
Token IDs/masks, both4096-byte router weights and captured operator arguments/
returns are authenticated before component evaluation. Each router's actual
native-loaded F32 bytes must equal the corresponding loaded-original weight SHA.
The only fetched model is the immutable7MB published F32 GGUF, not the original
checkpoint. No backbone/full-model forward is called.

The worker accepts an operator-supplied exact-version mounted capture root.
Private kernel identity/attachment metadata stays outside Git. It admits29
pre-pinned selected files and two4912-byte native I64 files. Those native hashes
were absent from the small capture metadata: they are explicitly **first
observed on the hosted worker**, after canonical little-endian schema/range/
unique-expert checks and pointwise validation of all307 rows against the actual
SHA-pinned original I64 captures. Layer0 is identical; layer1 differs only at58,
with original2/4 and actual native2/6. No array/gold is synthesized or replaced.
This authenticates the declared selected scope, not all400 outputs or a ZIP.
The producer's original completed/neutral metadata and external exact-v4 source/
script/Get/Status/List proof remain required lineage.

The test executable includes the unchanged production engine source and calls
its actual `linear` and `softmax` helpers. The exact native stable-sort/margin/
denominator/division loop is extracted with source-span/hash evidence into a
temporary test include. It preserves the production exact-tie rejection and
adds no production probe. The same root CPU profile as the actual v4 capture uses a06 GGML,
LLAMAFILE ON, configured ISA flags and contraction OFF for the scalar wrapper,
matching the actual native capture profile. The ordinary production macro
excludes observers. Only test binaries are built; root source/API/defaults stay
unchanged. The original Torch2.8CPU/NumPy2.2.6 wheel lock is reused byte for byte.
Torch invokes functional bias-free linear and real `aten.softmax.int`,
`aten.topk.default`, sum-plus-epsilon and `aten.div.Tensor`, with interop1.

For both layers, each backend first replays its **own** actual captured input,
logits and probabilities, using one/two threads and two repeats. Every retained
own logits/probability/ordered-ID/normalized-weight byte must match; Torch's
actual top-values/denominator bytes are checked too. Native raw selected values
are separately labeled diagnostic gathers, not retained native gold. Every
returned file repeats exactly. Cross-input arms are blocked unless all own
anchors pass, so a new-host component mismatch cannot masquerade as causality.

Then both natural input sources are admitted to both backends at three entries:
input→linear→softmax→selection/division; captured logits→softmax→selection/division;
and captured probabilities→selection/division. Completed coverage comprises96
component runs, with at most32 linear,64 softmax,96 selection and96 division
batch operations. These are288 primary component operations, not model calls;
denominator sum/add operations and native scalar row/division counters are
reported separately. The two pre-math malformed/NaN controls execute no operator.
Both full-model and original full-model forward counts remain zero.

Own anchors, same-input backend differences and the natural-input effect within
each backend produce at most144 actual `crispembed_diff::Ref` F32 comparisons and
72 exact ordered-ID comparisons. Three actual comparator controls exercise
positive, numeric and wrong-shape exits. Absolute0.002/minimum row cosine0.99999
remain unchanged; independent F64 metrics retain absolute/mean/RMS errors,
magnitudes and worst component indices. A native exact-tie rejection remains an
actual source outcome with partial row counters/absent outputs; no tie guard is
weakened and no missing output receives a passing flag. All comparisons are
saved before nonzero numeric exit. This does not qualify the old full model.

The hosted owner is
[tools/ci-heavy/moedern_router_real_v1.py](../tools/ci-heavy/moedern_router_real_v1.py).
After reviewed checkout/submodule initialization and an isolated CPython3.12
venv, invoke it with `--reviewed-source`, `--capture-root`, `--out` and `--scratch`.
The genuine Kaggle CPU admission does not spoof GitHub or require an artifact
credential: it reads the attached exact-v4 capture. Authentication precedes any
numerical import, package/model fetch or array interpretation. Runtime guards
require Linux x64, two CPUs,4GiB available RAM/8GiB free disk; the overall hosted
budget is90 minutes. All arrays, binaries, source fragments and manifests remain
remote; only needed small reports are collected. This source checkpoint has not
executed the component replay.

A reviewed hosted CPU invocation (the variables are supplied privately by the
operator; this does not submit a kernel) is:

```sh
python -I tests/test_moedern_router_real_v1.py
export CRISPEMBED_CAPTURE_SOURCE="$REVIEWED_SOURCE"
python -I tools/ci-heavy/moedern_router_real_v1.py \
  --reviewed-source "$REVIEWED_SOURCE" --capture-root "$CAPTURE_V4_ROOT" \
  --out "$TASK_OUTPUT" --scratch "$TASK_SCRATCH"
```

The parent must attach the exact capture version, check fresh whole-account
capacity, review the source/protocol and obtain the new submitted-script/
Get/Status/every-List-page proof. No local packages/models or dispatch are needed
for the source controls. Pending execution must prove the actual build, loaded
weights, own anchors, all96 runs, comparisons and retained outputs. A different
host/operator result or own-byte failure blocks causal interpretation. Even a
positive component result cannot qualify the old full-model/retrieval negative.
