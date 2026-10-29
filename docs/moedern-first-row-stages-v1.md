# One-document actual-stage MoEdern diagnostic v1

This is a source-prepared diagnostic of the unchanged1.5M CPU/F32 runtime,
not a production correction or a new reference. The
[retained eighteen comparisons](https://github.com/CrispStrobe/CrispEmbed/blob/0d9526de458ed81eb32383f2b87ff7ef19af2e0b/tests/results/moedern-colbert/three-doc-compare-v2-review.json)
show that document24726600 alone both reproduces its frozen original final
vectors within0.002/0.99999 and remains different from native93. At input
position58 (token29084, selected-vector row57), component19 differs by
0.022829396242741495. The other two documents' fresh-original drift remains
preserved and cannot replace their frozen gold.

The [protocol](moedern-first-row-stages-v1.json) authenticates the exact failed
[compare-only producer40834748271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834748271),
both whole artifact ZIPs, all nine small members and every raw ZIP member before
framework import or model access. Its three-document packet is retained unchanged;
a new one-document packet refers to its original and native anchors. All six
first-document old/fresh-original/native matrix hashes remain authenticated.
No full-corpus reference is recaptured.

The sequence is mandatory:

1. Execute one ordinary original call using the published checkpoint and frozen
   original source/runtime. Compare both final matrices with their frozen original
   targets using actual `crispembed_diff::Ref` and independent metrics at unchanged
   absolute0.002 and minimum row cosine0.99999.
2. Execute the plain native probe, preserving its six successful computations,
   default-off gates, invalid-input recovery and context ownership controls. Both
   final matrix SHA256 values must equal the retained93 outputs exactly.
3. Write the two ordinary barrier identities. Only then may any original or native
   module/operator observer be installed.
4. Execute two original observed calls, an unhooked repeat and the ordinary public
   selected-vector call. All complete feature tensors must match the ordinary
   feature raw bits, dimensions, dtype and strides. Both observed traces must have
   identical module arrays and actual operator arguments/returns, including SDPA.
5. Execute the observed native probe with eight successful computations, one
   first-input observer abort before embedding math, recovery and exact ordinary
   final anchors. Compare the earliest actual stage differences, magnitudes and
   routing indices; preserve numerical rejections as nonzero results after saving
   reports.

The maximum is **five targeted original neural-network calls**, **fourteen full
native computations** (six plain, eight observed), one pre-math observer abort,
six rejected encode attempts and six rejected initializations. The native probe
makes21 encode attempts. Failures retain partial counters and phase reports.
`original_rerun` is true as soon as a targeted original call is attempted;
`original_full_corpus_recaptured` remains false.

The observer captures actual `aten._softmax.default`, `aten.topk.default` values
and indices, and `aten.div.Tensor` while inside each original FFN module. It
records their real operator schemas, arguments and outputs. Snapshots run with
Python dispatch modes disabled, and the original operation receives its unchanged
arguments. No after-hook recomputation is router gold. The SDPA wrapper forwards
its original arguments and captures the returned tensor; it does not select a
backend. Signed-zero-sensitive raw SHA256 fingerprints cover complete features,
operator arguments and returns, and repeated captures. Unexpected operation
counts, missing stages, nonfinite tensors or changed observation outputs reject
the diagnostic.

The pinned publisher file is32641 bytes with SHA256
`528afffbd33210845ed521b12af5569f9b6d0b0eb3f176beebb3efd83cee6668` at
[revision9d360b8b, lines235–248](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/blob/9d360b8b88ae11bea5784bf9012ff509fdd5c2da/modeling_modernbert_hash_moe_v10.py#L235).
Its assignment `tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-9)` is an
out-of-place tensor/tensor division, matching the expected `aten.div.Tensor`;
it is not augmented assignment or `div_`. The exact function text SHA256 is
`c0860b582652a7d084acc5dea84a94173efaa433b71fab9b19666acd48e1c0d8`.
Source AST inspection finds one softmax, one top-k and one division expression.
For this document307×128×16=628736 is below the publisher's500000000 dense-path
threshold, and307 tokens fit a single8192-token chunk. The eval FFN delegates
to that path, optional shared linear/GELU projections and dropout identity;
its earlier FFN normalization lies outside the FFN observer context. No other
source-level softmax or division is present in this active route. These are
source facts; the hosted dispatcher must still observe exactly the declared
operations and reject unexpected counts.

A separate ordered `sdpa_dispatches` list records ATen operator names and schemas
exposed to `TorchDispatchMode` only during the real functional SDPA call.
Input/output snapshots remain outside that active interval and disable dispatch
modes, so bookkeeping operations are excluded. Both observed calls must produce
identical lists. This identifies the observed ATen flash/math route when exposed;
it neither forces a backend nor establishes ordinary or historical-binary
backend identity, and does not claim to enumerate operations hidden inside a
single native kernel. The eight primary router/SDPA captures and all mapped
comparison counts remain unchanged.

The mapping has44 actual F32 stage comparisons, two exact integer top-k
comparisons, three exact integer input observations and six native-only
magnitude diagnostics. Native attention probabilities and routed sums lack a
separate original observation; the two native selection margins are derived
quantities and have no original gold. No padding, reshaping, new gold, changed
router policy or wider gate is permitted.

The [standalone test build](../tests/moedern-first-row/CMakeLists.txt) reuses the
frozen93 source and root-profile GGML archives at a06. LLAMAFILE remains ON,
native host tuning OFF, configured ISA flags preserved, and scalar source
contraction OFF. The ordinary archive excludes observer symbols and stage strings;
the observed archive supplies the positive control. The probe copy changes case
cardinality and adds test-executable call accounting; production source is
unchanged.

The initial [GHA workflow](../.github/workflows/moedern-first-row-stages-v1.yml)
used two workers with14GiB available RAM/24GiB free disk before payload and a
60-minute job limit. Resource profile v2 below supersedes only the disk policy;
the RAM, worker, numerical and timing requirements remain unchanged. The [Kaggle CPU adapter](../tools/moedern-first-row/kernel.py) requires an
isolated CPython3.12 environment, a reviewed exact checkout with its pinned
submodule, and the genuine provider environment. The operator supplies an
approved credential-only input through `CRISPEMBED_ARTIFACT_TOKEN_FILE` or
`GH_TOKEN`; no account, token, private input path, model dataset or software cache
is part of the public protocol. API credentials are discarded before signed
artifact-storage redirects. External exact-version submitted-script, status and
all output-page authentication is required before admitting any Kaggle result.
It never sets `GITHUB_ACTIONS`.

The framework versions and Sentence Transformers source remain frozen and are
checked by the existing original loader, including every loaded checkpoint tensor.
This does not assert identity of all historical wheel binaries or CPU hardware;
the two measured ordinary barriers are mandatory regardless of setup success.
All models, arrays and builds stay hosted; small reports and exact source pins
are sufficient for independent review. No production policy, quantization,
retrieval qualification, publication or default changes follow from this source
preparation or from a local operator sensitivity result.

## Resource profile v2: one retained 307-token case

The [initial Kaggle v1 failure](../tests/results/moedern-colbert/first-row-kaggle-disk-initial-review.json)
at `e7ff72f1a2fe1290f5d7b879d41e3fd5b6da26b8` occurs before the first command:
32,672,014,336 bytes of available RAM, 20,940,525,568 bytes of free output disk
and four CPUs satisfy memory/CPU requirements, but the inherited 24 GiB disk
requirement exceeds this provider's output volume. No retained data, packages,
weights or inference were accessed; the original failure stays unchanged.

This separately declared resource-v2 successor requires **10 GiB free disk** at
bootstrap, retained-input preparation and before build/model work. Available
RAM remains **14 GiB**, with at least two CPUs and the same two-worker/thread
policy. It is not a bootstrap-only bypass. All five original and 14 native full
computations, ordinary barriers, actual operations, masks, 44 F32/two integer
maps, all frozen targets and 0.002/0.99999 gates remain byte-exact in source.
The original observation, native probe, CMake and comparator files are unchanged.

The new conservative disk allowance is 4 GiB packages/caches, 2 GiB source/build,
2 GiB retained output, and 2 GiB free reserve. Known extents are small: retained
ZIPs total 10,760,103 bytes, the native model is 7,037,952 bytes, the original
checkpoint and Dense weights total about 6 MB, and the frozen tracked checkout
is 164,459,632 bytes before Git metadata/submodule/build files. The allowance is
an estimate with enforced admission/reserve checks, not measured peak usage.
No original corpus or other model size is admitted by this profile.

Retained output has a 2 GiB aggregate ceiling, checked before/after commands and
at phase boundaries, with a 2 GiB live free-space reserve. A 2 GiB inherited
single-file limit applies to child processes. Existing original trace limits
(512 MiB across the two traces), native per-case limits (128 MiB), fixed case and
call counts, and the bounded Ref adapters constrain each indivisible capture.
The aggregate check is not a claim of continuous filesystem monitoring. Exceeding
a cap stops subsequent work and preserves evidence; no canonical files are
removed to manufacture admission. Private bootstrap checks cover its whole
output root, including input/build descendants. Only known task-created scratch
and build trees may be cleaned after their logs and results have been retained.

The resource profile and new merged source base are explicit in the protocol;
old e7 source remains available by immutable Git identity. The successor private
kernel is prepared for the same existing service kernel's next version, with
an unchanged approved credential-only input. Submission still requires source
review and a fresh whole-account capacity check by the coordinator. No GPU,
provider spoofing, numerical waiver or automatic retry is authorized here.
