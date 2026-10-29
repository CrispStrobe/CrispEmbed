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

The [GHA workflow](../.github/workflows/moedern-first-row-stages-v1.yml) uses two
workers with14GiB available RAM/24GiB free disk before payload and a60-minute job
limit. The [Kaggle CPU adapter](../tools/moedern-first-row/kernel.py) requires an
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
