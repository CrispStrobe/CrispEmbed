# Typed Gemma live snapshot repair: frozen design, implementation pending

The [machine-readable design](gemma-live-snapshot-repair-design-v1.json) is
**not an executable validation contract**. No repair implementation, build,
new forward, quantizer run or dispatch has occurred. Activation remains false.
The [initial CPUv1 rejection](gemma-first-q4-cpu-v1-initial-result.md) and its
[independent receipt](../tests/results/gemma-decision/first-q4-cpu-v1-initial-review.json)
remain unchanged, including all125 failed intermediate comparisons and the
768-wide bank-adapter structural failure. The88 completed native encoder calls
are source-derived; the failed Python collector did not complete the typed
parity verdict. Original forwards and quantized model runs remain zero.

The repair branch starts from frozen producer
`92ee94a77c95df33b272abfcbf4f365550c20a47`, then merges the main claim before
source changes. Its design base is
`80382e35ed83682fe6eb6c81158468ccb75e9708`. The JSON pins unchanged runtime,
a06 GGML, old probes/workers/contracts, the initial receipt and the proposed
adapter dependency by Git object. A future executable protocol must additionally
close every new observer, overlay, driver, workflow and control file before
payload access. These design pins do not make an unfinished worker activatable.

## Evidence and the source-backed hypothesis

All90 selected small files,933,234 bytes, were independently rehashed against
the initial collection: no mismatch. All five probes pass PLE, final_norm,
tokens and pooled embedding, while post_embed and all24 saved layer tensors
fail. In case0, all24 saved layers have exactly the same reported L2/min/max
as final_norm. The returned pooled embedding passes with maximum absolute error
2.55182385e-7. This supports an observer investigation; it does not establish a
model arithmetic failure or a confirmed overwrite mechanism.

The existing [snapshot code](https://github.com/CrispStrobe/CrispEmbed/blob/92ee94a77c95df33b272abfcbf4f365550c20a47/src/gemma_decision.cpp)
already marks tensors as outputs, then reads them after graph computation.
The exact [a06 allocator](https://github.com/CrispStrobe/ggml/blob/a06fc0bb46b73cc10011fc90ae4054aee8438aeb/src/ggml-alloc.c)
protects output nodes when reserving a fresh allocation plan. However,
`ggml_gallocr_needs_realloc` checks node/leaf counts and allocation sizes rather
than changed output flags. The probe runs ordinary first and observed second
on one context; scheduler reset does not discard the allocator's saved plan.
Reusing the ordinary plan despite new observed output flags is therefore a
specific hypothesis. Actual live-versus-late buffers must test it before any
cause is recorded as proven or any production allocator change is proposed.

## First activation only

One frozen original probe, case0 with18 token IDs, performs exactly three
native encodes in the existing order: ordinary, observed, repeat. No original
capture, typed encode, quantization, other probe or automatic full-panel fan-out
is permitted by this design.

A disposable test-only overlay will register the28 existing snapshot tensor
pointers and install a synchronized GGML scheduler evaluation callback. It must
copy each actual F32 result immediately after that connected node computes,
without numeric graph nodes, arithmetic changes, renamed runtime tensors or
reference injection. Record names, graph indices, operation/source connections,
shape/stride/output flags, storage-overlap groups, exact byte extents, finite
status and raw SHA256. Require exactly28 distinct connected nodes and one
completed capture per node, with explicit failure state checked even if a
callback stops computation without a backend error. Restore the callback on
all exits; exclude observer code from the plain archive.

The same observed forward retains the existing end-of-graph GGUF unchanged.
Compare immediate and late values, and separately compare the retained old
late values. Actual returned pooled bytes and IDs must match the prior case0
ordinary/observed/repeat anchors, and all three new calls must remain byte
neutral. Exact-version typed Get/Status/every List page and submitted-script
hash verification precede fetching selected retained anchors remotely. Old
raw-array hashes were not individually frozen in the small receipt: derive and
journal them honestly after authenticated retrieval, rather than claiming an
earlier hash pin. Keep all large files remote.

All29 original stages, including returned pooled embedding, retain actual
`crispembed_diff::Ref` plus independent metrics, minimum row cosine0.99999 and
maximum relative L2.003. There is no new universal absolute-error tolerance.
Report absolute/RMS/mean errors, magnitudes and the earliest live failure even
if the live observer does not repair the stage verdict. Preserve every initial
negative and make no precision, default or quality claim from this diagnostic.

## Bank adapter and follow-on gates

The new caller can reuse only the lossless I/O functions in the pinned
[Mini graph adapter](https://github.com/CrispStrobe/CrispEmbed/blob/92ee94a77c95df33b272abfcbf4f365550c20a47/tools/minicpm-ocr/graph_adapter.py), whose axes allow
300000, per-tensor elements16,777,216 and total payload256MiB. Preserve Gemma's
own comparator and numeric gates; Mini's numeric constants do not apply.
Exercise actual F32 shapes `[13,768]`, `[11,768]`, `[6,768]` through write/readback,
including signed-zero bits, and reject mismatched shapes, altered hashes,
nonfinite values, truncated extents and oversized shapes. Do not modify the
frozen MoE adapter with its512-axis bound.

After the first actual result is independently reviewed, retained typed banks,
head outputs and complete answers may support a separately closed zero-forward
collector repair. A subsequent full145-stage/eight-answer F32 activation and
any of the two original Q4 recipes require a new explicit review. No prior
original reference or threshold is replaced.

A fresh observer-only context or explicit fresh reservation with stable output
flags is a cheaper **future control** worth testing if the cached-plan hypothesis
is confirmed. It is not part of the three-call activation and has no extra
forward authorization here. Callback capture remains the approved first design.
