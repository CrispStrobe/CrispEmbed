# Typed Gemma CPU first-Q4 preflight: F32 capture and adapter failures retained

Exact Kaggle CPUv1 at source`92ee94a77c95df33b272abfcbf4f365550c20a47`
finishes its F32-only work and rejects before any Q4 conversion. The
[independent review](../tests/results/gemma-decision/first-q4-cpu-v1-initial-review.json)
authenticates submitted script/Get/Status/every List page and90 selected small
files,933,234 bytes. All459 outputs remain remote; no raw arrays/models were
downloaded locally. This initial worker remains structurally incomplete.

Actual source/provider/a06/build/activation/render/Ref controls, pinned F32
model and original fixture authentication pass. Five three-call probes perform
145 actual stage comparisons with exact18/20/25/19/549 token counts and ordinary/
observed/repeat pooled-byte neutrality. All five final pooled embeddings pass
unchanged original cosine/relative-L2 gates. Case0 pooled maximum absolute error
is2.55182385e-7, minimum cosine0.999999999997759.

The saved intermediate panel is negative:20 of145 stages pass,125 fail.
PLE/final_norm/tokens/pooled pass in each probe; earlier post_embed and24 layer
captures fail. Case0 post_embed max1929.546924/cosine-0.074559955 versus original
magnitude11.844038, while its final pooled output remains correct. The existing
snapshot already names/marks tensors as outputs but reads their buffers only
after complete graph computation. This pattern motivates live-at-compute
capture; it does not by itself establish an arithmetic or buffer-lifetime cause.
Keep these exact negative comparisons.

The C++ typed panel exits0 and completes30 explicit bank encodes,16 head-on-
original controls and four successful decision requests (43 source-derived
internal encodes). Python then rejects its bank adapter: the frozen MoE helper
limits every axis to512, while Gemma bank embeddings are768 wide. No typed
parity verdict was completed. Native F32 total is88 source-derived calls;
original calls, Q4 candidates and quantizer model runs remain zero.

Next is a separately closed test-only live snapshot repair and a correctly
bounded768-wide lossless adapter. First activation is only the existing18-ID
probe: preserve its actual ordinary/observed/repeat IDs and pooled bytes,
capture28 selected tensors immediately after actual computation, retain the old
end-of-graph reads separately, and compare all29 original stages. Do not add
numeric graph nodes, alter weights/math, replace gold or weaken gates. Only
actual retained source/live contrasts can support a buffer-reuse explanation.
After independent review, a separately declared full145/eight-typed/30-bank
F32 retry may precede the two unchanged Q4 recipes. No automatic model fan-out.
