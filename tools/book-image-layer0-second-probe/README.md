# Cache-closed second image chunk diagnostic

This separate test executable derives the proven first-image observer. It changes
only the observed tile to prompt offset20 (image rows16–31), with native cache
counters20→36. It retains whole266-row ordinary prefill, every replayed layer0
tile/main hidden equality, previous-cache immutability and exact final cache/policy
closure. Projected-core replay still must match actual block attention and state.
No production runtime or gate changes; no decoded generation or F32 text model.

The model-free window audit identified the first larger layer0 residual at image
row23, inside this second tile. Observation here tests the earliest known window;
it does not introduce a case-dependent runtime rule. Source actual252-row image
operators are sliced at image rows16–31 with unchanged physical head mapping.
All82 historical processor/vision/prefill/firstscore hashes must close before
interpreting any captured difference. Native before/after cache arrays remain
uninterpreted against source until their layouts are independently proven.

Five model-free controls cover source drift, unchanged ordinary source, generation
reversibility, worker import and observation scope. Execution uses the existing
immutable controller plus an explicitly hashed test overlay when a new public
controller commit is unavailable. Compiled runtime/ggml/model/input/gates remain
identical to the successful first capture; retain all overlay hashes/provenance.
