# PCS main-gate casing failure and future capture

Main `be20e49797c68c4a9fcc765d2f3b719257b48580` is **not CI green**.
The [retained punctuation job](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44247044277/job/134608391665)
passes post/pre/segmentation decisions 67/67 each, but casing is 66/67 and
decoded text 5/6: `I'm OK, how are you? I'm fine.` becomes
`I'm oK, how are you? I'm fine.` The following fullstop gate was not reached.
PR235's separate reviewed head `7ab522d6` passed 43 checks with two expected
skips before its owner merge; the reviewed and merged trees are identical.
The push-only PCS gate does not run on PRs. PR236 remains merged at `04e84458`.

## First quantity actually known to differ

The original [harness](../tests/pcs_parity.py) prints comparison row 37's
significant casing bits as reference `11`, native `01`. The frozen
[reference](../tests/regression/pcs/blueprint_ref.txt) places that row at
line 3, local token 3, piece `▁ok`, ID 3664. It removes the ignored `▁` bit
and padding; the first remaining bit is original slot 1, which controls `o`.
The [runtime](../src/pcs.cpp) reads that same slot during ASCII reconstruction.
This is a real character-case failure, not a padding discrepancy to exempt.

This mapping is **reference-derived**. The PCS harness parses reference IDs but
does not compare native IDs/pieces, unlike the preceding FireRed gate. Matching
67-row counts do not prove identical tokenization or chunk boundaries.
The observed false native bit means the native `logit > 0` predicate was false;
its value, magnitude, margin and absolute error are unavailable. It could be
negative, zero or NaN. The original reference exports bits, not casing logits.
This is the first retained discrete discrepancy, not a proven earliest
numerical divergence in the encoder, segmentation or casing head.

The native casing head consumes the encoder hidden state plus a sentence-start
bit: true for the first token of each chunk, otherwise the previous token's
segmentation **hard argmax**. The gate instead compares a `0.05f` probability
threshold. Conditional on the expected alignment and no chunk reset at this
row, the preceding threshold-false row constrains the native hard bit to false;
it still does not supply historical native token/chunk evidence or original
head inputs. Future capture records both quantities directly.

## Attribution limits

Seventy-nine relevant PCS/core/CMake/harness/reference/workflow files and ggml
gitlink `3a27b6c665b191cb230be367db5c52d81dd3e2ac` are identical across
`04e84458`, reviewed `7ab522d6`, and merged `be20e497`. This is source binding,
not proof that execution inputs or floating-point outputs were identical.
The failing log identifies GNU C/C++ 15.2.0 and `-march=native`, but does not
retain CPU identity/effective ISA, exact model revision/hash, binary hashes,
native token mapping, casing logits or numerical margins. The workflow fetched
the named Q4_K-imatrix model without an immutable revision. Neither model drift
nor native code generation is uniquely established as a cause.

## Source-only instrumentation

`CRISPEMBED_BUILD_PCS_CAPTURE=ON` exposes the explicit
`punct-ab-capture` target, excluded from default builds and installation.
[The adapter](../tools/pcs_gate_capture.py) requires unique insertion anchors
and creates a build-owned copy of PCS with diagnostic calls. It does not edit
the byte-shared production `src/pcs.cpp`, original harness, corpus, references,
assertions, scorer, tolerances or ordinary workflow. Removing only those
insertions recovers the original source exactly.

The test-only [observer](../tests/pcs_gate_trace.h) is default off through
`core_env::on("PCS_GATE_CAPTURE")`. Its path is `PCS_GATE_CAPTURE_PATH`; the
observer refuses to overwrite a previous capture. It records actual request
text, token IDs/piece bytes, chunk offsets, SBD logits/probability/hard decision,
each of the 16 casing logits as exact F32 bit patterns, its actual boolean and
sentence-start conditioning, plus compiler identity and selected ISA macros.
The reader records signed margins to the unchanged zero threshold and absolute
magnitudes, checks complete inventories and reports native token mismatches
separately. It uses the exact F32 `0.05f` tie semantics and retains nonfinite
values as bit patterns without relabelling them as negative margins.

The explicit future `fetch-model` command uses the existing Hub provider and
requires a full 40-character revision and predeclared file SHA-256. It records
that immutable revision, filename, hash and byte count. It never resolves a
mutable branch implicitly or substitutes another recipe. The offline
`provenance` command binds that receipt to the actual file and binary, compiler
executable/hash/version, CPU flags, CMake cache, all compile commands, generated
observer source and build-library hashes. The library inventory is not proof
of dynamic-loader selection.

No fetch, model execution, CMake configure/build, CI/provider request, workflow
dispatch or push was performed for this source-only work. The small
[controls](../tests/test_pcs_gate_capture.py) pass all 13 cases using synthetic
text/bit records and mock provider/compiler responses, not model weights or
native inference. Pinned formatting, environment inventory, Python syntax and
original runtime/gate/reference byte checks also pass.
The diagnostic target remains **uncompiled and unlinked**. Its original-driver
link substitution and observer OFF/ON neutrality require future validation;
unchanged arithmetic source is not a numerical neutrality proof.

## Future gate, only with separate execution authority

First freeze the exact same recipe/corpus/reference and an authenticated model
revision/hash, build/profile and source. The failed historical model/binary
cannot be recovered or identified from its filename alone. Compile both the
original `punct-ab` and explicit diagnostic target with the declared profile
and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`; retain actual compiler commands and CPU
identity. Preserve the historical native profile separately from a portable
Linux x86-64/AVX2 control; changing the profile is not a repair of the old result.

Require original-driver versus diagnostic-driver OFF, then diagnostic OFF/ON,
neutrality of all existing head dumps and decoded output before attributing
any trace to the original computation. Run the **unchanged** `pcs_parity.py`
with the explicitly chosen diagnostic driver in its expected `punct-ab` slot,
`PCS_GATE_CAPTURE=1`, and a new capture path. Preserve its original exit status,
stdout and stderr, all four existing dumps, trace, model receipt and build
provenance. Analyze capture independently; it cannot replace or turn a failed
decoded gate green. Original same-input numerical head/conditioning evidence
is still needed to attribute an earlier numerical cause. No new default,
quality/performance qualification, release, app pin or model distribution follows.

## Authorized native control — 18 November 2026

The [actual frozen native run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44272906599)
at `e6cfda21df2e0f4ab759ccbf954b4286cb62c82a` compiles and links both
drivers with exported compile commands. Original versus diagnostic OFF and
OFF versus ON match all four head dumps and decoded bytes exactly. The
[receipt](../tests/results/pcs/frozen-native-control-20261118.json) retains
artifact identity, every row-37 casing F32 bit/margin and the actual mapping.
All six requests/67 token IDs and pieces align; request 3 is one 11-token chunk.

GNU 15.2.0 native Release on Intel Xeon 6975P-C passes post/pre/seg/casing
67/67 each and decoded 6/6. Row 37's `o` logit is `0x3d1ff79e`,
+0.0390545055270195; `k` is +0.39889898896217346. The preceding SBD hard
and threshold decisions are both false, and sentence-start conditioning is false.
The immutable model revision is `27f17edbd7e33949f42a9a4fe4322b5af8172e6b`,
SHA-256 `faee4bf8469a502362b5967a2cfeb7a1732539fbd999c63b489d00efff3531da`.

This is a positive control, **not reproduction or repair** of the retained
66/67, 5/6 failure. Production PCS, references and acceptance checks remain
byte-identical. The [declared AVX2/original-graph follow-up](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44273793050)
is pending; no CPU-specific cause, production fix, fullstop or main closure
is established by this positive result.
