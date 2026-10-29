# MiniCPM retained vision analysis v2: complete structural proof

[Run40831080193](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40831080193)
completed at sourcecec1faebd39582a6103774dd9598b3e26d6533f1. Its
[independent review receipt](../tests/results/minicpm-ocr/retained-analysis-v2-review.json)
binds the526,540-byte small artifact, all134 member identities and the exact
947,005-byte analysis report. No model or original/native forward ran during
this analysis. Runtime83 and the prior19c/cec sources remain unchanged.

All23 retained arms now have six actual Ref comparisons with independent F32
metrics: **138 total,99 pass and39 reject** under unchanged0.002/0.99999.
The six old line arms and their36 comparisons match the initial partial
analysis exactly. Only the17 previously incomplete arms required102 new
comparisons. All23 original/native packed pixel streams are bit-exact.

The original merger returns a list and the frozen hook saved only its first
list element. That tensor maps exactly to the native overview prefix; it is
never padded or replicated across the native concatenation. Five earlier
vision stages retain their full concatenated patch rows. All88 later native
merger-part spans remain explicitly **unmapped**, with source/span hashes,
geometry, finite checks and magnitudes. This corrects the earlier structural
failure without inventing new original tensors or altering numerical gates.

F32, F16 and retained Q4 arms first exceed the strict budget at vision layer26.
F32 maximum errors at that boundary range from0.01654 to0.03345. Q8 arms first
exceed at vision layer0, with maxima0.09980–0.12817. The worst reported Q8
layer26 comparison has maximum639.7119140625 and minimum row cosine
0.9383880578630239 against original absolute maximum2037.9144287109375.
These magnitudes remain failures; structural completion is not numerical or
release qualification. The earlier23 decoded-exact results remain distinct
from these intermediate negatives.

The preserved canonical parent remains its immutable remote artifact; no second
canonical copy,23-arm recapture or original rerun was made. The next separately
versioned continuation may execute only the seven absent arms: tall/Q4 graph1
and tiny/F32 graph0/1, F16 graph0/1, Q8 graph1 and Q4 graph1. It must retain the
old models, source/compiler profile, caps and gates, authenticate this completed
structural proof first and preserve every existing numerical rejection.

Review used only the small ZIP, JSON/log inventories and reported scalar
metrics. It rechecked all138 reported gates/Ref metrics and the exact six prior
rows; no raw arrays, binaries or weights were downloaded or recomputed locally.
There is no model, publication, default or application-pin qualification.
