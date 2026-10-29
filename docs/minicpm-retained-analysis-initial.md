# MiniCPM retained analysis: initial structural partial

[Run40783734766](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40783734766)
reported workflow success at caller`1c8df48082d561b2f2c1698d64cdca47b9486594`,
checking out source`19c3f7bdf6d452614721946adefb95f6d3cf0180`. Its actual
analysis report has`completed:false`: six line arms completed36 actual Ref
comparisons, while seventeen multi-crop arms stopped during pair assembly.
The [initial receipt](../tests/results/minicpm-ocr/retained-analysis-initial.json)
preserves all288 small-member hashes, shapes, exact pixel outcomes and existing
Ref numbers/negative flags. Green workflow status did not establish completion.

The frozen original hook retains only the first merger output tensor. The
source-proven reason is its tuple/list handling, rather than repeated per-part
merger calls: Transformers5.21.0`MiniCPMOcrMerger.forward` returns
`processed_features`, a list ordered by`target_sizes`. The single
`self.merger(...)` call returns that list. Frozen
[reference.py](../tools/minicpm-ocr/reference.py) selects`o[0]` for tuple/list
outputs, then its first-call guard keeps the prefill observation. Thus the
original merger NPY covers only the overview. The original reference remains
immutable; its file hash is
`aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32`.

For paragraph/table, the original merger is`[252,1024]` while native concatenates
`[1260,1024]`; tall is`[252,1024]` versus`[2332,1024]`. Line has one overview and
both shapes`[235,1024]`, so all six line arms completed. The five earlier tower
observations cover full patch sequences and retain that full mapping. No Ref
metrics were produced for any of the seventeen failed arms because merger
shape rejection occurred before the batched comparison call.

The six completed line arms retain numerical negatives: F32/F16/Q4_K first
strict failures are at vision layer26; Q8_0 first fails at layer0. Correcting
reference coverage cannot erase those measurements or qualify quantized models.
All twenty-three packed-pixel comparisons were byte-exact. No model, original
or native forward ran in this retained-array analysis.

The source explanation uses the pinned public5.21.0 module,37602 bytes,
SHA256`45d60914094cdc51fab67dc8edcf85a36a4db61a770eaca891cc820675fcf352`,
retrieved earlier as an individual compressed source entry after
[official PyPI metadata](https://pypi.org/pypi/transformers/5.21.0/json).
Only that source text was inspected; the whole wheel was not downloaded,
installed or imported for this review. Native83 appends merger/debug rows in the
same overview-first part order. Actual original array shapes corroborate the
scope; no absent original part is synthesized.

The2706333-byte review ZIP12863177057 has SHA256
`6f13aaa56ea348e3d4db4e3ef33b2c39c1041ef25326e1a9e3347a10fc1b6cda`.
Canonical artifact12863940299 remains remote:6023649790 bytes, SHA256
`12d460253d53c89216dc291ffaf4c90f832f8cfc85d10ce9767ef6328ca5d339`,
expiring26 January2027. Only small reports and source text were reviewed locally.
The [separate v2 correction](minicpm-graph-retained-analysis-v2.md) carries the36
completed comparisons and computes the102 missing pairs without repeating any
original/model/native forward or changing runtime arithmetic.
