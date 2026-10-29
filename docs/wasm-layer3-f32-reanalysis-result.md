# Retained F32 weightcast negative: verified reanalysis

[Compare-only run40831082898](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40831082898)
succeeded at sourcebf4e237e0bfb6ed84192fe9d9bd768b782d1050a. It verified the
retained negative from [operator run40791643276](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40791643276),
whose optional F64 diagnostic had failed after its actual operator and comparator
outputs were already saved. This successful job qualifies the reanalysis process,
not the candidate operator or a model/runtime change.

The [unchanged small report](../tests/results/packaging/wasm-layer3-f32-reanalysis-report.json)
and [independent review receipt](../tests/results/packaging/wasm-layer3-f32-reanalysis-review.json)
bind the service/source and8,324-byte review ZIP SHA. The hosted job authenticated
all39 retained raw members, six original quantized-output anchors, exact cast
coefficient SHA and both lossless comparator adapters. The saved actual Ref
report is byte-identical to the original failed run's report. No model, operator
or original forward and no build ran during reanalysis. The optional F64 dot is
explicitly omitted; no F64 value was relabeled or coerced into an F32 reference.

All25 fixed-gate comparisons are retained: **7 pass and18 reject** at unchanged
max absolute0.002/minimum row cosine0.99999. Independent scalar review confirms
all reported F32 gates and Ref/independent metric agreement. The first rejected
row in comparator order is `native1.old_native1.candidate`: maximum absolute
0.07845640182495117, RMS0.008432731541593425 and minimum row cosine
0.9997049885682548 in the independent metrics. This is a comparator ordering,
not an additional claim about the earliest whole-network divergent layer.

The cast-weight native/browser same-input differences remain at most
1.9073486328125e-6, but the candidate changes the old quantized native target by
about0.07846. Thus the local cross-backend positive cannot justify a production
weightcast path. All prior whole-model negatives remain open. The next approved
source-only study tests native-order norm recovery before any computed-activation
quantized projection; it makes no numerical claim before its own hosted result.

Only small JSON/log/service evidence was reviewed locally. Raw arrays, model
bytes and binaries remain remote, with no local package install, compilation or
inference. The review rechecks reported scalars and identities; it does not claim
an independent local recomputation from raw arrays. No release, default or
publication qualification follows.
