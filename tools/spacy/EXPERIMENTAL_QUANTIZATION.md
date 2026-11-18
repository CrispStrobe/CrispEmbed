# W7 experimental storage qualification

The hosted `tools/ci-heavy/spacy_w7_quantize.py` driver creates a private
alternate library. It does not modify the production F32 sources, standalone
CMake target, accepted model, distribution manifest or package. Its generator
rejects an unexpected F32 loader source hash. Only the generated library reads
`CRISPEMBED_SPACY_QUANTIZED`; this additional gate defaults OFF and uses
`core_env::on()` with the usual false spellings. `CRISPEMBED_SPACY=1` remains
necessary. The generated target is not a production installation target.

The experiment admits closed F16, Q8_0 and Q4_0 storage profiles, authenticates
the complete file before loading, and expands non-F32 rows once into owned,
finite F32 arrays. It retains the scalar F32 numerical implementation. This
tests compressed storage and loading, not accelerated quantized matrix
multiplication. Loaded compressed buffers and expanded arrays can increase
resident memory. No performance improvement is assumed.
The Q4_0 recipe retains the quantizer's existing Q8_0 protection for `embed.*`
tensors. This closed mixed storage inventory is admitted explicitly; unrelated
Q8_0 tensors are not admitted by a Q4_0 declaration.

The existing `crispembed-quantize` implementation is compiled in isolation.
The private source overlay prevents its generic convolution flattening from
changing spaCy's rank-four parser tensor shape. All other quantizer behavior
is preserved and both original/generated sources and binary hashes are saved.
The GGUF writer canonically omits a leading unit axis from `parser.lower.pad`.
Only the experimental profile admits that exact shape-equivalent rank; every
non-unit dimension and the loaded four-dimensional shape remain fixed. The
initial hosted F16 attempt's rank-admission rejection is retained separately.

`quantized_parity.py` first calls the unchanged F32 bundle/reference validator.
Its explicit candidate admission adapter replaces only the independently
recorded quantized model hash and size; all other converted files, the frozen
development reference, stage comparisons, decoded scorer and numerical budgets
remain unchanged. Each stage also enters the real `crispembed_diff.h` bridge;
its cosine and magnitude values supplement the original absolute-error gates.
Neither comparator's threshold is relaxed. Candidate reports identify the
alternate library and generated loader provenance separately from unchanged
production source hashes.

The 16-case W1 development oracle may be used for this experiment. Neither the
consumed native acceptance split nor either consumed W6/W7 grammar panel may be
run by this driver. Candidate rejection is a terminal assessment result, not a
reason to replace F32, widen a tolerance or claim grammar quality. Hosted
diagnostic artifacts are labeled separately from accepted distributions.
Kaggle is unused because this closed backend has no real GPU inference path.
