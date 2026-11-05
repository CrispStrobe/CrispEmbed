# LightOn4B default-off BF16 vision norm candidate

The experimental BF16 vision path and its normalization alternatives remain
unqualified. Legacy shared vision stays unchanged. The new path requires
`CRISPEMBED_LIGHTON4B_VISION_BF16=1`; all three vision gates use value parsing
and default off. `CRISPEMBED_LIGHTON4B_VISION_BF16_NORM` preserves the rejected
separate-affine alternative. `CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM` selects
verified original AVX2 BF16 paired-vector moments, contracted final scalar
merging and both contracted affine operations. No default is promoted.

[Source-header42026490991](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42026490991)
atd7415c4db isolates actual original PyTorch2.13 compiler contraction.
[Production-helper42027253266](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42027253266)
at9c11a924c matches both573,440-value norms and all1,120 mean/rstd entries per
norm exactly, including raw SHA controls, with unset/empty/0 retaining old bytes.
Reference arrays and source-header probes are test-only and remain remote.

[Ordinary pinned A/B42028668863](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42028668863)
at6c3cb6ed9 retains exact original reconstruction and historical native baseline
hashes. Contracted norm reduces merged RMS0.0223861→0.00973525 (max2.46484→1.21094),
but Q4 first-logit RMS worsens0.545670→0.560135 and both independent two-token
outputs remain incorrect`<table>` versus original`<figure` prefix. This is an
operator correction/component improvement with **negative decoded selection**,
not model, EOS/document, quantization or publication qualification. Initial
AVX512 original-control failure42028040544 remains preserved. Source replays pin
original AVX2 ATen/oneDNN/MKL dispatch; native builds stay portable. No
cross-platform, speed, release or app-pin claim follows.

The existing native document branch integrates this bounded gated candidate
without removing mixed-policy, frontend, inspection or previous mapper work.
Targeted `tools/ci-heavy/lighton4b_vision_norm_integration.py` must pass assembled
source compilation, exact norm/source-header/raw-SHA controls, gate controls,
BF16 finite/tie rounding, native-context/image-compose negatives and environment
documentation. It downloads retained operator vectors only, with no model
inference. Full model results from the feature producer do not substitute for
this assembled-source check or for future decoded qualification.


The assembled-source gate passes in
[42033147414](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42033147414)
at43b2d2c35: exact norm/raw-SHA/moments/gate checks, finite/tie rounding, native
context9 and image-compose45 controls, compiled ordinary runner argument checks
and fresh environment documentation. The
[integration receipt](../tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-integration.json)
pins runtime bytes. Initial missing read-token and test-helper fallthrough failures
remain separate. This closes compilation/component controls only; decoded quality
is still negative and all defaults stay off.
