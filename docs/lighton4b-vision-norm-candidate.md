# LightOn4B default-off BF16 vision norm candidate

The experimental BF16 vision path and its normalization alternatives remain
unqualified. Legacy shared vision stays unchanged. The new path requires
`CRISPEMBED_LIGHTON4B_VISION_BF16=1`; all vision gates use value parsing
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


The scoped linear correction applies only to actual nn.Linear: all complete
QKV/out/up/down outputs match original bytes with1/4 threads in42032345547.
Conv3D uses a separate source four-partial, left-associated accumulation under
`CRISPEMBED_LIGHTON4B_VISION_BF16_CONV`; actual42037359930 verifies every573,440
patch output bit with1/4 threads and independently of NN/PATCH alternatives.
`CRISPEMBED_LIGHTON4B_VISION_BF16_PATCH_LINEAR` preserves the earlier rejected
Conv-as-linear path; CONV takes precedence when explicitly enabled. All remain
default-off, and old gate bytes remain tested.

The retained combined42033651799 was decoded-negative: Conv-as-linear regressed
patch20→34 values; merged RMS improved0.00973525→0.00772041, but first-score
RMS worsened0.560135→0.568035 and both texts remained`<table>`. Exact operators
do not establish current composed or decoded quality. The assembled control gate
replays norm/linear/Conv and decoder controls together, preserving exact source
ZIP13414500337 remotely for90 days in this script only; unrelated retention is
unchanged. No source reference arrays are shipped in production artifacts.


[Final assembled42038802495](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42038802495)
atedbd2b782 passes all norm/full NN/full Conv byte and old-gate controls, native
builds/controls, and decoder20 controls. Its
[receipt](../tests/results/qwen35-ocr/lightonocr3-4b-vision-linear-integration.json)
records exact source ZIP preservation through3 February2027 and actual runtime
hashes. A separate ordinary Q4 smoke now isolates these vision corrections with
the decoder candidate explicitly off; operator success does not establish decoded
quality, and every runtime gate remains off by default.

The later attention lane preserves independent gates and source-fed controls:

| Gate suffix after `CRISPEMBED_LIGHTON4B_VISION_BF16_` | Scope and evidence |
| --- | --- |
| `ATTENTION` | Original QK/PV reduction order; partial pooling differences 65 to 20. |
| `SOFTMAX_AVX2` | Original vector exp/reduction, used with `ATTENTION`; all source-fed probability and pooling bytes exact. |
| `GELU_AVX2` | Source-derived double-float tanh and original cube grouping; all 2,293,760 source activation bytes exact. |
| `MERGER_ERF_AVX2` | Source erf polynomial/exp for final merger only; original/actual merger byte correction and focused gate/build validation passed. |
| `ROTARY_TABLE` | Complete 128 KiB configured coordinate/frequency arithmetic, generated without images or activations; complete-domain and actual production/fallback/source-first-block controls passed. |
| `ROTARY_AVX2` | **Rejected alternative, keep off.** SLEEF trig differs from original MKL backend. |

The `SOFTMAX_AVX2`, `GELU_AVX2` and `MERGER_ERF_AVX2` gates require x86_64 AVX2/FMA and nearest
rounding. Unsupported
platforms retain the preceding path. The rejected rotary alternative additionally
limits its math to the fixed theta/dimension and coordinates below 125; that
range restriction does not qualify its numerical result. Product code links
neither Torch nor MKL. Source helpers remain under tests; runtime arithmetic
adaptations retain the source licenses.

[Exact-attention integration 42055105090](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42055105090)
passes full source probability/reciprocal/pooling byte controls, old gates,
native/context builds and decoder controls. Ordinary Q4
[42056882067](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42056882067)
is nevertheless decoded-negative: the candidate reduces first-block differences
to 40 but worsens merged/first-logit error, and still produces `<table>` against
original `<figure>`. These are two-token smokes, not complete-document validation.

[Retained first-block attribution 42060487656](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42060487656)
explains all 40 actual candidate differences: rotary alone 32, activation alone
8, combined 40, with exact original and combined byte controls. Source-derived
GELU then passes
[final integration 42064278334](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42064278334),
including old gates and the expected first block with original GELU and native
rotary. Direct reference-only oneMKL 2024.2 `vmsCos`/`vmsSin` HA calls match all
2,000 original grid angles; installed SLEEF vector trig and fused/nonfused scalar
adaptations do not. This records the true backend dependency without shipping a
reference lookup table or adding MKL to the product. A separate Q4 A/B enables
verified GELU while explicitly disabling rejected rotary and decoder candidates.

[Ordinary GELU Q4 42065352208](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42065352208)
preserves all 30 fixed baseline hashes. The actual first block has the expected
32 rotary differences; merged RMS improves to 0.00775425 from baseline 0.00973525,
but first-logit RMS 0.56432682 remains worse than baseline 0.56013479 and both
smokes still decode wrong `<table>`. Quality remains negative. Its test-only
activation replay still evaluated legacy GELU; the independently verified
production helper and ordinary native quality path are separate. The observer
is corrected for future runs without changing runtime or repeating this smoke.

The final all-verified-vision
[Q4 run 42070806605](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42070806605)
includes merger erf and the corrected source-fed activation observer. All eight
source-fed first-block operators now have zero differences, and all 30 fixed
baseline hashes reproduce. Ordinary runtime still has the known 32 rotary
differences. Merged RMS is 0.0077542505 against baseline 0.0097352499; first-logit
RMS 0.5643236502 is still worse than baseline 0.5601347910, and both outputs
remain `<table>` against the original `<figure` prefix. Native elapsed is
105.17 s baseline / 115.84 s candidate for this single sample. This is neither
full-document quality nor a throughput benchmark. Complete source, timings and
limits are retained in the final merger-erf negative receipt. Gates remain off;
no release or app-pin promotion follows.


Complete-table continuation now establishes all 29 recorded vision-stage byte
identities on the one Q4/Q8 image. Q4 remains decoded-negative downstream;
corrected Q8 first-logit RMS improves to 0.0742289135 against historical original
F32-vision 0.13049504 and earlier BF16 candidate 0.2005369525, with source/host
differences explicit. The Q8 `<figure` prefix was already correct historically.
These are two-token observations, not full-document/EOS qualification. See the
[complete rotary profile](lighton4b-rotary-arithmetic-profile.md) and its exact
receipts for supported domain, source generation, fallback, raw hashes and limits.
