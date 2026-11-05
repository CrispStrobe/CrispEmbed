# Proposed complete LightOn4B rotary arithmetic profile

This lane reproduces fixed model arithmetic from configuration and the pinned
original CPU math backend. It does not read images, checkpoints, reference
activations or OCR outputs. Complete-domain source/reader proof passed; the default-off production integration
passes focused source-fed/fallback/build verification. The rejected SLEEF
rotary alternative remains off.

The source configuration is
`lightonai/LightOnOCR-3-4B@b71010f095dc3735de7ee6b9969bff8f8f50b39a`,
SHA `99b4c363a998c2d8c5ed5b3f3a40ec5cb97a156034df5002d0d789c74324f5f0`.
Its vision width 1024 and 16 heads yield head dimension 64. The pinned
[Transformers implementation](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
constructs `VisionRotaryEmbedding(head_dim // 2)` with default theta 10000,
so there are 16 inverse frequencies. It computes F32 outer products of integer
coordinates and inverse frequencies, then F32 cosine/sine using oneMKL 2024.2
build 20240605, AVX2, HA mode in the frozen Torch 2.13.0 CPU environment.

Production `geometry` caps each axis size at 1024. Thus every emitted coordinate
is in 0..1023. Merge size 2, per-frame area at most 1024 ordinarily or 2048 in
the admitted experiment, total tokens at most 4096, up to 16 grid rows and 16
frames per row do not increase that coordinate bound. Frames repeat the same
two coordinate axes. A complete bank therefore contains
`1024 coordinates * 16 frequencies * 2 functions * 4 bytes = 131072 bytes`.
It covers the entire admitted axis domain, including extreme 1024-by-2 and
2-by-1024 geometry. No image-dependent selection or fitted geometry is used.

The published generator is [generate_lighton4b_rotary_table.py](../tools/generate_lighton4b_rotary_table.py).
With the exact reference dependencies installed, it can be reproduced using:

```sh
python tools/generate_lighton4b_rotary_table.py \
  --output lighton4b_rotary_table_data.inc \
  --manifest rotary-table-generation.json
```

The generator requires Linux x86_64 and enforces Torch `2.13.0+cpu`, Torch git
`cf30153c4c131c8164ee7798e5022d810682e2cb`, NumPy `2.3.5`, oneMKL version/build,
AVX2 dispatch and the source configuration hash. It emits integer encodings of
F32 values to avoid compiler decimal conversion. The manifest records all raw
and text hashes. The generator and generated numerical asset use this
repository's MIT license; the existing source-configuration Apache-2.0 provenance
is retained. No MKL implementation code or binary is copied or redistributed.
The reference environment uses the separately distributed Torch/oneMKL software;
no product Torch/MKL dependency is introduced.

The proposed reader has a separate default-off gate and requires the exact
width/head/merge/theta profile, supported x86_64 AVX2/FMA capability and nearest
rounding, valid lengths and in-range coordinates. Unsupported or disabled
requests leave output vectors unchanged, allowing the preceding runtime path
to execute. The source profile is intentional; this is not generic libm parity
or a cross-platform quality claim.

The targeted proof regenerates the asset twice, checks every constant against
the direct original MKL route, verifies compiled bytes, and compares the native
reader to original Torch rotary construction for every admitted single-frame
height/width pair and every admitted total token count, plus temporal and
multigrid boundary packings. It also checks fallback/unsupported conditions and
that the native reader links no Torch/MKL library. Geometry-domain coverage is
mathematical; finite layout checks additionally guard against backend packing
surprises. No model forward or decoded-quality claim follows from this proof.


[Complete-domain proof 42096520795](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42096520795)
at `79fbff4e74ba084c35804ba96178a15d2e3e2da0` passes duplicate generation,
32,768 direct-MKL constant controls and compiled byte verification. Native reader
and original source agree on all 681,459,712 F32 values across 3,873 layouts,
including all 3,280 admitted single-frame h/w pairs, all 1,024 admitted total
token counts, and temporal/multigrid boundaries. All 17 original reader fallback
guards pass; no native Torch/MKL dependency is present. The 128 KiB raw asset SHA
is `30ea508c92d92a8400604e276fac161a1a85f7e89cd79877f227684a6d330e0c`.
The [generation manifest](lighton4b-rotary-arithmetic-manifest.json) records exact
original generation sources; current generator additionally rejects other OS/
architecture profiles, and must reproduce the same asset bytes.

The separate `CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE` gate now selects
these configured constants before preceding rotary alternatives. The complete
reader was moved into production unchanged, and disabled/unsupported lookup
leaves output vectors untouched before the original generator runs. A focused
integration job repeats the whole domain through the actual production helper,
checks ordinary/unsupported fallback and precedence over rejected SLEEF, then
checks retained same-input original tables and first-block output. No image is
used in generation; retained neural inputs are verification only. A full model
forward and decoded quality remain separate qualifications.


[Production integration 42098420346](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42098420346)
at `7648267c79453b277b6ae3f8c43aa54769903264` regenerates the committed asset
and repeats every domain case through the actual runtime helper. All values and
22 fallback/precedence checks pass; native dependencies remain clear. Original
source cosine/sine, pooled attention and complete first-block output now match
bytes, while the preceding disabled path is preserved. Native/context/decoder
controls pass. Ordinary Q4 inference will separately measure actual stage,
first-score, decoded and timing impact; no quality claim follows yet.
