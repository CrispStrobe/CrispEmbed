# Native HF-layout weighted Q4_K_M-style candidate

This is an experimental, default-off native coordinate profile, not a model
release, public alias or publisher-byte-equivalent import. Its exact type
schedule matches the authenticated publisher Q4_K_M header: 216 Q4_K + 33 Q6_K
text matrices (including one shared Q6 embedding/output head), 177 F32 text
parameters and 297 protected F32 vision tensors. The existing 32-Q6/Q8-head and
24-output recipes remain unchanged. Current unqualified plain Q4 is not promoted.

The [recipe](../tools/lighton4b_imatrix_q4km_recipe.py) validates all 33 selected
names against the [bounded public type census](../tests/fixtures/lightonocr3-4b-source/publisher-q4km-type-census.json).
The head remains one physical tensor with identical embedding/head pointers.
No additional output matrix or Q8 donor is accepted. The 32 body Q6 matrices
and 216 Q4 matrices must all consume the approved importance vectors. The head
has no vector in the source file and intentionally uses unweighted Q6.

```text
crispembed-quantize ORIGINAL_F16 CANDIDATE q4_k --lighton4b-imatrix-q4km-style --imatrix ADAPTED
```

This dedicated flag rejects overrides, legacy headers, absent/unapproved stats,
and any source except the immutable original-derived unquantized F16 container
SHA `757ad5d0…`. It constructs the named schedule itself, verifies every final
tensor type, records 248 actual applications before atomic publication, and
stamps source/statistics/coordinate/name-digest provenance. It never requantizes
an existing Q4/Q8 file. The standard file-type bucket15 describes its mixed type
schedule; its distinct `lighton4b-weighted-q4km-style-q6head-v1` tag remains
mandatory. It must not appear as a generic user-selectable Q4_K_M model.

Native admission requires both `CRISPEMBED_LIGHTON4B_Q6_K=1` and
`CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD=1`, exact typed metadata, all 249 named
matrix types and the tied head. Missing counts/source tags, wrong dtypes,
untied output, old profile tags, conflicting modes and unset gates reject.
Public format-5 inference/release admission stays fenced separately.

The 24 GDN output importance vectors are inverse-permuted into HF grouped-V
coordinates. Upstream uses 128-column V tiles across 256-value quant blocks;
therefore independently quantizing the native layout does not reproduce the
publisher's block bytes. This recipe is type-schedule equivalent, not a proof of
same effective weights or upstream arithmetic. The separate physical-import
lane must establish that comparison without lossy quant-block repacking.

Payload growth versus plain Q4/Q8-head is only 10,946,560 bytes: the 32 Q6 body
promotions add 164,904,960 bytes and the Q6 tied head saves 153,958,400 bytes.
Expected combined size is about4.042 GB plus small metadata, with protected
vision unchanged. The previous 18 GiB disk/8 GiB available RAM guards suffice;
conversion expands at most one tensor, never a complete F32 model.

Before a checkpoint run, one Kaggle CPU model-free suite must verify the new
schema/metadata/closed-CLI controls together with unchanged old recipes and
independent Q6 embedding/tied-head operators. No GitHub CI or new PR is used.
A later same-calibration candidate may measure table/Arabic first scores with
exact regenerated f873 and frozen-score controls, then a separately budgeted
complete decoded comparison. The previous 45-minute table timeout is not a
quality result. Matching the independent llama.cpp Q4_K_M output/quality floor
and complete EOS/cap evidence are required before any shipping discussion;
component metrics alone do not satisfy that requirement. Heldout data remains
outside this candidate-selection lane.
