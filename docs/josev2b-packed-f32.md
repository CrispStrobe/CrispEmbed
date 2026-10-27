# Josev F32 packed converter preparation

This source-only lane repacks three immutable, reviewed components into one
GGUFv3 file. It performs no checkpoint conversion, quantization, tensor arithmetic,
framework import, network fetch or inference. Existing component/runtime code,
root CMake, bindings and historical guards remain unchanged. The packed file
requires a separately implemented and validated borrowed loader; existing
component loaders intentionally do not accept this architecture.

The exact input identities, all442 tensor names/ne dimensions/F32 payload hashes,
and source artifact lineage are frozen in
[`josev2b-packed-f32-profile.json`](josev2b-packed-f32-profile.json).
The schema comes from the published reviewed conversion receipts, not from a new
model execution. Input components are:

| Component | Bytes | SHA256 |
| --- | ---: | --- |
|320-tensor text backbone|7,527,324,576|`8dc144a89e4b2bbf253fe395eeb9bc4d1eb7b3a27474442847535e55c79b361c`|
|122-tensor joint head|121,900,352|`5b0a4bfe769526569102e3cde115887593875124c94bb723fb9087bc40c29507`|
|Zero-tensor tokenizer|9,954,944|`6fe488e538b172d2f3f831e7422714b7158602d9d9f2d34f336db389363c9562`|

Reviewed source evidence is the
[backbone40438416809](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40438416809),
[head artifact40429513945](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40429513945)
and [tokenizer40446469585](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40446469585).
Their declared original reference is51a2e16b and workerb2edf8c8. These component
proofs are distinct from composed text-to-answer qualification. Initial composed
[40447333269](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40447333269)
failed in its artifact adapter before inference and cannot authorize packing.
The corrected [40449265616](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40449265616)
at `ed23f1aeb5f448b80a89b567592986bffb951de9` subsequently passed all227 stages
and four full typed responses. Its independently reviewed `hosted.json` SHA256
`45689505bcfd6de6e32a3c353ac4a502f0c82826227b2100788e9ac9b15d0fd7`
is now explicitly frozen in the profile, along with service review artifact
12710133899 (101216B, ZIP SHA256
`755ad31a56a880347a5aa9742be5e5dcad06112924beafd1bd96c79ae6119979`).
Production packing requires that exact positive receipt before component access. No latest-source
lookup, negative-result acceptance or implicit promotion is provided.

The proposed borrowed-loader metadata contract is exact:

| Key | Type | Value |
| --- | --- | --- |
|`general.architecture`|STRING|`josev2b_systemone`|
|`general.name`|STRING|`Josev-2B native SystemOne text F32`|
|`general.license`|STRING|`apache-2.0`|
|`josev2b.packed.format`|STRING|`crispembed-josev-packed-f32-v1`|
|`josev2b.packed.profile`|STRING|`official-pytorch-text-batch1-512-no-extra-temperature`|
|`josev2b.packed.backbone_gguf_sha256`|STRING|Exact backbone SHA above|
|`josev2b.packed.head_gguf_sha256`|STRING|Exact head SHA above|
|`josev2b.packed.tokenizer_gguf_sha256`|STRING|Exact tokenizer SHA above|

All other distinct component KV records retain their exact encoded bytes,
including `josev2b_backbone_reference.*`, `josev2b_head_reference.*`,
`josev2b.tokenizer.*`, `josev2b.serializer.*`, `tokenizer.ggml.*` and publisher
source repository/revision. The original `general.architecture` and optional
`general.name` values are retained as
`josev2b.packed.COMPONENT.general_architecture` and
`josev2b.packed.COMPONENT.general_name`, for COMPONENT=`backbone`, `head` or
`tokenizer`. The source converter definitions establish the different
architecture/name values and identical Apache license. The only permitted
repeated retained keys are byte-identical `general.license` and optional
`general.alignment` (UINT32,32). Any other metadata collision, including an
identical undeclared key, rejects. There is one output architecture/name and one
copy of any permitted identical common key. Tensor names stay `l.*` and `head.*`;
no rename, alias, transpose, cast or inferred tensor is introduced.

The manual stdlib parser accepts little-endian GGUFv3, F32 tensors only, rank1–4,
positive bounded dimensions, at most442 tensors, files≤8GiB and headers≤64MiB.
It rejects duplicate keys/names, unsupported/nested arrays, invalid UTF8,
noncanonical booleans, nonfinite scalar metadata, wrong types/alignment,
overlap, unexplained gaps, nonzero padding, truncation and extra trailing data.
Primitive KV arrays retain their types. Tensor extents and all schema dimensions
must match the declared reviewed component exactly. Existing finite-F32 component
proofs remain bound by their exact whole-file and per-tensor hashes; no new
billions-element Python floating point pass is implied.

The largest embedding is2,034,237,440 bytes. Copy and hashing use at most4MiB
payload blocks; tensors are never loaded into Python arrays or converted through
`tobytes`. Output headers contain offsets for the unchanged bytes. After writing,
the converter independently parses the output and rehashes every tensor range,
checks every KV record, and rehashes input whole files again. Exclusive publication
occurs only after readback; a partial file remains on failure, with no completed
receipt. The receipt records source/Git objects, component hashes, output hash,
all tensor payload hashes/dimensions/offsets and KV record hashes. Packaging alone
never sets full-model or release qualification true.

After source review and explicit corrected-composed pinning, the hosted caller can
provide already authenticated component files and run:

```sh
python tools/pack_josev2b_f32.py \
  --backbone backbone-f32.gguf --head joint-head-f32.gguf \
  --tokenizer josev2b-tokenizer.gguf --composed-receipt hosted.json \
  --output packed/josev2b-systemone-f32.gguf --report packed/repack.json
```

The public CLI additionally requires hosted Linux and exact committed source
closure. It downloads nothing. Small stdlib tests use only authored fake F32
values and tiny GGUF fixtures; they do not activate production packaging. All
multi-gigabyte reads/writes, eventual packed-loader builds, inference, artifact
preservation and qualification belong on hosted CI. No packed asset, public API,
registry alias, model publication, default change, tag or app update is claimed.

The prepared reusable workflow
[`josev2b-packed.yml`](../.github/workflows/josev2b-packed.yml) and
[`josev2b_packed.py`](../tools/ci-heavy/josev2b_packed.py) add an explicit hosted
packing/qualification caller. Its separately pinned
[`josev2b-packed-validation-v1.json`](josev2b-packed-validation-v1.json)
uses runtime `0f0b1e5955a8791855fdba04d018b536b6a879f8` in a detached hosted
checkout; the converter checkout and all historical component guards remain
unchanged. No root public API is edited by this lane. The runtime's own source
contract, metadata/legacy-rejection controls, actual Ref0/1/2 controls, compiler
flags and ordinary/observer archive exclusions must pass before bulk model
archives are downloaded. Tokenizer extraction uses its exact reviewed nested
artifact member before multi-gigabyte source retrieval.

The worker fetches only the exact retained component archives and reviewed
composed receipt, verifies full original251-array inventory and notices, runs the
streaming packer, then executes actual packed plain/observed/repeat four-request
probes. Exact `ed23f1ae` low-level comparison helpers are authenticated before
import and reused without invoking or changing historical source guards. The
same frozen51a2e16b originals remain the targets:227 mapped Ref+independent
comparisons,24 explicitly unmapped traces, exact IDs/masks/spans, native lexical
means, all raw final/observer/repeat checks and four complete typed responses,
with every existing budget unchanged. Numerical failures retain all completed
arms and reports before a nonzero result. A packed asset/notices artifact and a
separate raw-trace artifact remain remote for90 days; small review reports include
packing hashes, compiler evidence, command logs and every comparison. This worker
has not been dispatched or qualified yet. Its model runs/builds require hosted
CPU resources with at least10GiB available RAM and25GiB free scratch; no original
framework/checkpoint recapture or local payload execution is provided.
