# Big-LaMa: exact archive conversion and provenance

> **Internal research only for Big-LaMa (owner decision 2026-11-01).** Big-LaMa's checkpoint
> has no dedicated weight-licence text (see "Licensing" below). On this branch line it is for
> internal research and testing: no LaMa weights, GGUF, registry entry or public upload. MI-GAN,
> Moebius, PixelHacker and the shared SDXL VAE carry explicit MIT/Apache-2.0 weight terms.


The experimental `lama` engine accepts the original Big-LaMa generator as F32
GGUF. Its **837 exported tensors contain 204,228,108 bytes** (about 204 MB),
plus a small GGUF header/alignment overhead. This is stored model size, not a
peak RAM or latency measurement. Native execution uses the shared CrispEmbed
CPU/GGUF engine, the newly implemented Fourier operations, and the existing
image/mask CLI. It is off by default and requires `CRISPEMBED_LAMA=1`.

The native input profile accepts dimensions from 9 to 1,024 pixels. It converts
RGB8 to `[0,1]` F32, concatenates masked RGB with the binary white-is-fill mask,
and pads the right/bottom edges symmetrically to multiples of eight. The
generator retains the original reflection padding, 18 FFC residual blocks,
orthonormal real FFTs, disabled local Fourier units, and sigmoid output.
Known RGB pixels are copied exactly after compositing.

## One authoritative download identity

The [pinned original README](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/README.md)
links to `smartywu/big-lama` on Hugging Face. The converter accepts this exact
archive, frozen to its immutable revision:

| Item | Identity |
| --- | --- |
| Original code | `advimman/lama`, commit `786f5936b27fb3dacd2b1ad799e4de968ea697e7` |
| Official-linked archive | [big-lama.zip](https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip) |
| Archive bytes | **381,428,720** |
| Archive SHA-256 | `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6` |
| Inner checkpoint | `big-lama/models/best.ckpt`, **410,046,389 bytes** |
| Original configuration | `big-lama/config.yaml`, SHA-256 `4fdeed49926e13b101c4dd9e193acec9e58677dfdb4ba49dd6a3a8927964e2a7` |

The original archive digest authenticates its complete compressed contents.
There is no independently published inner-checkpoint SHA-256 in this download
index. The converter computes and records that digest **after** extracting
the verified archive. It never substitutes a similarly named mirror's digest.
In the native metadata, `lama.checkpoint.sha256` therefore identifies the
**official ZIP**; `lama.checkpoint.container=official-zip` and the explicit
member path make that meaning unambiguous. The receipt records the observed
inner digest separately.

The machine-readable [source contract](lama-source-contract.json) includes
14 original source-file hashes, the archive index, the exact configuration
hash, and the complete inert tensor census. Only bounded HTTP ranges were
used to prepare that census; no full checkpoint download or forward pass was
needed to inspect it.

## Hosted conversion

Use a hosted worker for full downloads, conversion, builds and inference under
the repository's contribution policy. The converter does not download files
or import Torch, Lightning, OmegaConf, NumPy or the upstream source code.

```sh
python models/convert-lama-to-gguf.py \
  --checkpoint-archive checkpoints/big-lama.zip \
  --source-dir original/lama \
  --output artifacts/big-lama-f32.gguf
```

`--archive` is an alias for `--checkpoint-archive`. Check out the exact source
commit above first. The output parent must exist; existing GGUF, receipt or
licence paths are rejected. A successful conversion creates:

- `big-lama-f32.gguf`, with original `generator.model.*` names and F32 bytes.
- `big-lama-f32.gguf.json`, binding actual GGUF size/SHA-256, every tensor
  readback hash, the authenticated outer archive, the observed inner digest,
  source/configuration identity, explicit omissions and the command arguments.
- `big-lama-f32.gguf.licenses/`, preserving the original code licence and
  the separate weight-licence evidence.

Use the **actual `gguf_sha256` from that receipt** for
`--inpaint-model-sha256`. The archive SHA-256 is not the converted GGUF digest.
No converted model digest is supplied in this source package because a real
conversion has not yet run.

The full archive hash is checked before ZIP interpretation. Extraction streams
to a private temporary file and verifies the exact member/configuration
inventory. The Lightning pickle is reduced to allow-listed inert metadata;
its original Python classes and methods are never loaded or executed. The
converter validates all **2,185 state entries**. It exports the 837 floating
generator tensors, records and omits 152 scalar BatchNorm training counters,
and excludes 1,196 evaluation-only entries. The two callback score references
are also accounted for. Evaluation-state storage aliases are allowed only
where the frozen original census records them; generator storage is required
to be complete, contiguous and unaliased.

## Independent reference and current evidence

The manual `lama-inpaint.yml` workflow offers a model-free build phase and an
explicit original/native phase. The latter captures the unchanged original
`FFCResNetGenerator` before converting weights or executing native inference.
The original capture uses the same inert reader to copy exact generator state,
including counters, into the original module. It retains Fourier-stage and
generator outputs and requires an observed replay to equal an ordinary
original forward bit-for-bit.

The eight owned fixtures cover a subtitle bar, an edge-touching mask, a center
mask, an empty mask, a full mask, one pixel, a wide image and a thin strip.
Non-square and non-multiple-of-eight dimensions exercise the padding boundary.
All acceptance limits were frozen before any full-model execution: stage
maximum absolute error 0.002, relative L2 error 0.0003 and cosine at least
0.99999; final prediction maximum absolute error 0.0002; masked RGB maximum
error one byte, mean error at most 0.1 and PSNR at least 50 dB. Known pixels
must have zero errors. The workflow retains failed and partial evidence.

The executed local checks are the bounded stdlib archive/parser/converter
tests, Python syntax checks and independent metadata/source comparisons.
No complete official archive conversion, original model execution, native
model execution, representative quality evaluation, or runtime memory/latency
measurement has been performed in preparing this source change. The
`native_qualified` and `representative_quality_qualified` flags remain false.

## Weight licence qualification

The [repository licence](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/LICENSE)
is Apache-2.0. In [issue 230](https://github.com/advimman/lama/issues/230#issuecomment-1817604197),
an upstream collaborator answered a question explicitly about the weight
licence by pointing to that repository licence. The pinned archive contains
no dedicated weight-licence text. This is the available first-party evidence;
the implementation preserves its qualification instead of presenting a
separate explicit weight grant as established fact.

`general.license=Apache-2.0` is accompanied by the fixed
`lama.weights.license.status` field, which records that documentary gap. The
unchanged code licence and evidence notice are retained in
[`licenses/lama`](../licenses/lama/NOTICE.md). No model weights are bundled.
