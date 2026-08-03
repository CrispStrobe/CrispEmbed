# Image provenance: marking and Content Credentials

Every image CrispEmbed emits says what produced it. This is the EU AI Act
Art. 50(2) machine-readable marking; [`POLICY.md` §5](../POLICY.md) explains why
we mark by default even though we argue the duty does not bite for the document
case.

Two levels, and it matters which one you have:

| Level | Always? | What it proves |
|---|---|---|
| PNG `iTXt` chunk | yes | **Nothing.** It is a *claim*, strippable with any editor. It satisfies "machine-readable marking". |
| C2PA manifest | only with a signing identity | The bytes have not changed since signing. **Who** signed is a separate question — see below. |

## Default behaviour

Output is **PNG**, not raw Netpbm. That is a deliberate format change: Netpbm has
no metadata container, and C2PA has no PPM binding, so provenance was
impossible in the old format. PNG is also usually smaller — a 96×96 restoration
went from 27,661 bytes of PPM to 19,505 bytes of PNG *including* the metadata.

```bash
crispembed --esrgan-model esrgan-x4.gguf --esrgan-sr in.png > out.png
python -c "from PIL import Image; print(Image.open('out.png').info['CrispEmbed'])"
```

```
generated=true
software=CrispEmbed
engine=esrgan-sr
digitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced
note=AI-processed image. Not an authentic record of the original; ...
policy=https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md
```

`exiftool out.png` and `magick identify -verbose out.png` read the same chunk.

### Why `algorithmicallyEnhanced`

The IPTC term is deliberate. `trainedAlgorithmicMedia` means wholly synthetic
media; our input is a real capture that an engine enhanced. Asserting the
stronger term would be a **false provenance claim**, which is a worse position
than making none — a reviewer who trusts it would discard a genuine photograph
as AI-generated. The unit test asserts the wrong term is absent.

### Getting the old format back

```bash
CRISPEMBED_IMAGE_FORMAT=ppm crispembed --adair-model m.gguf --adair in.png > out.ppm
```

Raw Netpbm, with only the header-comment marking. This is how the benchmark
harnesses (`tools/eval_restoration_quality.py`, `tools/scan_cleanup_bench.py`,
`tests/ocr_preprocessor_benchmark.py`) are pinned — they parse pixel bytes
directly and gain nothing from a PNG decoder.

## HTTP server

The super-resolution and restoration endpoints (`/esrgan/sr`, `/swinir/sr`,
`/hat/sr`, `/dat/sr`, `/pan/sr`, `/safmn/sr`, `/tbsrn/sr`, `/text/sr`,
`/restormer`, `/scunet/denoise`, `/instructir/restore`, `/adair/restore`)
return the image base64-encoded in their JSON response. That payload is a
**marked PNG**, and the response says so:

```json
{ "image": "iVBORw0KGgo...", "format": "png", "width": 384, "height": 384, ... }
```

Until recently these base64'd raw RGB bytes, which carried no provenance at
all — the engines that synthesise detail were the least marked surface in the
project. `CRISPEMBED_IMAGE_FORMAT=ppm` restores raw bytes and reports
`"format": "raw"`, for clients consuming RGB directly.

## Content Credentials (C2PA)

Off unless you configure a signing identity, because **CrispEmbed ships no
signing key**.

That is not an oversight. A private key published in an MIT repository lets
anyone mint a manifest naming CrispEmbed as the software agent for an image it
never touched, and re-sign after altering the pixels. Both jobs a C2PA
signature exists to do would be destroyed, while the output still *looks*
signed — worse than no manifest at all.

### Build with C2PA

```bash
cmake -S . -B build -DCRISPEMBED_C2PA_FETCH=ON
cmake --build build -j
```

Fetches the prebuilt c2pa-rs native library. Absence is a supported state: no
library, or no certificate, still yields a marked PNG. The configure output says
which you got.

### Generate a per-installation identity

```bash
./scripts/make-c2pa-cert.sh
export CRISPEMBED_C2PA_CERT="$HOME/.config/crispembed/c2pa/cert.pem"
export CRISPEMBED_C2PA_KEY="$HOME/.config/crispembed/c2pa/key.pem"
```

Two constraints the script encodes, both learned from c2pa-rs refusing:

* **A self-signed certificate is rejected outright** (`the certificate was
  self-signed`). The script builds a leaf plus a local CA.
* **The key must be PKCS#8** (`BEGIN PRIVATE KEY`), not SEC1 (`BEGIN EC PRIVATE
  KEY`), or you get an opaque `PKCS#8 ASN.1 error`.

### What a locally-rooted manifest is worth

Verifiers will show **unverified signer**. The chain is not on the C2PA trust
list, so the manifest attests *what was done*, not *who did it*. That is the
same trust level a bundled certificate would give — minus the shared secret that
would let anyone impersonate the project.

For attributable provenance you need a certificate from a CA on the C2PA trust
list, and then `CRISPEMBED_C2PA_CERT`/`_KEY` point at that instead.

### Verifying

```bash
c2patool out.png            # or https://contentcredentials.org/verify
```

The manifest asserts `c2pa.edited` with the same `digitalSourceType` as the
`iTXt` chunk — the two levels are generated from one source so they cannot drift.

## Environment variables

| Variable | Effect |
|---|---|
| `CRISPEMBED_IMAGE_FORMAT=ppm` | Emit raw Netpbm instead of PNG |
| `CRISPEMBED_C2PA_CERT` | PEM certificate chain (leaf first, then CA) |
| `CRISPEMBED_C2PA_KEY` | PKCS#8 private key |
| `CRISPEMBED_MARK_GENERATED=1` | Header-comment marking in Netpbm mode |

## Limits, stated plainly

* An `iTXt` chunk is metadata, not evidence. Anyone can strip or forge it.
* A C2PA manifest signed by a locally-rooted chain proves integrity since
  signing, not origin.
* Neither makes a restored image an authentic record. Upscaling a licence plate
  or a face invents a plausible completion; it does not recover information that
  was never captured. See [`POLICY.md` §5](../POLICY.md).

## Tests

`tests/test_image_provenance.cpp` — 35 checks: default format, chunk structure
and CRCs (validated with an independent bit-by-bit CRC-32, since a wrong CRC is
accepted by lenient decoders and rejected by strict ones), `iTXt` field layout,
buffer/file paths byte-identical, MIME correctness, input rejection, and real
signing when a certificate is supplied. Runs on every push; the signing half
reports itself skipped rather than passing vacuously when no certificate is
configured.
