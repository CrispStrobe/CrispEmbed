# Frozen W7 hosted distribution

The owner merged the independently validated Linux package through PR237 at
`73f8a91ea19473d5ab784e3c3c1520efac6e1edc`; exact reviewed head `fa964043`
passed 12 checks with ten expected skips and seven green workflow records.
The 1,031 retained assertions and 25 package/21 consumer controls are separate
from new hosted transport evidence.

The declared durable destination is the model repository
`cstr/crispembed-spacy-de-core-news-sm-3.8.0`. The hosted
[`spacy_w7_publish.py`](../ci-heavy/spacy_w7_publish.py) authenticates the
official source wheel, reproduces the five original F32 bundle files with the
locked Python 3.12.14 environment, and uses the unchanged accepted native archive.
It refuses to publish unless the complete package is the original
7,808,129-byte SHA-256
`4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01`.
All native/model/source notices remain included. No new training data is shipped.

An upload is not a successful publication gate. The script records its actual
immutable Hub commit, checks the complete tree, then freshly downloads every
file over HTTP into an empty cache and checks each full digest/length. It also
authenticates the downloaded manifest and performs a new atomic installation
with complete immutable-tree verification. The final receipt records exact
resolve URLs, sizes, digests and installer outcome. Until that receipt passes,
publication is pending. Generated binaries and complete capsules stay outside Git.

This transport gate loads no native model and does not repeat consumed native
acceptance/W6 grammar panels. The accepted Linux profile still names its six
exact external dependency builds; no broader Linux portability or other-platform
qualification follows. macOS signing, sandbox/app loading, new grammar evidence
and quantization are separate owner gates. The app provider remains default OFF.

## Platform capsules and model card (schema v2)

Additional platforms use [`platform_distribution.py`](platform_distribution.py):
the same closed canonical USTAR/gzip format and fixed five-file model bundle,
with member pins in the embedded `distribution.json` authenticated by an
externally trusted archive SHA-256. The library sits under
`native/<library sha256>/`; every notice of the accepted Linux capsule is
required. The frozen Linux v1 capsule and `distribution.py` are unchanged.

[`spacy_w7_platform_package.py`](../ci-heavy/spacy_w7_platform_package.py) runs
on a real runner for `linux-x86_64`, `linux-arm64` or `macos-arm64` (no
emulation). It builds from the checked-out source with static PCRE2 (and static
libstdc++/libgcc on Linux), runs ctest, packages, verifies, installs atomically,
then runs the development parity panels and text-API controls against the
library **installed from the capsule**. Without `--publish` it uploads nothing.

```sh
gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/spacy_w7_platform_package.py \
  -f runner=macos-15 -f isolated_lane=spacy-pkg-macos-arm64 -f args="--platform macos-arm64"
```

`--publish` is accepted only on `main`. It makes two Hub commits: the capsule
and its receipt under `packages/<platform>/<source commit>/`, then the merged
`hub-catalog.json` and the model card rendered by
[`model_card.py`](model_card.py) from [`hub_catalog.json`](hub_catalog.json).
Each commit is checked for the exact expected tree and every uploaded file is
freshly downloaded from the immutable revision and hash/length verified; the
capsule is then verified and installed again from those downloaded bytes.
`--card-only --publish` refreshes only the card and catalog.

A new capsule's analysis claim is the hosted development parity of that exact
library; its grammar claim is always "not qualified". Consumers (the Quds
catalog) add a package only by its archive SHA-256, manifest digest and library
digest from the receipt. No hosted Windows lane exists yet.
