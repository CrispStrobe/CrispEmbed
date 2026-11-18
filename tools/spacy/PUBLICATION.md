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
