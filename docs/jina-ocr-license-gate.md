# Jina OCR and CC-BY-NC native loading

The Unlimited OCR native loader requires
`CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE=1` before loading identifiable Jina OCR v1
weights or an artifact declaring CC-BY-NC terms. The check applies to direct
`crispembed_ocr_model_init()` calls and to paths returned by the model resolver,
including existing files and cached registry aliases. It runs before a model
context, tokenizer storage, backend or weight buffers are allocated.

## License basis

The [official Jina OCR v1 model card](https://huggingface.co/jinaai/jina-ocr-v1#license)
declares **CC BY-NC 4.0** and directs commercial users to Jina AI. The
[Creative Commons license summary](https://creativecommons.org/licenses/by-nc/4.0/)
describes the attribution and noncommercial conditions; the
[legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
contains the full terms. These sources were checked on 9 November 2026.

The acknowledgement records an explicit loading decision. It does not grant
commercial-use rights, replace the upstream license, or change the license of
converted/quantized weights. This change introduces no model downloads or
upstream implementation code. CrispEmbed's own implementation remains under
its existing MIT license.

The existing [Jina conversion lane](../tools/ci-heavy/jina_ocr_v1.py) already
supplies `--name jina-ocr-v1`, `--license cc-by-nc-4.0` and the official model
source to the [Unlimited converter](../models/convert-unlimited-ocr-to-gguf.py).
The [registry](../examples/cli/model_mgr.cpp) already labels both Jina aliases
as `cc-by-nc-4.0`. Their immutable GGUF URLs and hashes are unchanged.

## Native gate policy

The [loader](../src/unlimited_ocr.cpp) checks the following string fields:

| Evidence | GGUF metadata fields | Decision |
|---|---|---|
| Jina identity | `general.name`, `general.source`, `general.url`, `general.source.url`, `general.source.huggingface.repository` | A bounded `jina-ocr-v1` word sequence requires acknowledgement. This includes the official source, the `cstr/jina-ocr-v1-GGUF` rehost and names with quantization suffixes. |
| Noncommercial license | `general.license`, `general.license.name`, `general.license.link` | CC-BY-NC labels, Creative Commons BY-NC license URLs and the expanded Creative Commons Attribution NonCommercial name require acknowledgement. The same check covers BY-NC-SA and BY-NC-ND labels. |
| Malformed evidence | Any of the fields above exists with a non-string type | Reject, including when acknowledgement is enabled. |
| Permissive or legacy artifact | No recognized restricted evidence and no malformed evidence | Continue with ordinary model loading. |

Identity and license evidence are independent. A recognizable Jina name/source
continues to require acknowledgement when a different field says `MIT` or the
license field is absent. A renamed artifact still requires acknowledgement
when its license metadata declares CC-BY-NC. Conflicting or combined license
text containing CC-BY-NC is treated conservatively as requiring acknowledgement.

Matching folds ASCII case and treats punctuation, spaces and underscores as
word separators. It uses whole word sequences: `Jina OCR v1 (Q8_0)` matches,
while `jina-ocr-v10` does not identify the known v1 model. A source field that
claims Jina identity still requires acknowledgement when it points at a mirror
or an unfamiliar host; a URL string is not authenticated provenance.

The shared engine continues to load permissive Unlimited OCR and Unlimited OCR 2
artifacts without this acknowledgement. Existing older files with no identity
or license metadata retain their previous behavior. The check does not infer
a permissive license for such files. It is scoped to known Jina identity and
declared CC-BY-NC terms in this engine; it does not audit every model family or
every custom license in the registry.

Metadata is editable. Fully stripping or replacing every identifying and
licensing field cannot be detected by a metadata gate. Callers remain
responsible for authentic model provenance and applicable rights. Download
hash verification and this acknowledgement serve separate purposes.

## Acknowledging a permitted load

Set the acknowledgement in the process that loads the model:

```sh
CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE=1 your-application
```

The gate uses the shared `core_env::on()` contract. Unset, empty, `0`, `false`,
`off` and `no` are off; word comparisons ignore ASCII case. Values are not
trimmed. Use the documented value `1` to acknowledge.

`--accept-license cc-by-nc-4.0` and `CRISPEMBED_ACCEPT_LICENSE` continue to govern
registry downloads. They do not satisfy the native acknowledgement. Previously,
the resolver returned existing local files and cached entries before its
download-acceptance branch. The native check now also runs when those returned
paths are loaded. No prompt or new public API is introduced.

This acknowledgement does not enable experimental inference paths, alter
Unlimited OCR 2 feature/quality gates, or qualify any precision or model output.

## Model-free regression coverage

[The native regression driver](../tests/test_unlimited_ocr_license.py) writes
tiny GGUF metadata fixtures and calls the actual shared-library public APIs.
It checks direct loads, the existing-path resolver and cached Jina names,
filenames and Q8 aliases with downloads disabled and offline mode enabled.

The matrix covers every off spelling, acknowledgement controls, independent
identity/license fields, misleading MIT metadata, missing license metadata,
mirror sources, case/separator variants, malformed field types, ordinary
permissive Unlimited artifacts and legacy files without identity/license
fields. A generic download acknowledgement does not bypass the native gate.

Every fixture has zero tensor headers. Acknowledged/permissive controls must
reach the distinct `GGUF contains no model tensors` check after licensing;
restricted fixtures must stop at the earlier acknowledgement diagnostic. The
zero-tensor guard also rejects invalid artifacts before model/backend
allocation. No weights are read. A crash, assertion, timeout, unexpected success
or rejection at the wrong stage fails the test. The JSON report records the
library and fixture hashes, individual diagnostics and overall status.

Hosted native validation uses the same pinned `gguf`/NumPy dependencies as the
TeleOCR loader controls:

```sh
python -m pip install numpy==2.3.5 gguf==0.19.0
python tests/test_unlimited_ocr_license.py \
  --lib build/libcrispembed.so \
  --report build/unlimited-ocr-license.json
```

These controls establish metadata and acknowledgement behavior. They make no
OCR accuracy, performance, source-parity or model-publication claim.

On 9 November 2026, the owner-authorized Linux CPU execution passed all 106
controls against final native source `891ea3d3632c6a3c3155e25b34705da6a02ec807`.
The [retained report](../tests/results/marker/pdf-craft-local-2026-11-09/real-pdf-lifecycle/final-native-python/test_unlimited_ocr_license.json)
includes every fixture hash and observed diagnostic. No model weights were used.
