# LightOnOCR-3-4B source-fixture notice

This notice was authored for CrispEmbed on 4 November 2026. It is not an upstream
model file or a statement of native-model qualification.

## Original source material

The configuration files, chat template and two README files are unchanged bytes
from [LightOnOCR-3-4B at b71010f0](https://huggingface.co/lightonai/LightOnOCR-3-4B/tree/b71010f095dc3735de7ee6b9969bff8f8f50b39a).
`huggingface_model_api.json` is the captured public API response for that pinned
revision. `model.safetensors.header.json` contains only the original 89,456-byte
JSON header, excluding its eight-byte length prefix and retaining its original
padding. No tensor payload is included.

`LightOnOCR-3-4B-tokenizer.prefix.txt` and
`LightOnOCR-3-0.8B-tokenizer.prefix.txt` are the exact first 65,536 bytes of their
respective tokenizer files, captured with verified HTTP range responses. The
comparison checkpoint is [LightOnOCR-3-0.8B at 36d86369](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/tree/36d8636925327c671bd8e479c517a004f7566bd9).
These prefixes are incomplete JSON and cannot replace either full tokenizer.

## Derived collection records

`audit_manifest.json` and `tokenizer_prefix_audit.json` are authored audit reports
containing source URLs, transport records, hashes and derived metadata. They are
not files published by the model repository. The original `audit_manifest.json`
is unchanged by the separate tokenizer/license audit. The public tokenizer
report uses a repository-relative reference to that original manifest.

The preflight independently checks the captured bytes and re-derives the complete
tokenizer metadata fields preceding each `model` object. It does not parse the
incomplete vocabulary/merges payload. Full tokenizer and checkpoint hashes are
upstream declarations unless the full corresponding file is separately verified;
prefix/header checks do not establish full-file integrity or runtime parity.

## License provenance

`LightOnOCR-Apache-2.0-LICENSE.txt` is the unchanged Apache-2.0 license from the
[official LightOnOCR code repository at 36755d46](https://github.com/lightonai/LightOnOCR/blob/36755d461be079737860a5f03ae0c803501269e9/LICENSE).
It is retained here under a descriptive filename. It was not present as a
standalone license file in the pinned Hugging Face checkpoint.

The model checkpoint separately declares `apache-2.0` in its pinned README and
public model metadata. These two sources remain distinct: the copied file is
the official repository license; the checkpoint declaration is preserved in its
own source records. No weight conversion, native inference, model publication or
consumer qualification follows from this source-only fixture collection.
