# Native Marker-style pipeline: licenses and provenance

Audit date: 31 October 2026. This integration implements a native document
pipeline using PDFium, deterministic processing and CrispEmbed's existing
TeleOCR backend. It does not run Python in production and does not incorporate
the original Surya checkpoints. [Feature coverage](marker-feature-coverage.md)
distinguishes this implementation from upstream Marker.

## Source-code and model licenses are separate

Current Marker and Surya code is Apache-2.0. The older description of Marker as
GPL-licensed does not apply to the revisions audited here. Apache-licensed
code can be included in CrispEmbed with the applicable license and notices;
it is not relabeled as MIT-only. Files referring to [NOTICE.marker](../NOTICE.marker)
identify their Apache-2.0 scope. The complete retained license is
[models/licenses/apache-2.0/LICENSE](../models/licenses/apache-2.0/LICENSE).

| Source | Exact revision/version | Observed terms | Role |
| --- | --- | --- | --- |
| [Marker](https://github.com/datalab-to/marker/tree/71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4) | `71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4`, 2.1.0 | [Apache-2.0](https://github.com/datalab-to/marker/blob/71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4/LICENSE) | Document-processing source reference and deterministic original implementation used by tests |
| [Surya source](https://github.com/datalab-to/surya/tree/9d5aadfebb07939755c24e8a80d0e61b3be80895) | `9d5aadfebb07939755c24e8a80d0e61b3be80895`, 0.23.0 | Apache-2.0 | Upstream reference's imported schemas; no predictor is allowed to run |
| [pdftext](https://github.com/datalab-to/pdftext/tree/0755265856d479d089d8afa8acbafba1155a4443) | `0755265856d479d089d8afa8acbafba1155a4443`, 0.7.1 | Apache-2.0 | Test-only original PDF text extraction |
| [pypdfium2](https://pypdfium2.readthedocs.io/en/stable/readme.html#licensing) | Reference environment: 5.10.1 | Apache-2.0 OR BSD-3-Clause, plus PDFium dependency licenses | Test binding, fixture rendering; absent from native runtime |
| [TeleOCR model/source and Rust reference](teleocr-license.md) | Pinned in the existing TeleOCR audit | **Qwen RESEARCH LICENSE (non-commercial)** for the weights: the vision tower derives from Qwen2.5-VL-3B-Instruct despite TeleOCR's apache-2.0 declaration ([lineage](../tests/results/licenses/qwen-vision-lineage.json)); code Apache-2.0 | Native ggml inference and document-model vocabulary. Every TeleOCR mode needs `CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1`; the PDFium text mode loads no model |

Marker has no root `NOTICE` at the inspected revision. Its absence does not
remove the Apache license or source attribution requirements. Copies and
adaptations must retain applicable notices and identify changes. See Apache
[section 4](https://www.apache.org/licenses/LICENSE-2.0).

PDFium is dynamically loaded from an explicitly selected native library.
Its binary and all build-specific dependency notices must accompany a
redistributed PDFium binary. The pypdfium2 project explicitly notes that
licensing depends on the PDFium build. This audit is not a substitute for the
license inventory of a distributor's chosen binary.

## Why the original model weights are excluded

The three current model LICENSE files below are identical to Marker's
[MODEL_LICENSE](https://github.com/datalab-to/marker/blob/71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4/MODEL_LICENSE)
after whitespace normalization. Their declared modified AI Pubs OpenRAIL-M
terms are not permissive Apache/MIT terms.

| Model | Exact revision | License file |
| --- | --- | --- |
| Surya 2.5 | `00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3` | [LICENSE](https://huggingface.co/datalab-to/surya-ocr-2.5/raw/00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3/LICENSE) |
| Official Surya 2.5 GGUF | `eb78671ae9af9bdddfac2e25507c55337b6098e2` | [LICENSE](https://huggingface.co/datalab-to/surya-ocr-2.5-gguf/raw/eb78671ae9af9bdddfac2e25507c55337b6098e2/LICENSE) |
| Fast layout and reading order | `0aee81d5fd9275c0582e545bf3a56944b1e75679` | [LICENSE](https://huggingface.co/datalab-to/surya_layout2/raw/0aee81d5fd9275c0582e545bf3a56944b1e75679/LICENSE) |

The text restricts use by entities exceeding the stated revenue/funding
thresholds, and independently restricts competing products/services
(Attachment A, section 2). Sections 7 and 8 extend attribution and purported
share-alike conditions to model output. Section 5 covers evaluation and
reparameterization as well as ordinary inference. Section 1(a) explicitly
separates complementary source code from this model license. A GGUF conversion
or new C++ runtime would not turn these weights into MIT-licensed data.

The Surya/GGUF LICENSE SHA-256 is
`636dfb8eb19efc5835ed6efb878f8b694a93f028c715d2547d831e2a55440276`.
The differently formatted layout LICENSE hashes to
`b6be9042dce439368b606444256cd97624a97c67dfae33957b87c3a0b461ae03`.

The original fast-layout upload
[`5a4317259137c04ebcc7b6492a8bb5f7c2e411b4`](https://huggingface.co/datalab-to/surya_layout2/commit/5a4317259137c04ebcc7b6492a8bb5f7c2e411b4)
briefly contained Apache-2.0 declarations. Six minutes later the publisher
changed the license/card with an explicit license-correction commit. LFS
metadata shows the same layout tensor file at both revisions; the reading-order
model was added after the correction. This implementation does not rely on a
legal conclusion about that historical grant or distribute either revision.

## OCR-error checkpoint and optional dependencies

Pinned Surya settings select `s3://ocr_error_detection/2025_02_18` for a
DistilBERT OCR-error classifier. Its manifest and license endpoints returned
HTTP 403 during this audit, so its exact file-level terms and hash were not
verified. The older official
[Hugging Face OCR-error model](https://huggingface.co/datalab-to/ocr_error_detection/tree/c1cbda3757670fd520553eaa5197656d331de414)
declares CC-BY-NC-SA-4.0, but is not proven identical to that S3 checkpoint.
The native pipeline uses explicit text-quality heuristics instead.

Marker's `full` extra resolves EbookLib 0.18, whose
[release metadata](https://pypi.org/project/EbookLib/0.18/) declares AGPL.
Neither `marker-pdf[full]` nor EbookLib is used in the reference setup or
native runtime. PyMuPDF appears in upstream development dependencies, but is
not needed by the selected normal PDF path and is not added here.

Current Marker also uses restricted weights in fast/CPU layout mode.
`disable_ocr` by itself still selects that detector. The test-only reference
therefore combines `disable_ocr` with `force_layout_block=Text`, injects
predictors that throw on every call, supplies an existing font, disables
network access and uses only owned digital PDFs. These original deterministic
outputs are not neural-model outputs or evidence of Surya inference parity.

## Limits

This records source evidence and the selected implementation boundaries. It
does not reconstruct all training-data rights, audit every optional upstream
dependency, or make a claim about model-license enforceability. No authors
were contacted and no additional rights were assumed. Production code and
test receipts must continue to identify the actual TeleOCR backend rather
than report the original Surya model as having run.
