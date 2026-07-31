# EasyOCR model license manifest

This manifest is required before publishing converted GGUF artifacts.

| Artifact family | Upstream source | License | Conversion status |
|---|---|---|---|
| Gen-1 recognizers | JaidedAI/EasyOCR | Apache-2.0 | pending conversion + provenance check |
| Gen-2 recognizers | JaidedAI/EasyOCR | Apache-2.0 | pending conversion + provenance check |
| CRAFT detector | clovaai/CRAFT-pytorch | BSD-2-Clause | pending conversion |
| DBNet-18/50 detectors | EasyOCR release assets; DB implementation lineage | verify checkpoint terms separately | pending provenance confirmation |

The GGUF metadata must retain `general.source`, `general.license`, and the
upstream copyright notice. A model release is not complete until the DBNet
checkpoint terms have been confirmed from the authoritative asset/source.
