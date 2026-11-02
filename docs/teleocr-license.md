# TeleOCR: source license and provenance audit

Audit date: 31 October 2026. The integration uses the published TeleOCR
checkpoint, licensed model-code facts, CrispEmbed's existing native engine,
and the Apache-2.0 Rust reference. It does not include or directly translate
the official GitHub document client's unlicensed Python files.

## Weight license lineage

**The TeleOCR weights are governed by the Qwen RESEARCH LICENSE AGREEMENT
(non-commercial), not only by the model card's `apache-2.0`.** The paper
(arXiv:2608.12898, Section 4) says the vision encoder is "inherited from
Qwen2.5-VL" without naming the size; per-tensor weight cosines settle it
([evidence](../tests/results/licenses/qwen-vision-lineage.json),
`tools/weight_lineage.py`): TeleOCR's block-31 attention projection is 0.9926
against Qwen2.5-VL-3B-Instruct and 0.52-0.56 against every Apache-2.0 tower
(Qwen2.5-VL-7B, 32B, Omni-7B). Qwen2.5-VL-3B (and Omni-3B) are released under
the Qwen RESEARCH LICENSE AGREEMENT: use, reproduction, distribution and
derivative works for non-commercial purposes only (section 2), redistribution
with the agreement, a NOTICE line and "Built with Qwen" (sections 3-4), and
added terms only where they otherwise comply (section 3d). The decoder derives
from Qwen3-0.6B (Apache-2.0). No Apache-2.0 Qwen release carries the 3B tower,
and TeleOCR fine-tuned it, so it cannot be swapped without retraining.

Consequences in CrispEmbed: loading needs `CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1`
(engine gate, like LFM2-VL's), the `teleocr-*` registry entries are
`qwen-research` (download needs `--accept-license qwen-research`), and the
published `cstr/TeleOCR-GGUF` ships the agreement as `LICENSE`, the Apache text as
`LICENSE-APACHE-2.0` and the required `NOTICE`. CrispEmbed's own source code is
unaffected. MinerU2.5-Pro, by contrast, descends from Apache-2.0 Qwen2-VL-2B and
Qwen2-0.5B weights and is consistently Apache-2.0.

## Findings

The absence of a root `LICENSE` in the official GitHub repository is real, but
the source-code situation is more specific than "all official code has no
license." The checkpoint's `modeling_naviocr.py` has an explicit Apache-2.0
header, independently of the model card's license field. A vLLM implementation
inside the GitHub repository also carries an Apache-2.0 header. The separate
document client and table postprocessor inspected here have no license header.

| Source | Exact version | Observed declaration | Use in this integration |
| --- | --- | --- | --- |
| [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed/tree/15adb672f71f8470a23db1284dafa102515adae9) | `15adb672f71f8470a23db1284dafa102515adae9` | MIT, with existing third-party notices | Existing ggml engine, public interfaces and newly authored converter |
| [TeleOCR checkpoint and model card](https://huggingface.co/XingChen-AGI/TeleOCR/tree/b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d) | `b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d` | Model card declares `apache-2.0` | Weight tensors, tokenizer, configuration and evaluation images |
| [Checkpoint inference source](https://huggingface.co/XingChen-AGI/TeleOCR/blob/b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d/modeling_naviocr.py) | Same checkpoint revision | Explicit Apache-2.0 source header; Qwen/Hugging Face attribution | Architecture contract and original reference inference for parity |
| [teleocr-rs](https://github.com/rzafiamy/teleocr-rs/tree/957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe) | `957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe` | Apache-2.0 `LICENSE`, explicit code license in README | Independent Rust implementation reference and adapted document-parser algorithms |
| [Official Python repository](https://github.com/caipeng328/TeleOCR/tree/1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d) | `1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d` | GitHub API reports `license: null`; recursive tree has no `LICENSE`, `COPYING` or `NOTICE` | Read-only license/provenance inspection; no direct client/postprocessor translation |
| [Official vLLM model implementation](https://github.com/caipeng328/TeleOCR/blob/1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d/TeleOCR-vllm/TeleOCR_vllm/qwen2_5_vl.py) | Same official GitHub revision | Explicit `SPDX-License-Identifier: Apache-2.0`, vLLM/Qwen/Hugging Face attribution | License evidence; no vLLM dependency or copied vLLM source |

The inspected official GitHub files `infer.py`,
`TeleOCR/vlm_utils/TeleOCR_client.py`,
`TeleOCR/vlm_utils/TeleOCR_model.py`, and
`TeleOCR/vlm_utils/post_process/otsl2html.py` contain no license declaration.
Neither the model card nor the third-party Rust port is treated as a grant for
those Python files. The Rust port's own license is relevant to its own
implementation, not a mechanism for relicensing another repository.

## File scopes and redistribution

New converter and converter-test code is MIT. Existing CrispEmbed code retains
its MIT terms. The TeleOCR-specific adaptations identified in native source
comments and the auxiliary `tools/teleocr_parse.py` adapter, with its
`tests/test_teleocr_parse.py` regression tests, retain Apache-2.0 attribution.
The adapter derives from these explicitly licensed Rust files:

- [`crates/teleocr/src/pipeline.rs`](https://github.com/rzafiamy/teleocr-rs/blob/957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe/crates/teleocr/src/pipeline.rs)
- [`crates/teleocr/src/otsl.rs`](https://github.com/rzafiamy/teleocr-rs/blob/957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe/crates/teleocr/src/otsl.rs)

The native path is implemented using ggml and the existing CrispEmbed engine.
It does not link Candle or Rust. The converter only parses JSON and the
non-executable safetensors format. It never imports `modeling_naviocr.py` and
never calls `trust_remote_code`. The separate test/reference capture uses the
pinned, inspected Apache-licensed original module to obtain actual reference
outputs; reference execution is not a runtime dependency.
The isolated operator diagnostics in `tools/teleocr_diagnostics/` and
`tests/test_teleocr_ffn.cpp` also retain Apache-2.0 model-source attribution.
They explain numerical differences using the original licensed module and
source equations; their reports do not change the frozen acceptance criteria.

Redistribute [NOTICE.teleocr](../NOTICE.teleocr) and the full
[Apache-2.0 license](../models/licenses/apache-2.0/LICENSE) with the affected
source or binaries. Source comments identify modifications. Apache-2.0
[section 4](https://www.apache.org/licenses/LICENSE-2.0) requires retaining the
license and applicable notices and identifying changed files; its permission
to apply different terms to modifications does not erase the underlying
Apache obligations. CrispEmbed can keep its MIT project license while carrying
these third-party terms. Calling every incorporated portion "MIT-only" would
be inaccurate.

Converted checkpoints retain the publisher's Apache-2.0 declaration. The
converter writes `general.license=apache-2.0`, the source revision and a tensor
census. A changed storage format is not a change of license. A generated GGUF
is also not a guarantee of inference parity; see the accompanying test results.

## Architecture and base-model provenance

The authors' [technical report, section 4](https://arxiv.org/html/2608.12898v2)
describes a Qwen2.5-VL vision encoder, Qwen3-0.6B language model and an aligner
trained from scratch. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) declares
Apache-2.0. The paper does not identify an exact Qwen2.5-VL vision checkpoint
revision. Therefore this audit records the publisher's direct Apache-2.0
declaration for TeleOCR and does not claim to have reconstructed the full
training-weight lineage. This distinction matters because the separate
[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) distribution
uses Qwen Research terms, while
[Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) declares
Apache-2.0. Identical architecture dimensions do not prove which weight
distribution initialized TeleOCR.

The model card's acknowledgement of MinerU is not used as permission to copy
MinerU code or weights. No MinerU package is required by this integration.

## Reproducibility anchors

| Item | SHA-256 |
| --- | --- |
| Pinned `modeling_naviocr.py` | `7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922` |
| Pinned `config.json` | `92c5e15f1768dcb54bceae170043fbfefd248508159700b2179824b88c33ae75` |
| Pinned `preprocessor_config.json` | `f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0` |
| Rust `LICENSE` / retained Apache license text | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

No authors were contacted and no additional license grant was assumed. The
scope above is the repository evidence supporting this implementation, not a
claim about rights in every training example or every upstream dependency.
