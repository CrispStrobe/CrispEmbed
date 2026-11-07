# LightOn4B production fence

Current LightOnOCR-3-4B Q4 artifacts must not ship as a default, public model
alias, application model or qualified release. The target is measured output
quality against the actual llama.cpp Q4_K_M baseline, with fixed image/prompt,
source and model hashes. A Q4_K_M label, imatrix ingestion, mixed bit counts,
component cosine or faster execution is not that evidence.

The current boundaries are separate:

- [CLI registry](../examples/cli/model_mgr.cpp) contains no LightOn4B alias,
  filename or candidate payload hash. Existing remote experimental GGUFs remain
  research artifacts and preserved evidence, not application defaults.
- [Production Qwen loader](../src/qwen35_ocr.cpp) admits formats1–4; experimental
  LightOn4B format5 is rejected. Enabling prototype gates does not open this
  public loader. The generic C ABI routes `qwen35_ocr` through that loader.
- [Release packaging](../.github/workflows/release.yml) copies explicit CLI,
  server and quantizer binaries, not experimental probe executables or models.
- [Native prototype](../src/lighton4b_native_context.h) remains a separate gated
  admission path. Q6 tied embedding/head support is a storage/execution capability,
  not shipping or decoded-quality approval. Experimental precision/recipe gates
  remain default-off and must not be enabled inside production source.

`test-lighton4b-release-fence` exercises public format5 metadata rejection with
prototype flags off and on, without payload acquisition. The Kaggle-only
[controller](../tools/kaggle/lighton4b_release_q6_head_controls.py) also checks the
registry, public format admission and explicit release binary lists.
`test-lighton4b-q6-tied-head` uses independently decoded Q6_K bytes for gathered
boundary rows and aliased tiled vocabulary projection, including BF16 weight
mode, explicit-off/quant gates and immutable storage. Those tests cannot qualify
an actual model or establish llama.cpp output parity.

Production admission or a model alias needs a separate reviewed change after
paired actual generation/EOS and output quality pass the frozen acceptance
cases, with original negative results and limitations retained. No F32 original
model is a prerequisite for the BF16 source comparison. Current capped or
incorrect Q4 output remains a rejection; it is not replaced by a favorable
component or prefix result.

The public OCR dispatcher rejects raw `qwen35` and experimental LightOn4B
format 5/6 metadata before backend selection and payload loading, even when
prototype gates are enabled. Supported custom formats 1–4 and unrelated legacy
architectures retain their existing dispatch. The physical format-6 prototype
uses its separate experimental context; this fence does not admit it publicly.
The model-free controls exercise the shared metadata predicate and direct public
Qwen35 rejection, with a source check for its early dispatcher placement.
