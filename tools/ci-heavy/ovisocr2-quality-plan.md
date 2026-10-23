# OvisOCR2 bounded quality and quantization lane

The self-authored [panel](../../tests/results/qwen35-ocr/ovisocr2-quality-panel.json)
was frozen before candidate OCR outputs. Its four development cases may inform
one recipe choice; its four heldout cases must remain unseen by calibration or
recipe tuning. All images and transcripts are CC0-1.0. The panel is short
synthetic text and cannot establish broad document quality.

The manual [hosted workflow](../../.github/workflows/ovisocr2-quality.yml)
requires a retained positive native baseline review before packages or weights.
It first reproduces the three previously reviewed, separately licensed
[OCR documents](../../tests/regression/ocr-quality/panel.json) using their
original Pillow 11.3.0/CairoSVG 2.8.2 renderer and checks exact PNG/RGB hashes.
Only then does it install the pinned original HF CPU stack and call
`render_panel()` using Pillow 12.1.0,
preserve `rendered-panel.json`, every PNG and their raw RGB hashes, then run
the pinned original and each candidate on the exact same pixels and prompt.
Keep raw generated IDs/text, processed text, finish and stop reasons, token
cap, source commit, model checksum, runner log and package versions. `score_panel()`
reports exact text, CER/WER, completion, cap and format outcomes separately for
development and heldout. `paired_quality_screen()` rejects any heldout OCR
error increase, incomplete generation or format failure. A passing screen is
only a small diagnostic; decoded task quality and release suitability still
need independent review on the three realistic licensed documents. The
original model runs first on all frozen pixels. Native F16 and quantized
outputs come from the experimental owning-context test runner, which records
actual token IDs, confidence and stop enum. A separate public C ABI smoke
records text and confidence; that API does not expose IDs or stop reason.
The original run records actual selected-token probabilities from
`generate(output_scores=True)` and must be independently reviewed as
`tests/results/qwen35-ocr/ovisocr2-original-quality.json`. Later phases
require its explicit `original_run` number, verify the successful GitHub
workflow source, artifact ZIP digest, report SHA, all 11 frozen input
identities and caps, README/source proof, and recomputed scores. They reuse
the original outputs without rerunning original inference.

After the F16 native model has passed its original and owning-context checks,
test only `q8-protected` first. Its `crispembed-quantize` command is generated
by `quant_command()` from the reviewed 473-tensor source schema. It preserves
the F16 visual tower, token embedding, recurrent A/B gate matrices and all
non-matrix/norm tensors; other eligible matrices use Q8_0. Compare the F16
and Q8_0 arms on the frozen development and heldout panels, with native
decoded output plus early-stage numerical evidence. Do not infer quality from
the quantizer exit code or tensor type alone.

`q4-attention-q8` is a second, gated size experiment. It keeps the same
protected tensors, attention matrices at Q8_0 and other eligible matrices at
Q4_K. It requires an independent review of the actual Q8 diagnostic and an
explicit `allow_q4_diagnostic` decision before any run. A failed Q8 numerical
arm remains in the record and blocks Q8 publication, while Q4 may still be
measured. The native
unsupported Q4_0 fallback is a failed tensor audit. For each
candidate, read actual GGUF tensor names, shapes, types and raw byte hashes,
then call `validate_quant_tensors()`. Retain the quantizer log, input/output
GGUF SHA-256, every per-tensor record, original/native decoded outputs and
failure reports remotely. A separate `q4-calibrated` phase invokes the existing
native importance collector on only the four frozen development images and
requires all 72 FFN importance vectors to be applied; heldout images and
documents never enter calibration. Both Q4 arms require a reviewed positive
Q8 diagnostic review. A quantized candidate GGUF and upstream attribution stay in a
remote workflow artifact for later independent review and possible publication.
Each quantized phase also downloads the small original activation artifact from
the reviewed model-audit run, verifies its ZIP and reference GGUF/JSON/image
SHA-256 values, and runs `test-qwen35-decoder --native-frontend` with the
original float comparison thresholds and all-layer coverage. The report keeps
every metric, the earliest failing stage and magnitudes, IDs/stop/text evidence,
and the complete log. This is a quantization drift diagnostic. Its failure
does not stop the separate full OCR quality panel, but it blocks candidate
qualification.
Paired realistic document gates reject increased raw or parsed CER/WER,
incomplete or malformed output, and any error in the controlled 15-cell grid.
No model alias or release is qualified by this workflow alone.
