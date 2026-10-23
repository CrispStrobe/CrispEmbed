# OvisOCR2 bounded quality and quantization lane

The self-authored [panel](../../tests/results/qwen35-ocr/ovisocr2-quality-panel.json)
was frozen before candidate OCR outputs. Its four development cases may inform
one recipe choice; its four heldout cases must remain unseen by calibration or
recipe tuning. All images and transcripts are CC0-1.0. The panel is short
synthetic text and cannot establish broad document quality.

The hosted driver should first call `render_panel()` using Pillow 12.1.0,
preserve `rendered-panel.json`, every PNG and their raw RGB hashes, then run
the pinned original and each candidate on the exact same pixels and prompt.
Keep raw generated IDs/text, processed text, finish and stop reasons, token
cap, source commit, model checksum, runner log and package versions. `score_panel()`
reports exact text, CER/WER, completion, cap and format outcomes separately for
development and heldout. `paired_quality_screen()` rejects any heldout OCR
error increase, incomplete generation or format failure. A passing screen is
only a small diagnostic; decoded task quality and release suitability still
need independent review on realistic licensed documents.

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
Q4_K. Do not run or publish it until Q8 has a reviewed outcome. For each
candidate, read actual GGUF tensor names, shapes, types and raw byte hashes,
then call `validate_quant_tensors()`. Retain the quantizer log, input/output
GGUF SHA-256, every per-tensor record, original/native decoded outputs and
failure reports remotely. No model alias or release is qualified by this plan.
