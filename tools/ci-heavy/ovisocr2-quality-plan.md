# OvisOCR2 bounded quality and quantization lane

The self-authored [panel](../../tests/results/qwen35-ocr/ovisocr2-quality-panel.json)
was frozen before candidate OCR outputs. Its four development cases may inform
one recipe choice; its four heldout cases must remain unseen by calibration or
recipe tuning. All images and transcripts are CC0-1.0. The panel is short
synthetic text and cannot establish broad document quality.

The manual [hosted workflow](../../.github/workflows/ovisocr2-quality.yml)
requires a retained positive native baseline review before packages or weights.
It downloads the immutable reviewed archive of three separately licensed
[OCR document PNGs and prepared metadata](../../tests/regression/ocr-quality/panel.json)
before installing model dependencies. It verifies the pinned archive revision,
ZIP SHA-256/size, exact four members, member hashes, source/split/license
contracts and saved PNG/RGB hashes. It reuses those exact derivatives; the
original Pillow 11.3.0/CairoSVG 2.8.2 document renderer is recorded provenance.
After installing the pinned original HF CPU stack, it calls `render_panel()`
only for the eight authored synthetic cases using Pillow 12.1.0, preserves
`rendered-panel.json`, every PNG and raw RGB hash, then runs
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

The first independently reviewed native F16 quality run
[40056641800](../../tests/results/qwen35-ocr/ovisocr2-f16-quality.json)
at638380f6 completes all eleven cases. All1,841 generated IDs and every raw
and processed text match the frozen original HF outputs; all cases reach
EOS248046 before the cap. The eight authored cases have WER0 and the inventory
table has15/15 cells. Both bounded paired screens pass. This exact reviewed
report and its F16 GGUF SHA are the immutable eleven-case baseline for each
quantized comparison. The native run used AMD EPYC7763, four logical CPUs and
two threads; original HF used a different CPU. Its recorded wall times and
cumulative child RSS are diagnostics, not paired speed evidence.

Quant phases first fetch and recompute the small reviewed F16 report on all
eleven preserved inputs before candidate weights or the large original
activation reference. The reusable workflow defaults `f16_host_control` to
`dev-hello`: after exact source-tree, ggml gitlink, native runner/quantizer,
vision/scorer/converter, CMake, environment AST, package and thread checks
against the first F16 source, the current host executes that one F16 case.
Its PNG/RGB hashes, nine IDs, raw/processed text, EOS, cap, confidence,
wall time and cumulative child RSS upper bound are retained. The observed
confidence difference is recorded without an invented cross-host waiver.
This one case does not prove eleven-case F16 identity across hosts. Quantized
candidate comparisons use the reviewed first F16 eleven-case scores, and
paired document scores do not establish a causal speed comparison. Direct
driver calls default to `full`; the workflow accepts `full` explicitly for
the original eleven-case same-run host control. Original and first F16 phases
always retain their full eleven-case executions.

After the F16 native model has passed its original and owning-context checks,
test only `q8-protected` first. That
[hosted attempt40061433357](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40061433357)
is complete and [independently rejected](../../tests/results/qwen35-ocr/ovisocr2-q8-quality.json).
The first captured decoder-layer error is0.00376427 with cosine0.999920726;
183 numerical metrics fail unchanged gates. All eleven candidate cases reach
EOS without a cap, but the two-column page changes `Lorem Ipsum` to
`Lorem Ispum`, grows697→700 generated IDs and worsens raw/content CER/WER.
Ten cases retain first-F16 IDs/text; the first F16 eleven-case baseline stays
qualified. This complete Q8 diagnostic permits the two Q4 experiments but
does not qualify Q8 for publication.
Its `crispembed-quantize` command is generated
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
GGUF SHA-256, all473 before/after name/shape/type/raw-hash records in the small
quality report, original/native decoded outputs and failure reports remotely.
A separate `q4-calibrated` phase invokes the existing
native importance collector on only the four frozen development images and
requires all 72 FFN importance vectors to be applied; heldout images and
documents never enter calibration. It retains the small exact stats TSV and
importance GGUF with SHA-256/byte declarations in the quality report artifact;
the large candidate GGUF stays in its separate remote artifact. Both Q4 arms
require a reviewed complete
Q8 diagnostic and explicit permission to measure Q4; positive Q8 quality is
not required. A quantized candidate GGUF and upstream attribution stay in a
remote workflow artifact for later independent review and possible publication.
Corrected [plain Q4 40066573198](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40066573198)
and [calibrated Q4 40066574754](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40066574754)
are complete at7924d850 and independently rejected for numerical/document
regression. The positive development calibration proof is separate from its
negative candidate. The completed selection chooses qualified first F16. Earlier calibrated Q4 and raw replay
attempts failed before model access on a pure historical-Git-object test in
shallow checkout, while the first plain Q4 attempt was cancelled before
controls. The [journal](../../tests/results/qwen35-ocr/initial-ovisocr2-quant-shallow-control-failure.json)
keeps exact sources and failures; correction8fdf28f1 pins the unchanged
comparison function body SHA without changing model arithmetic or thresholds.
Each quantized phase also downloads the small original activation artifact from
the reviewed model-audit run, verifies its ZIP and reference GGUF/JSON/image
SHA-256 values, and runs `test-qwen35-decoder --native-frontend` with the
original float comparison thresholds and all-layer coverage. The report keeps
every metric, the earliest failing stage and magnitudes, IDs/stop/text evidence,
and the complete log. This is a quantization drift diagnostic. Its failure
does not stop the separate full OCR quality panel, but it blocks candidate
qualification.
Corrected Q4 runs additionally record a same-run F16 first-linear control
(`QUANT=0`), candidate first-linear probe (`QUANT=1`) and full decoder probe
with unchanged cosine/absolute thresholds. Every observed `Ref.compare` and
TRACE vector is retained under the separate remote `native-diff` artifact;
the small report keeps actual stage/shape/byte/SHA manifests and logs. All
three probes and the independently reviewed raw artifact must pass before
quantized publication. Original Q8 source8b421ba5 lacked raw vectors;
[candidate-only replay40066576197](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40066576197)
is complete with243 retained vectors, supplementing its immutable negative
quality receipt using the
same retained candidate and frozen original arrays. It omits a same-run F16
control and does not reconvert, rerun original inference or repeat the panel.
Raw completeness cannot reverse its numerical or recognition failure.
Paired realistic document gates reject increased raw or parsed CER/WER,
incomplete or malformed output, and any error in the controlled 15-cell grid.
No model alias or release is qualified by this workflow alone.
