# Provisional held-out executor preparation

The [frozen selection](lighton4b-heldout-v1.json) remains unchanged: historical
receipt, two-column page and owned inventory table, one prepared image each,
no replacement/prompt/crop adaptation. Existing transcriptions are provisional;
expert-human adjudication and embedded-text rights review remain release gates.
Private diagnostic execution does not establish true ground truth or authorize
new distribution of uncertain embedded text.

The [execution amendment](lighton4b-heldout-executor-v1.json) freezes the corrected
same-weight candidate, tested vision/text gate tuple and reference executables.
Its model path remains blocked until a hosted model-free CLI/build-control
receipt is pinned. No held-out candidate output has been read or inference run
while preparing this source. This is source preparation, not a passed benchmark.

The new reference-only `--provisional-heldout-cap1536` driver flag additionally
requires `CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT=1`, four threads and greedy-full.
It admits prompt≤512/image≤256 with prompt+1536≤2048 before vision/prefill. Old256,
observed512 and table512 modes retain their bounds; production source/CMake/GGML
remain the tested PR155 bytes. A mandatory observed167 row/stage replay checks
that the changed test executable retains existing behavior before any held-out
forward. All new CLI admissions/rejections and supported CPU guards must pass
before checkpoint acquisition.

Before each case, authenticate all three prepared PNG/RGB/reference assets and
frozen scorers. Upstream metadata preflight loads text vocabulary only, disables
projector warmup, and calls tokenization/preprocessing APIs without encode or
decode. Native metadata preflight emits the actual prompt, modalities, grid and
positions. These must match the source preflight exactly; an exclusive pre-forward
JSON seal binds them and the template before any held-out completion request.
The source server independently repeats its template/token checks against that
seal. There is no implicit downsize or shortened generation budget.

Execution order is source observerOFF, source observerON, native composed primary,
native source-features secondary. Full source IDs, text, probabilities and stop
reason must remain identical across OFF/ON. The ON arm preserves every actual raw
logit row and argmax; OFF raw-vocabulary bytes are not claimed. Native runs retain
independent greedy IDs/EOS and every raw row. Complete score files are preserved
remotely in verified lossless ZIP_STORED archives before owned individual row
files are removed. Incomplete attempts retain available partial files and fail
completion acceptance; no prefix is called a pass.

Each case has6000seconds total and1200seconds per arm, tenGiB available RAM,
25GiB scratch and eightGiB output admission, four compute/two build threads and
one worker at a time. A6000second signal deadline and process-group cleanup bound
HTTP, builds and native/source execution. At most three frozen cases are eligible;
a capped or failed case is not replaced. No original BF16 reference model runs.

Scoring uses the frozen `crispembed-ocr-quality-v1` parser and raw/parsed CER/WER.
Unsupported markup remains a parse failure; no result-selected normalization is
added. Inventory additionally needs the expected positional table shape and cell
errors no worse than source. A secondary feature-fed result cannot rescue a
composed failure. Reports always retain `qualified=false`: provisional execution
and an observed floor are distinct from general/release qualification.

Primary API sources: [pinned mtmd public interface](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/tools/mtmd/mtmd.h),
[pinned model interface](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/include/llama.h).
