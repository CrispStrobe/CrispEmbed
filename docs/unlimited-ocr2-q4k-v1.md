# UnlimitedOCR2 bounded mixed-Q4_K experiment v1

This is prepared source, awaiting review and a separately authorized hosted CPU
run. It qualifies no model, alias, default, release or app. Historical failed
precision sweeps and their original references remain unchanged.

The runtime/build/converter/quantizer closure is fixed to
[`83eac856cc684808e417ebfed7db7c2f677dc09f`](https://github.com/CrispStrobe/CrispEmbed/tree/83eac856cc684808e417ebfed7db7c2f677dc09f).
The original inference blueprint is
[Unlimited-OCR-2 revision04647c7a](https://huggingface.co/baidu/Unlimited-OCR-2/tree/04647c7a6098d8d58915f003536a741c82eafb40),
with the existing [inference spec](unlimited-ocr2-inference-spec.md). Model-card
license metadata says MIT; this lane makes no broader license assertion.

The frozen [four-page trial](../tests/results/unlimited-ocr2/trial.json) supplies
original IDs and full decoded text for paragraphs, table, receipt and German.
It contains no intermediate arrays. The older three-page original array capture
is a different panel and supplies no four-page intermediate gold here. The new
intermediate comparisons are explicitly paired, same-host F16-to-Q4 measurements.

The strict first control is the immutable
[published F16 GGUF](https://huggingface.co/cstr/Unlimited-OCR-2-GGUF/blob/e3af0c0c4bfa241c161db64ca36f3c63d3ca8378/unlimited-ocr-2-f16.gguf),
6,692,677,696 bytes, SHA256
`487b887eddaf73e070cd98c3e10d112d21749036ca06b5e7fc103c3fb4def517`.
Its vocabulary metadata omits831 added pieces. The exact authenticated
`src/unlimited_ocr2_text.h::added_token` fallback supplies IDs128000..128830;
this experiment does not describe the published metadata as complete. The
corrected converter remains pinned but is not executed: it would produce a
separate source model. No original weights or inference outputs are recaptured.

Every unselected tensor retains its actual published floating precision,
physical axes and payload bytes. This includes F16 vision/projector, vocabulary,
head, router and normalization tensors where present in that source; no F32
calibration or F32 critical-tensor claim is made. Any four-axis F32 input is
rejected before conversion because the generic quantizer would flatten it.
Actual inventory determines precision, rather than a filename or recipe label.

Three predeclared recipes quantize only stacked expert gate/up matrices in the
last layer11, last four layers8..11, or all eleven MoE layers1..11. Their actual
Q4_K tensor counts must be2,8,22, respectively. Each has ggml shape
`[1280,896,64]`; width1280 is divisible by256. Expert down matrices have width896
and remain their original floating type. Q4_0 fallback and any unselected tensor
change are hard rejections. Exact-name overrides preserve all other source
records, including critical fields and shape/payload identity. The quantizer
internally promotes F16 values for encoding; it does not recover original F32
weights.

All four F16 cases finish first with exact original IDs/full text, no token cap,
and unchanged body/box/task checks. The ordinary CLI, overlay CLI with observation
disabled, enabled observation, and repeated observation must agree on IDs/text.
Observed and repeat stages must be bit-exact. Only then may any candidate forward
execute. A failed first control stops the experiment and retains all evidence.
The native trace excludes EOS; completion is explicitly source-derived from the
EOS/page-guard loop and the strict1024-token cap, not a captured EOS-ID claim.

A new test-only generator authenticates the exact `src/unlimited_ocr.cpp` SHA,
then writes an observational overlay into a separate hosted worktree. Production
source files are unchanged. Both builds use the real root CMake, portable CPU
ISA and LLAMAFILE-ON backend; their actual caches and compile flags must agree.
The plain binary must exclude observer symbols. Retained outputs are connected
existing graph tensors marked as outputs, read synchronously before graph and
KV owners are destroyed. No reference input, hidden state or logits are injected.

Each observed forward records all12 full first-prefill layer outputs, the final
norm and first unguarded logits, and first16 rows of each MoE layer's normalized
input, selected expert IDs, gate pre-activation and up projection:58 typed stage
records per page. These four prompts must fit the declared2048-row chunk; changed
coverage rejects. Assembled vision outputs are also compared byte-for-byte.
Only first-prefill states and the selected projection rows are observed; this is
not every decode step or an exhaustive operator proof.

Natural candidate stages are compared against F16 under unchanged maximum
absolute0.002 and minimum row-cosine0.99999 gates. Natural router-ID changes are
reported separately, since quantization may change routing. For each selected
weight, a separate CPU probe feeds the frozen F16 normalized input and expert
IDs to actual stored-type `ggml_mul_mat_id` and a dequantized-F32 control. It
compares the actual F16 capture, fixed-input F16 versus Q4, and Q4 stored-type
versus dequantized-F32. This distinguishes intended quantization loss from
implementation drift without asserting full-network causality.

Every floating comparison uses lossless raw-F32 GGUF adapters, actual
`crispembed_diff::Ref`, and independent NumPy metrics. Reports retain maximum,
mean and RMS errors, signed mean error, both L2 norms, signed/absolute ranges,
row cosine and first failing stage. Real comparator positive, numerical-rejection
and wrong-shape controls run before model access. Numerical failures remain
structured results and do not abort remaining candidates or pages.

Candidate task checks separately require exact body text (including table HTML),
aligned box coordinates within3 of original and completion. Exact original IDs
and full raw text are also reported; task tolerance never changes the numerical
gates or turns a failed numerical candidate into a qualified model. All raw
stages, command logs and per-file hashes remain in hosted output. Candidate
weight files alone are task-owned scratch and are removed after their full
panel; every unmodified source weight is reproducible from exact pins. No model
publication is authorized by this worker.

The prepared entry point is
[`tools/kaggle/unlimited-ocr2-q4k/kernel.py`](../tools/kaggle/unlimited-ocr2-q4k/kernel.py).
Render `@COMMIT@` only after source review, check all current hosted CPU jobs
against the five-job cap immediately before submission, and retain the returned
immutable service version. Resource guards require10GiB available RAM and32GiB
scratch; builds use two workers and the panel four CPU threads. The overall
11.2-hour deadline includes builds/conversions. Large raw results stay remote;
fetch only the needed small JSON/Markdown/log reports. There has been no actual
execution of this new protocol yet.

### Separate retained real-scan decoder control

The hosted worker also runs the [production text-decoder replay](../tools/unlimited-ocr2/replay_real_decode.py)
on the twenty frozen original and twenty initial native real-scan ID streams.
It uses the already authenticated published F16 file's 128,000 base pieces,
checks every piece against the pinned original tokenizer, and tests every
original831 added token through the actual runtime fallback. All twenty original
raw strings must match their frozen bytes; C++ clean rendering must also match
its independent Python implementation on all forty streams.

One failed encoded-image proof remains ineligible and visible. All eleven
eligible pairs with identical IDs must now have identical corrected raw text;
the other eight retain their raw/clean comparisons and pending model attribution.
This adds no model/original forward and cannot claim repaired whole-model
real-scan parity. It runs before the separately scoped four-page quantized model
panel and does not change its inputs, recipes, numerical or decoded gates.
