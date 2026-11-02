# TeleOCR validation record

Date: 31 October 2026.

**Experimental; disabled by default. Strict numerical qualification: false.**
Enable the native path explicitly with `CRISPEMBED_TELEOCR=1`. The complete
F32, F16 and Q8_0 panels reproduce the original model's generated token IDs,
decoded text and EOS on all six frozen cases. Q4_K matches five of six cases;
its code output adds one trailing newline. All 24 runs reach EOS, and all have
complete stage coverage and valid confidence counts. Intermediate activations
exceed the fixed numerical budgets, so every `numerical_pass` and overall
`pass` remains false. Output agreement, interface tests and successful
conversion do not erase these failures.

This report describes the retained evidence in
[`tests/results/teleocr/`](../tests/results/teleocr/). See
[usage and build instructions](teleocr.md) for the implementation contract and
[the license audit](teleocr-license.md) for the source boundary. The evidence
separates original-model comparisons, native interface consistency, document
parser smoke checks, storage audits and tests that require no checkpoint.

## 1. Sources, licenses and converted artifacts

| Component | Pinned source |
| --- | --- |
| CrispEmbed patch base | `15adb672f71f8470a23db1284dafa102515adae9` |
| Included ggml submodule | `a06fc0bb46b73cc10011fc90ae4054aee8438aeb`, from `CrispStrobe/ggml` |
| TeleOCR checkpoint and original module | `XingChen-AGI/TeleOCR`, revision `b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d` |
| Apache-licensed Rust reference | `rzafiamy/teleocr-rs`, commit `957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe` |

The [source manifest](../tests/results/teleocr/source-manifest.json) records
checkpoint, configuration, tokenizer, original-module and image hashes.
`model.safetensors` has SHA-256
`9817b18041bd96403f75e38a33b28ed0cd5fb2641f67eead673afabd3c408109`;
the executed original `modeling_naviocr.py` has SHA-256
`7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922`.

The original GitHub repository lacks a repository-wide license, but the pinned
HF model module has an explicit Apache-2.0 source header. The implementation
uses that licensed architecture/reference source, the Apache-2.0 Rust port,
and CrispEmbed's existing native engine. The inspected unlicensed official
document-client and table-postprocessor files were not directly translated.
Original CrispEmbed and converter code retain MIT terms; identified Apache
adaptations retain [NOTICE.teleocr](../NOTICE.teleocr) and the
[Apache license](../models/licenses/apache-2.0/LICENSE). The checkpoint carries
the publisher's Apache-2.0 declaration. The exact initial Qwen2.5-VL vision
weight revision is not established by the published provenance; this report
does not claim a complete training-weight lineage audit. The
[license audit](teleocr-license.md) links the exact upstream declarations.

All 701 source tensors were checked. The duplicate tied `lm_head.weight` was
verified against the token embedding and omitted, producing 700 tensors per
GGUF. Conversion reads JSON and safetensors data without executing checkpoint
Python. The four actual output artifacts are identified below; their full
tensor inventories are in the linked manifests.

| Artifact | Bytes | Actual tensor types | SHA-256 |
| --- | ---: | --- | --- |
| [F32](../tests/results/teleocr/model-manifests/teleocr-f32.gguf.json) | 5,043,933,472 | 700 F32 | `054be3381a91fa3e641271ece00de8a99ef2d6a6f74e1239fa6f89650f9996cb` |
| [F16](../tests/results/teleocr/model-manifests/teleocr-f16.gguf.json) | 2,526,109,984 | 360 F16; 340 F32 | `6c52a2800e313964717cf4e5d06e0392c2b5936d8adeac423d563b8a026a2780` |
| [Protected Q8_0](../tests/results/teleocr/model-manifests/teleocr-q8_0.gguf.json) | 2,113,233,248 | 196 Q8_0; 164 F16; 340 F32 | `9de7e45ce6c13ba03d3d2879f2a1bff9f8407e12808be588cf057cd2553f224b` |
| [Protected Q4_K](../tests/results/teleocr/model-manifests/teleocr-q4_k.gguf.json) | 1,893,033,088 | 196 Q4_K; 164 F16; 340 F32 | `1933ab72f57d14bd0473a86ecadb4bd815e13003caa40ab94478727ac12770cd` |

The quantized recipes compress decoder attention/FFN matrices. They preserve
vision, merger and the tied embedding in F16, plus normalization and bias
vectors in F32. The Q4 command is:

```sh
build/crispembed-quantize teleocr-f16.gguf teleocr-q4_k.gguf q4_k \
  --tensor-type 'v.*=f16' --tensor-type 'l.embed_tokens.weight=f16'
```

The [final Q4 audit](../tests/results/teleocr/quantize-q4-final-audit.json)
verified all shapes, all 504 protected tensor payloads against F16, and all
700 payloads against the initial Q4 artifact. The latter initially inherited
an inaccurate `f16` recipe and used `general.file_type=12`. The scoped metadata
fix writes the actual ordered recipe, resulting tensor census and correct
Q4_K file type `14`. All tensor payloads remained byte-identical; the original
[negative audit](../tests/results/teleocr/quantize-q4-initial-audit.json) is
retained. The [final F16 conversion check](../tests/results/teleocr/convert-f16-final-verification.json)
also confirms that enriching source-hash provenance did not change the F16
file or any of its 700 tensor entries.

## 2. Original-model reference and frozen task panel

The reference executes the pinned, inspected original HF model module on CPU
in float32, using PyTorch `2.8.0+cpu`, Transformers `4.57.1`, NumPy `2.5.4`,
Python `3.12.14`, SDPA attention and two threads. Image preprocessing uses the
slow `Qwen2VLImageProcessor` (`use_fast=False`) with **Pillow 13.0.0**.
The generation cap was frozen at **1,024 new tokens per case**, with greedy
decoding and repetition penalty `1.05`. The accepted EOS IDs are `151645` and
`151643`; all six actual reference sequences ended with `151645`, before the
cap. These settings and complete IDs are retained in the
[reference records](../tests/results/teleocr/reference/reference-results.json).

Five images came from the pinned publisher snapshot. The sixth is an authored
three-line text fixture, frozen with its own gold before inference. The table
shows the image handed to preprocessing, before the processor's patch-grid
resize. Layout deliberately uses a 1,036 × 1,036 input derived from the
publisher's 1,653 × 2,206 image.

| Case | Source dimensions | Input dimensions | Prompt tokens | Generated IDs, including EOS | Task |
| --- | --- | --- | ---: | ---: | --- |
| Text | 941 × 205 | 941 × 205 | 268 | 66 | Plain text |
| Formula | 824 × 435 | 824 × 435 | 500 | 166 | LaTeX |
| Table | 482 × 164 | 482 × 164 | 140 | 283 | OTSL |
| Code | 931 × 1,436 | 931 × 1,436 | 1,717 | 366 | Code parsing |
| Layout | 1,653 × 2,206 | 1,036 × 1,036 | 1,396 | 768 | Layout boxes/labels |
| Synthetic text | 640 × 220 | 640 × 220 | 214 | 19 | Authored transcription gold |
| **Total** | | | | **1,668** | Six complete sequences |

Every case retains **75 required tensor observations**: all 32 vision-block
outputs, all 28 decoder-block outputs, and 15 input, merger, attention,
normalization, logit and generated-ID boundaries. Intermediate decoder
observations cover the captured prompt/prefill execution; generated IDs cover
the full autoregressive sequence. They are not a capture of every layer at
every decode step. The [independent reference review](../tests/results/teleocr/reference-peer-review.json)
checks tensor inventory, shapes, image hashes, prompt lengths, IDs and EOS.

The [executed producer snapshot](../tests/results/teleocr/dump_teleocr_reference.executed.py)
and [producer receipt](../tests/results/teleocr/reference-producer.json) are
retained. The delivered dumper subsequently gained configuration preflight,
required-stage inventory and output-preservation checks. Review found no
change to inference, prompts, images or hook mathematics. A hooked/unhooked
original-model control matched on the text case; it is not evidence for
observer neutrality on every possible input.

## 3. Native differential results

The native comparator uses CPU execution with two threads and each case's
frozen 1,024-token cap. All **24 cases are complete**. The independently
aggregated [final matrix](../tests/results/teleocr/summary.json) preserves the
following separate verdicts. The [independent full-matrix review](../tests/results/teleocr/final-matrix-peer-review.json)
rechecks all 24 reports, 6,672 generated IDs and 1,800 stage records.
“Exact decoded” requires every generated ID,
complete raw text and EOS to agree with the original. “Coverage” means every
required stage was observed, not that its values passed the numerical gate.

| Precision | Exact decoded cases | EOS reached | Stage coverage | Valid confidence counts | Strict numerical cases passing |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 6/6 | 6/6 | 450/450 | 6/6 | **0/6** |
| F16 | 6/6 | 6/6 | 450/450 | 6/6 | **0/6** |
| Protected Q8_0 | 6/6 | 6/6 | 450/450 | 6/6 | **0/6** |
| Protected Q4_K | **5/6** | 6/6 | 450/450 | 6/6 | **0/6** |
| **Total** | **23/24** | **24/24** | **1,800/1,800** | **24/24** | **0/24** |

Each precision produces 1,668 IDs across the six cases. Inputs are exact on
all 24 cases: image patches, grid, prompt IDs and rotary positions. The
retained run receipts are [F32](../tests/results/teleocr/native-f32/summary.json),
F16 [small](../tests/results/teleocr/native-f16-small/summary.json) and
[large](../tests/results/teleocr/native-f16-large/summary.json), Q8_0
[small](../tests/results/teleocr/native-q8-small/summary.json) and
[large](../tests/results/teleocr/native-q8-large/summary.json), and Q4_K
[small](../tests/results/teleocr/native-q4-small/summary.json) and
[large](../tests/results/teleocr/native-q4-large/summary.json).

Every case has one finite confidence in `[0,1]` per generated ID, including
the terminal token. This checks confidence integrity and count; it does not
establish equality to original-model probabilities or confidence calibration.
Every case process returns exit code `1` because its strict numerical
comparison fails, and those exit codes are retained.

### The Q4_K code-output difference

The [actual Q4_K code report](../tests/results/teleocr/native-q4-large/code.json)
contains exactly the 1,260-character reference plus one final newline. At
zero-based generated index 364, reference token `92` (`}`) becomes native
`532` (`}\n`); both sequences contain 366 IDs and finish with the same EOS.
The original tokenizer independently reproduces both recorded strings, as
checked in the [quantized-output review](../tests/results/teleocr/native-quant-peer-review-partial.json).

| Q4_K code metric | Edits / reference units | Value |
| --- | ---: | ---: |
| Raw CER | 1 / 1,260 Unicode characters | 0.0793651% |
| Whitespace-normalized CER | 0 / 1,031 characters | 0% |
| Whitespace-based WER | 0 / 134 words | 0% |

The raw decoded gate remains **failed**. Whitespace normalization is a
separate diagnostic score, never a replacement for the frozen raw comparison.
Q4_K layout and the four smaller cases match their full reference outputs.
This single formatting difference does not demonstrate a semantic code error,
but it prevents an exact raw-output claim for the complete Q4_K panel.

| Precision | Strict stage passes / 450 | Maximum floating-stage absolute error | Minimum stage row cosine |
| --- | ---: | ---: | ---: |
| F32 | 174/450 | 1.39697265625 | 0.999994938501 |
| F16 | 174/450 | 1.16235351562 | 0.999996443749 |
| Q8_0 | 144/450 | 72.8646240234 | 0.990959283043 |
| Q4_K | 143/450 | 857.625 | 0.55391287739 |

These extrema aggregate different layers and coordinates; they are not the
final OCR quality score. Quantized internal drift is measured separately from
the original F32 rounding investigation below. The latter does not explain
away quantization error or qualify either quantized candidate.

### Fixed acceptance criteria

The [comparator](../tests/test_teleocr_diff.cpp) applies all of these checks:

- `image_grid`, `prompt_ids`, `mrope_positions` and `generated_ids`: exact
  equality, maximum absolute error `0`.
- `input_patches`: maximum absolute error at most `0.000002` and minimum
  row cosine at least `0.99999`. The corrected six-case F32 panel actually has
  maximum input-patch error `0` on every case.
- Other floating-point observations: maximum absolute error at most
  **`0.002`** and minimum row cosine at least **`0.99999`**, with finite values.
- First-step logits additionally require the same argmax token.
- All reference observations must be present with the required sizes and row
  widths. Decoded text, complete generated IDs, EOS and valid confidence
  counts are checked separately. Overall `pass` requires all components.

These budgets were not relaxed after seeing the results. Cosine, relative
error, ULP counts and isolated operator probes below are explanatory
measurements; none replaces the recorded acceptance criteria.

### F32 measurements and localized numerical differences

The [independent F32 review](../tests/results/teleocr/native-f32-peer-review.json)
confirms exact pixels, grid, prompt IDs and rotary positions on all six cases,
plus exact full decoding and first-logit argmax. First-logit maximum absolute
errors are small even though earlier unnormalized activations fail:

| Case | First-logit maximum absolute error | Strict numerical verdict |
| --- | ---: | --- |
| Text | `0.0000121593475342` | Fail |
| Formula | `0.0000495910644531` | Fail |
| Table | `0.0000371932983398` | Fail |
| Code | `0.0000159740447998` | Fail |
| Layout | `0.0000119209289551` | Fail |
| Synthetic text | `0.0000108480453491` | Fail |

The text case's earliest investigated failing vision boundary is block 17;
the preceding block passes. Decoder block 2 likewise fails after block 1
passes. Coordinates below use `[row, channel]`, and block numbers are zero
based. The [paired-activation advisory](../tests/results/teleocr/f32-paired-activation-advisory.json)
preserves measured source/native values and per-coordinate errors.

| Boundary | Maximum absolute error | Reference at worst coordinate | Native at worst coordinate | ULP distance there | Elements above `0.002` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision block 16 | `0.000267028808594` | `-64.7177963257` | `-64.7180633545` | 35 | 0/1,218,560 |
| Vision block 17, `[913,849]` | `0.08673095703125` | `-926.9636840820` | `-926.8769531250` | 1,421 | 11/1,218,560 |
| Decoder block 1 | `0.0009765625` | `364.2160949707` | `364.2170715332` | 32 | 0/274,432 |
| Decoder block 2, `[1,35]` | `0.03125` | `39418.69921875` | `39418.66796875` | 8 | 8/274,432 |

Vision block 17's error RMS divided by reference RMS is approximately
`4.8583e-6`, with minimum row cosine `0.999999999992`. Decoder block 2's ratio
is approximately `4.7524e-7`, with minimum row cosine `0.999999999991`. At the
decoder's worst coordinate, one float32 spacing is `0.00390625`, already
larger than the absolute budget. These scale observations help explain why a
small relative change can fail this particular absolute gate. They do not
make the failing values exact or qualified.

The original licensed vision block was then run on the native previous-layer
output. At `[913,849]`, it produced `-926.8772583008`, reproducing
`0.08642578125` of the native pipeline's `0.08673095703125` discrepancy:
approximately **99.65% at that coordinate**. The remaining native/source
difference there was `0.00030517578125`. This supports propagated
floating-point sensitivity as the cause of that large local gap. It is not a
proof for every coordinate: the same-input full-block comparison still has a
maximum error `0.00390625` elsewhere and fails the fixed gate. See the
[vision propagation report](../tests/results/teleocr/portable-diagnostics/vision17/propagation.json).

Isolated native FFNs fed the same source inputs have output maximum errors
`0.00244140625` at vision block 17 and `0.03515625` at decoder block 2.
Evaluating the source FFN equations in F32 versus F64 gives output maximum
errors approximately `0.0032409742` and `0.0153265806`, respectively.
The [vision native log](../tests/results/teleocr/portable-diagnostics/vision17-native.log),
[decoder native log](../tests/results/teleocr/portable-diagnostics/decoder2-native.log),
[vision F32/F64 comparison](../tests/results/teleocr/portable-diagnostics/vision17/original-f32-vs-f64.json)
and [decoder propagation/F32-F64 report](../tests/results/teleocr/portable-diagnostics/decoder2/propagation.json)
retain their own failures. All four portable producer/native diagnostic
phases returned the expected strict-failure exit code `1`, recorded in the
[diagnostic receipt](../tests/results/teleocr/portable-diagnostics/receipt.json).
The [portable probes and rerun instructions](../tools/teleocr_diagnostics/README.md)
are included. F64 is diagnostic data, never substituted for the frozen
original F32 reference.

### Preserved initial failures and corrections

The [initial native text run](../tests/results/teleocr/native-f32-initial/summary.json)
matched decoded output but exposed a preprocessing difference and a missing
post-attention observation. The latter came from an unused observation node;
wiring the contiguous tensor into the live graph restored coverage. The
preprocessing difference was isolated to Pillow 13's separable bicubic pass
order. The scoped native implementation selects the vertical pass first when
the positive height reduction exceeds twice the width reduction. With this
correction, all six frozen input-patch arrays match exactly. The
[Pillow comparison report](../tests/results/teleocr/preprocess-pillow13/results.json),
[13 preprocessing checks](../tests/results/teleocr/preprocess-final.log),
[first corrected text run](../tests/results/teleocr/native-f32-pillow-fix/summary.json)
and original failures remain available. The later activation failures above
persist after these corrections.

## 4. Production interfaces and document processing

### Native public interfaces and compiled consumers

The [public F16 suite](../tests/results/teleocr/public-f16/suite.json) ran
actual native inference through the C ABI, Python wrapper, explicit OCR
orchestrator stage, CLI and HTTP server. Its bounded eight-token table-image
anchor was `表8 出厂检验和型`, exercising non-ASCII UTF-8. This is intentionally
a capped interface anchor, not a completed table transcription or an
independent original-model reference.

The [individual public checks](../tests/results/teleocr/public-f16/public.json)
cover disabled-gate rejection, default and changed caps, prompt setting/reset,
generation settings and invalid values, native text/confidences, Python
consistency, stage 24 selection and wrong-model rejection, raw and pipeline
CLI paths, HTTP invalid-image handling, per-request cap restoration and HTTP
confidence output. The gate follows the existing shared convention: unset,
empty and literal `0` disable it; other nonempty strings, including `false`,
enable it.

Rust and Dart were also exercised as actual compiled consumers using their
safe wrappers and the same native anchor. Rust's release build and Dart's
analysis/native-executable compilation succeeded; both consumer executions
returned `0`. The [build receipt](../tests/results/teleocr/binding-compilation/build-checks.json)
records Rust `1.99.1`, Dart `3.13.7` and Dart FFI `2.2.0`, separately from the
subsequent model-backed executions in the public suite. The consumers check
stage 24, generation settings/reset, UTF-8 text and context ownership. Their
source snapshots and hashes are retained. These results establish interface
consistency on this sample, not a second independent inference implementation.

An additional [production CLI observer control](../tests/results/teleocr/production-observer-control-f16/report.json)
used the normal F16 CLI, without test-only activation observers, on the frozen
text case. All 66 generated IDs, decoded text and EOS matched the original.
This control does not compare activations; the CLI's JSON also does not expose
the complete confidence array. Those checks belong to the differential and
public C API evidence above.

The final review found and closed a malformed-file admission gap: a correctly
shaped F16 or quantized norm/bias vector could reach an unsupported GGML CPU
elementwise operation. The loader now rejects unsupported tensor types and
unflattened ranks before weight allocation, and checks type/contiguity after
loading. The [manifest preflight](../tests/results/teleocr/final-dtype-guard-preflight.json)
confirms that all four tested artifacts already satisfy the new rule: each
contains 340 F32 norm/bias vectors.

The [final native loader run](../tests/results/teleocr/teleocr-loader-post-review.json)
passes **43/43** cases, including 17 new tensor-header controls and negatives.
The [rebuilt production CLI](../tests/results/teleocr/production-post-review-f16/report.json)
again reproduces all **66 IDs, full text and EOS** on the frozen F16 text case.
The [build receipt](../tests/results/teleocr/build-post-review.json) records the
final library/CLI/server/comparator hashes and confirms that test-observer
exports are absent from the production shared library. The
[installation check](../tests/results/teleocr/install-post-review.json) verifies
that MIT, TeleOCR NOTICE and Apache-2.0 license text are installed unchanged.

The earlier full panels and public consumers retain their original source
snapshots and binary hashes. The [source-delta review](../tests/results/teleocr/final-source-review/source-delta.json)
reconstructs the exact earlier runtime bytes after removing only the three
loader-validation edits. Getter header changes are explanatory comments; all
other recorded public-suite sources are byte-identical. No graph, arithmetic,
preprocessing, decoding or threshold changed. This targeted final-source
check does not claim that every earlier public consumer or full-model case
was rerun after the loader-only edit; the separate run receipts identify the
actual binary used for each execution.

### Two-pass document parser

The optional Python parser performs layout interpretation, cropping and
formatting around the native GGML engine. The
[F16 parser suite](../tests/results/teleocr/parser-f16/suite.json) ran two
sequential end-to-end smoke cases, with a 1,024-token cap for each generation:

| Mode and input | Native generations retained | Typed blocks | Generated tokens in total | Verdict |
| --- | ---: | --- | ---: | --- |
| [Detection, synthetic text](../tests/results/teleocr/parser-f16/detection/report.json) | 4 | Three `title` blocks | 92 | Pass; no cap reached, no warnings |
| [Segmentation, table image](../tests/results/teleocr/parser-f16/segmentation/report.json) | 3 | `table_caption` and `table` | 391 | Pass; polygons retained, no cap reached, no warnings |

Each case ran layout and crop extraction, produced nonempty Markdown and typed
JSON, kept finite bounded confidences, and passed serialization/output checks.
Raw layout and crop generations are retained alongside
[detection Markdown](../tests/results/teleocr/parser-f16/detection/document.md)
and [segmentation Markdown](../tests/results/teleocr/parser-f16/segmentation/document.md).
These assertions do not establish correct semantic block labels, layout
accuracy, table-cell accuracy or document-wide transcription accuracy. For
example, the synthetic lines being labeled `title` is an observed model
output, not a label-accuracy pass.

### Independent transcription gold and PDF ingestion

The [authored fixture](../tests/fixtures/teleocr/synthetic-text.json) fixes:

```text
Hello TeleOCR
Invoice 42
Total: EUR 19.50
```

The reference output is byte-for-byte equal to that gold; each completed
native synthetic case has the same full output. After collapsing whitespace
while preserving case and punctuation, the
[gold review](../tests/results/teleocr/reference-peer-review.json) reports
**CER `0/41 = 0` and WER `0/7 = 0`**. This is one independently authored,
three-line image. The five publisher examples test implementation agreement;
they were not independently labeled into an OCR accuracy benchmark.

The [PDF test](../tests/results/teleocr/pdf-ingest.json) rendered one page at
72 DPI with `pypdfium2` version `5.14.0`, yielding a 640 × 220 RGB
image. Its scope is **PDF input/rendering only**. Native OCR and parser
inference were tested separately with PNG inputs; the PDF check is not an
end-to-end PDF OCR accuracy or pagination result.

## 5. Tests without checkpoint inference

The [final Python unit log](../tests/results/teleocr/python-unit-tests-final.log)
records **43 passed tests and 5 passed subtests**. These cover converter,
parser and public-surface contracts using bounded fixtures. Converter tests
include synthetic safetensors-to-GGUF F32/F16/Q8 round trips, exact naming and
shape mapping, tied-head handling and malformed/missing/extra input rejection.
These tests are separate from the real 700-tensor conversion and original
model inference results.

| Check | Retained result | Scope |
| --- | --- | --- |
| [Native metadata loader baseline](../tests/results/teleocr/teleocr-loader-final.json) | 26/26 pass | Metadata-only subprocess cases; gate, typed arrays, counts, scalar/array mismatch, format version and conflicting profile rejection |
| [Final native loader with tensor guards](../tests/results/teleocr/teleocr-loader-post-review.json) | 43/43 pass | All 26 baseline cases plus 17 vector/matrix type and tensor-rank controls; no weight payloads |
| [Native quantizer provenance](../tests/results/teleocr/quantizer-provenance-tests.json) | 9/9 pass | Bounded synthetic tensors; ordered/escaped overrides, flags, imatrix recording, type census, standard file types and unrelated-architecture control |
| [Native preprocessing](../tests/results/teleocr/preprocess-final.log) | 13/13 pass | Grid/patch order, repeated temporal frame, RGB/grayscale/RGBA, rounding, Pillow resize order and invalid input bounds |
| [Environment gate regression](../tests/results/teleocr/env-gate.log) | 19 checks, 0 failures | Existing shared enable/disable semantics |
| [BPE pretokenizer](../tests/results/teleocr/bpe-pretokenize.log) | 246 checks, 0 failures | Shared tokenizer regression |
| [Qwen pretokenizer](../tests/results/teleocr/qwen-pretokenize.log) | 39 checks, 0 failures | Qwen tokenizer regression |
| [CLI engine names](../tests/results/teleocr/engine-selectors.log) | 25 IDs reachable | Public engine-name routing |
| [One-shot exit convention](../tests/results/teleocr/clean-exit-final.log) | 198 entry points checked | Repository clean-exit convention |

The initial loader test incorrectly assumed the literal string `false`
disabled the shared gate. The [original failing report](../tests/results/teleocr/teleocr-loader-initial-gate-assumption.json)
is preserved; the corrected test follows the existing implementation instead
of changing a shared policy. A separate
[build-race attempt](../tests/results/teleocr/teleocr-loader-build-race.json)
is also retained rather than presented as a completed loader verdict.

## 6. Reproduction and limits

[The usage guide](teleocr.md#tests-and-interpretation) gives relative-path
commands for dependency installation, a full pinned checkpoint download,
conversion, reference capture, differential runs, public interfaces and
parser checks. The production converter's minimal download is intentionally
insufficient for original-module reference execution: obtain the complete
pinned snapshot, including executable reference source and sample assets.
Use [the reference dependency file](../models/requirements-teleocr-reference.txt)
and retained [environment lock](../tests/results/teleocr/python-requirements.lock)
to reproduce the recorded processor/runtime contract.

Use a new output directory for each reference or native run. The runner
verifies image/reference hashes, uses the cap recorded in each reference,
retains case logs and JSON even after failures, and refuses to overwrite a
nonempty output directory. Preserve nonzero numerical-failure exit codes.
The delivered source and reports include image fixtures, model/tensor
manifests, binary/source hashes and executed-source snapshots. Multi-gigabyte
checkpoints, generated GGUF model files and activation arrays are excluded
from the source deliverable; the pinned sources and tools allow them to be
regenerated. Raw report and snapshot bytes retain their original execution
paths so recorded hashes remain meaningful; documented rerun paths are
portable and relative.

The current evidence supports a bounded CPU implementation and clearly
separated decoded/interface results. It leaves the following limits:

- **Strict numerical qualification remains false**, including F32. The
  diagnostic explanation does not waive any failing observation.
- The task panel contains five publisher examples and one synthetic gold
  image. It does not measure general OCR, handwriting, multilingual, formula,
  table or layout accuracy, long-document behavior or robustness.
- Confidence integrity and interface consistency do not establish confidence
  calibration. Prefill activation coverage does not cover every decode-step
  activation, batch size or sequence length.
- The executed model tests use CPU and two threads. GPU placement is an opt-in
  experimental path; CUDA, Vulkan, Metal, other CPU/kernel combinations,
  macOS, Windows, Android/iOS and WASM do not acquire parity or deployment
  qualification from these results. Dart execution on Linux does not qualify
  a Flutter mobile application.
- Reported run durations are individual validation executions, not a
  controlled throughput/latency benchmark or a comparison with another
  backend.
- There is no verified public model publication, immutable download alias or
  registry hash pin in this change. Local conversion and the recorded
  artifact hashes are the supported delivery path for these experimental
  candidates. Future publication, larger quality panels and additional
  platforms require their own retained checks.
