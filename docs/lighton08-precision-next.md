# LightOnOCR-3-0.8B: review under the current precision policy

The 1 November owner policy uses minimum final-output cosine, with strict
intermediate budgets and decoded differences retained as diagnostics. The
earlier LightOn decisions predate that policy. A read-only review of their
complete recorded vocabulary vectors is now reproducible with:

```sh
python tools/lighton08_precision_review.py --output retained-review.json
python tests/test_lighton08_precision_review.py -v
```

The actual result is
[`lighton08-precision-review-20261103.json`](../tests/results/qwen35-ocr/lighton08-precision-review-20261103.json).
Every input receipt's SHA-256, producing source/workflow and original result
are retained. For each of the three frozen pages the tool checks the entire
473/683/248-position original teacher-forced sequence and all 248,320 vocabulary
values per recorded score vector. Missing, reordered, duplicate or top-k-only
scores cannot silently pass.

| Recipe | Worst recorded final-logit cosine across all three cases | Current owner bar | Recorded numerical comparison |
|---|---:|---:|---|
| F16 text / F32 vision | 1.0 at the receipt's printed precision | > 0.98 | Clears the bar |
| Q8 text / F32 vision | 0.999256611 | > 0.97 | Clears the bar |
| Q8 attention / Q6 FFN | 0.997738481 | No separately stated Q6 mixed bar | Report only; do not call it Q8_0 |
| Q4 text / F32 vision | 0.928607225 | > 0.94 | Fails; receipt and inventory are below the bar |

This does not rewrite historical verdicts or qualify current main. The F16
receipt records `GGML_NATIVE=false`; the quantized receipts do not contain an
explicit portable-build record. A current qualification must verify both
`GGML_NATIVE=OFF` and `CRISPEMBED_NATIVE=OFF`, the exact runtime source and the
actual model/recipe hashes. Printed cosine 1.0 is not bitwise identity.

Decoded findings remain separate: F16 matches all three original outputs and
completes each page. Q8 changes the receipt/book text while completing them.
Q4's receipt is incomplete, its book differs, and only the inventory has exact
IDs. All strict vision/prefill/score diagnostics remain in the original receipts.
Teacher-forced agreement does not establish native free-generation correctness
or independently annotated document quality.

## Next bounded run

1. Freeze a new **current-source portable replay** referring to the existing
   model revision `36d8636925327c671bd8e479c517a004f7566bd9`, unchanged three input
   derivatives, original prompts, 1024 cap, original gold and old scorer. Preserve
   all old outcomes. Use a new receipt/protocol identity for the current runtime.
2. Run the same-source F16 control and the existing Q8 recipe. Verify all original
   frozen anchors before comparing the new native arm. Measure the complete
   final-logit minimum, worst intermediate, free-generation IDs/text/stops,
   CER/WER/structure and portable build metadata.
3. If Q8 clears the current policy, prepare an explicit precision decision with
   the changed text and representative-quality limits. Publication still needs
   complete source/artifact/license/consumer evidence; this review alone cannot
   create an alias or flip a default.
4. Keep Q4 rejected on the recorded policy comparison. A mixed Q4 search needs
   a separately frozen development protocol and full end-to-end evidence; do not
   tune its recipe against the already used evaluation pages.

The preserved book-vision operator attribution remains useful diagnostic work.
Its old absolute errors are not, by themselves, a reason to spend another broad
vision sweep before checking the current final-output acceptance policy.

## Prepared current-source portable job

`tools/ci-heavy/lighton08_portable.sh` now implements a separate replay. Its
versioned protocol is `tests/fixtures/lighton08-portable/protocol.json`.
Each dispatch admits exactly one of the existing three cases and reconstructs
the historical F16/F32-vision GGUF with the original frozen converter. It checks
the full baseline and Q8 model hashes, all protected tensor bytes, the immutable
prepared PNG archive, unchanged gold/scorer/prompt/cap, and the full/compact
original capture before native execution. Original outputs and CPU controls
must reproduce the retained reference except for measured generation time.

```sh
gh workflow run heavy-cpu.yml --ref feat/ocr-benchmark-integration \
  -f script=tools/ci-heavy/lighton08_portable.sh \
  -f args=inventory-table -f build_mode=portable \
  -f isolated_lane=lighton08-portable-inventory
```

The other two dispatch arguments are `historical-receipt` and `two-column-page`,
with separate lane names. Collect all three at the same source commit before
making a panel-level precision decision. The job builds with both native-CPU
flags off, records every current Qwen source hash and native binary hash,
compares all 77 prefill stages and every 248,320-wide generated score vector,
and retains native IDs, text, confidence, timing, EOS/cap and old quality scores.
The original arrays and comparison logs remain in the hosted artifact; weights,
package environments and builds stay in scratch. It requires at least 10 GiB
available RAM, 15 GiB free scratch and 8 GiB free output capacity before
installing dependencies. When scratch and output share a filesystem, the job
requires their combined 23 GiB budget.

The unchanged native test harness enables the quantized-weight gate during
both numerical arms and separately verifies literal `0` admission behavior;
the report records that override explicitly. Strict intermediate/text failures
stay visible even when the current numerical bar is met. A successful one-case
job neither completes the three-case panel nor qualifies publication.
The eight new lightweight controls and the preflight pass locally and in the
[hosted shared controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41735520715). They
reject tokens after EOS without changing the frozen historical helper. The actual
original/native replay remains pending hosted dispatch.
