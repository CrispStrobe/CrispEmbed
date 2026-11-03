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
