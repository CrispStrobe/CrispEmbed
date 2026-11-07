# Reproduce the experimental weighted Q4KM-style profile

This route converts the pinned original F16 carrier once and measures a named
custom profile. It does not admit a production model, alias or generic Q4_K_M
claim. The public OCR entry point continues to reject unqualified LightOn4B
containers. Run this lane on Kaggle CPU; no GitHub CI dispatch is required.

The profile `lighton4b-weighted-q4km-style-q6head-v1` has216 Q4_K body tensors,
32 Q6_K body tensors, a Q6_K tied embedding/head and474 protected F32 tensors.
Its33 higher-precision names match the pinned publisher's named schedule;
HF grouping and quant-block coordinates prevent publisher-byte equivalence.
The [recipe](../tools/lighton4b_imatrix_q4km_recipe.py) preserves both earlier
32-Q6/Q8-head and24-output recipes.

## Source and statistics guards

The [closed protocol](lighton4b-imatrix-q4km-candidate-protocol-v1.json) pins the
full original carrier SHA, published statistics, adapted statistics, frozen
inputs, null file and both null first-score hashes. Text weights come from the
unquantized original carrier; quantized donors never supply newly generated
weights. The carrier contains F16 text and protected F32 tensors, not a full
F32 neural reference model.

The [adapter](../tools/lighton4b_imatrix_adapter.py) validates248 sum/count pairs,
aggregates them according to pinned upstream semantics, and inversely permutes
24 recurrent-output input vectors into native HF grouping. Counts, widths,
finite values, source domain, full coverage and SHA pins are mandatory. There
is no missing-body-statistics fallback. The publisher supplies no head vector;
the head's unweighted Q6 treatment is explicit. The external statistics corpus,
its underlying license and source-weight identity remain unverified; these are
text statistics, not pretrained image/OCR importance statistics.

On a sufficiently sized hosted worker, the explicit conversion commands are:

```sh
python tools/lighton4b_imatrix_adapter.py \
  --input published-imatrix.gguf --output adapted-imatrix.gguf \
  --receipt adapter.json
crispembed-quantize original-f16.gguf candidate.gguf q4_k \
  --lighton4b-imatrix-q4km-style --imatrix adapted-imatrix.gguf
```

Those names do not waive the full-file identity checks. The dedicated quantizer
flag builds the exact schedule itself, rejects overrides and incompatible modes,
records actual248 weighted applications, and checks the final723-tensor inventory
before atomic publication. Metadata stamps the source, statistic provenance and
coordinate profile. Ordinary quantizer modes and old formats remain intact.

## Bounded Kaggle workers

The [producer template](../tools/kaggle/lighton4b-imatrix-q4km-candidate/kernel.py)
fetches one reviewed source commit, creates an isolated pinned Python environment,
runs model-free controls, and builds only the explicit diagnostic runners. It
checks both frozen plain-Q4 first scores before downloading the original carrier
and quantizing. It audits all723 payloads and all474 protected equalities, then
records candidate first scores and retains the exact runtime executable/library
bundle. The complete model, raw Q6 head fixture and arrays remain remote.

Attach an authenticated retained BF16 source output matching the frozen capture
and array hashes. The metadata template uses a placeholder for that operator-owned
source; no credential or private source reference belongs in Git. If the retained
source is unavailable, the protocol blocks rather than accepting a fresh anchor.
The original F16 and statistics downloads use the pinned public repositories.

Producer budgets are75 minutes for the controller,80 for its bootstrap and90 for
the service, with18 GiB initial free disk and8 GiB available RAM. The null file
is removed only after its two score gates pass; original-plus-output staging is
about13.8 GB. CPU runners are explicit: allocating a GPU does not accelerate this
qualified path.

The [complete-decode template](../tools/kaggle/lighton4b-imatrix-q4km-decode/kernel.py)
requires a sealed producer source/receipt SHA and its mounted output. It checks
runtime executable loaders and input assets before copying the model, independently
rehashes the complete candidate, and reuses the exact producer binaries. Complete
IDs, text, all vocabulary rows and actual EOS or256-token cap are required. The
table child has75 minutes, Arabic20, controller105 and service120. The prior
45-minute/225-row timeout is not complete decoded evidence. Timing uses Python
monotonic and child resource accounting; no external `time` executable is needed.
Peak RSS is explicitly cumulative completed-child usage, not a per-child claim.

The [form template](../tools/kaggle/lighton4b-imatrix-q4km-form/kernel.py) uses the
same sealed model/runtime, first reproduces the producer's table row and common
stages, and checks all five original form input-array hashes by canonical NumPy
serialization. Its form child has90 minutes, controller105 and service120. Form
is a previously observed post-selection diagnostic, not heldout. It neither
changes the fixed recipe nor consumes other heldout cases.

All paths retain partial failures and keep reference captures out of production
artifacts. The pinned table/Arabic and form floor descriptors compare raw CER/WER
against the same BF16 and observed llama.cpp Q4_K_M outputs. This observed-case
floor cannot authorize shipping or imply broad OCR quality.

## Exact executed sources and current limits

The producer executed [`a4f2c908f`](https://github.com/CrispStrobe/CrispEmbed/commit/a4f2c908ffaf3f17eb770973017e3de8a08bb121)
and produced candidate SHA
`5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29`
(4,041,905,312 bytes). All248 applications,216/33/474 types,474 protected payloads
and both null first-score hashes passed. Its CPU identity keys were not stripped;
raw vendor/model/flags remain empty, while four CPUs and their quota are recorded.
The future metadata parser fixes that omission without relabeling the old run.

The first decoded continuation stopped before model forward on an unavailable
external timer. Corrected source [`e822a6de8`](https://github.com/CrispStrobe/CrispEmbed/commit/e822a6de84c8fb546623a464c7e9c79a62e19bbc)
uses standard-library resource accounting and checks loaders/assets before model
copy. Its table/Arabic complete run is pending at this source-integration snapshot.
The [stacked qualification evidence](https://github.com/CrispStrobe/CrispEmbed/blob/2366b72d0/docs/lighton4b-imatrix-q4km-candidate-results.md)
preserves both initial and corrected sources, raw receipts and mixed first-score
metrics. No complete quality result or fixed remote-retention expiry is inferred.

The [read-only score tool](../tools/lighton4b_first_score_diagnostics.py) authenticates
BF16-expanded F32 score rows and reports raw/centered RMS and source-to-candidate
one-position KLD. It never loads a model. These posthoc diagnostics are not the
publisher's corpus KLD benchmark or a substitute for complete decoded output.
