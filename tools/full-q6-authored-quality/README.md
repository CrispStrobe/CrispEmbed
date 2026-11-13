# Authored physical Q6 versus Q4 document screen

[Completed result](../../docs/lighton4b-authored-q6-q4-quality.md): both profiles
produce identical IDs/text/EOS on all three pages; no Q6 quality gain observed.

Three new MIT-licensed clean printed pages have explicit authored transcriptions:
a field report, a 16-cell invoice and an accented French/German/Portuguese dispatch
note. The final 512×512 PNGs use directly rendered 14-pixel body text and 19-pixel
headings, within the existing native 1,024-patch bound. The PNGs were visually inspected against the gold before model
execution. Workers consume these frozen PNGs, never rerender them. The generator
records font and renderer identities; another font/renderer can produce different
bytes. This panel covers Latin script and synthetic clean print only.

Run the separate `worker.py` on Kaggle CPU with the retained observed full-Q6
model/runtime output and source-aligned observed Q4 model output mounted. No
original model, BF16, prior-case inference, new import or runtime build is needed.
Both precision profiles use the same retained runtime, four CPU threads,
independent native vision/greedy decoding and an unchanged 512-token cap. Q4
repacking is enabled only for Q4. All six arms continue after decoded negatives.

The report retains raw text, content CER/WER using the existing versioned
`tools/ocr_quality.py` policy, strict 16-cell invoice structure, token counts,
EOS/caps, finite vocabulary/greedy checks and paired prefill-layer absolute
errors/magnitudes. Markdown formatting is reported separately from content.
This is a small quality screen, not a representative benchmark, source-operator
parity, performance comparison, release, default or application promotion.
Large rows/layers stay remote; only needed reports/logs are retrieved.

Set `HEAVY_OUT` and `HEAVY_SCRATCH` to new task-owned directories, then run:

```sh
python tools/full-q6-authored-quality/worker.py
```

The worker needs 10 GiB available RAM and 12 GiB output disk. It uses the existing
full-Q6 runtime/profile evidence as practical artifact metadata, with ordinary
transport/setup retries allowed. It does not require prior panel pass statuses,
baseline inference, source-capture repetition or an approval acknowledgement.

The first attempt at source `1fed3673a` stopped at preprocessing (return 6)
before vision/decode because the original 1000×1000 render exceeded that bound.
The normal retry directly rerenders the unchanged authored text at 512×512;
all three revised images were inspected again. No arithmetic or runtime limit
was changed, and the initial failure is retained separately.
