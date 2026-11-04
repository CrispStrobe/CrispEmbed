# MinerU 2604 Q4 real-scan text experiment

This bounded experiment uses the published `mineru25-pro-q4_k-mix` alias and
`docs/ocr-real-scans-v2.json`. It exercises whole-image text recognition through
`crispembed --ocr`, not the two-step document, layout, table or reading-order
pipeline. The manifest includes partial CORD references: interpret their word
recall separately and do not treat CER as complete transcription quality. This
is an exploratory dataset, not an independent calibration/held-out layout set.

The `native-mineru-q4` mode authenticates every selected encoded input and the
registry-pinned model, uses the unchanged manifest caps and scores, and records
failed and skipped pages, output, elapsed time and peak child RSS. It fails before
inference if no inputs are admitted or the model smoke/hash check fails. The CLI
does not expose MinerU IDs or EOS here: `ids` and `cap_hit` are null when not
observed, not evidence of completion. A successful process or a high score does
not establish EOS, full document quality, model parity or Metal performance.

Run only on hosted infrastructure after reviewing the exact committed source;
render the existing kernel placeholders with that commit and mode and retain
its complete JSON, input hashes and logs. Review all rows and input failures,
coverage, time-budget skips and per-page quality before summarizing. Follow with
an independently annotated layout/reading-order and table panel through the
full document API; use model-backed EOS and token traces before any completion
or Q4 qualification claim. No weights or large build outputs belong in Git.
