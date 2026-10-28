# Unlimited-OCR-2 trial follow-up

The frozen synthetic trial and original reference are in
[trial.json](../tests/results/unlimited-ocr2/trial.json). The first verification
[verify-v1.json](../tests/results/unlimited-ocr2/verify-v1.json) retains default
F16 tiling/ID equality, bounded speed measurements and the initial text failure.
The original decode is `tokenizer.decode(gen, skip_special_tokens=False)` with
cleanup disabled (`modeling_unlimitedocr.py:1201` at upstream revision
`04647c7a6098d8d58915f003536a741c82eafb40`). EOS is excluded by the original loop.

The published F16/Q8_0 GGUFs contain only the 128000 base BPE pieces. The tokenizer
also contains 831 added IDs 128000–128830, including layout and table markup.
Runtime source `8ca9d3de2a8160e6f5235156d3550c9719583744` retains the exact added
pieces as a fallback for those existing files. Raw original output is the
default. `CRISPEMBED_UNLIMITED_OCR2_OUTPUT=clean` separately strips layout headers
while preserving table HTML; `=legacy` preserves the prior marker-dropping text.
The completed v2 text verification uses that frozen source and the original four
trial cases. All four default-tiled F16 ID streams and raw UTF-8 byte strings
match the frozen original. The exact producer receipt is
[verify-text-v2.json](../tests/results/unlimited-ocr2/verify-text-v2.json); the
[independent review](../tests/results/unlimited-ocr2/verify-text-v2-review.json)
records each text hash, Q8 differences and clean/legacy controls. Both compiled
text/tiling controls pass, all seven regenerated fixture hashes match, and fresh
alias caches verify both published model hashes. No original was rerun or retargeted.

The converter now merges base and added pieces at their original numeric IDs,
checks duplicate/conflicting IDs and embedding-table bounds, and writes both the
complete `tokenizer.ggml.tokens` array and explicit added-token metadata:
`unlimited_ocr.tokenizer.added_ids`, `.added_content`, `.added_flags`. Flag bits
are single_word(0), lstrip(1), rstrip(2), normalized(3), special(4). Existing
base IDs, BPE merges, tensor mapping and conversion math stay unchanged. Stdlib
controls extract the pure functions without importing any model framework and
check all 831 added pieces against the pinned source fixture. This is a converter
source fix; no new GGUF conversion or weight publication is claimed by these
controls. Existing published weight hashes and runtime fallback remain intact.

The new [card-only workflow](../.github/workflows/unlimited-ocr2-card-refresh.yml)
requires a frozen source with committed `verify-text-v2.json`. Before package,
credential or Hub access, the publisher independently compares its four F16 ID
streams and raw text against the original trial, verifies default tiling/output
and requires compiled text/tiling controls. The committed successful receipt satisfies the original-text gate; coordinator
review and workflow registration precede card publication.

After that gate, `refresh-card` checks the exact previous manifest/card and
verbatim upstream notice hashes. It preserves the old manifest's `files` and
`source_commit`, checks unchanged weight LFS identities, and commits exactly
`README.md` and `manifest.json` against the expected Hub parent. It rereads both
new small files and all unchanged notice hashes at the immutable new revision.
A state receipt records the new commit before readback, retaining partial failure
evidence. No model payload is downloaded or uploaded by this card-only path.
The returned `card-update.json` belongs under `card_update` in the publication
receipt after independent review. Root coordinates workflow registration/main
integration and declaration re-pins; this branch does not push main.

This text-only rerun does not execute the first verification’s before/after,
ablation, exact-decoder, native-build or vision-cache arms. Their missing fields
are unexecuted; this receipt supplies no new speed or exact-decoder conclusion.
The initial failure and broader first-run evidence remain unchanged.

F16 agrees exactly on the trial. Published Q8_0 diverges on
the table token 7 and receipt tokens 131/258 bounding-box coordinates. Its
independently cleaned body matches all four originals, and raw text matches the
paragraphs/German originals. The coordinate differences remain recorded, with
no new precision or quality claim. Q4_K needs a separately frozen experiment;
this change does not modify head, vision or quantization recipes.
