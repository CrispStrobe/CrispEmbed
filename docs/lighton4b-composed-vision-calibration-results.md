# Corrected composed vision: table and Arabic

The bounded Kaggle run (kernel `lighton4b-composed-vision-calibration-v1`) at source `a03412aeb3b4df9c8a5a9f093d9ab699b9d41258` completed in 643.45 seconds. The [raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-composed-vision-calibration/v1/composed-vision-calibration.json) has SHA256 `f69d23ffe6d65d9abf6c7c5c660723613ce84cc901383c5151e17d8f8dc5fe64`.

Both pixels-composed outputs match their source-feature baseline IDs, raw text, all greedy score rows and 38 text stages. Both merged vision arrays match the actual upstream arrays byte for byte. Arabic completes with 6 tokens/EOS and zero character or word edits against both references. Table completes with 282 tokens/EOS, but remains **349/607 character edits versus upstream 348/607** against the original BF16 transcription; both have 97/106 word edits. The table floor fails; the model remains unqualified and must not ship.

Composed child wall times were 201.50 seconds for table and 27.89 seconds for Arabic, with observed RSS 4,290,285,568 and 4,284,391,424 bytes. These are diagnostic measurements. Q4 repack and BF16 head gates remain off; no full F32 weight cache is instantiated.

The independent small-output audit rehashed 12 native files, verified EOS and identical baseline/composed IDs and raw text, and recomputed original-reference edit counts. Large score and activation arrays remain remote. Table first-score RMS/max error against actual upstream is 0.188628814/1.061775208; Arabic is 0.109235580/0.691092014. Actual upstream text internal stages are absent, so this closes the vision confound without identifying the earliest text operator responsible. Held-out qualification remains pending; no aggregate waiver applies.
