# BF16 original versus native Q8/Q4 diagnostic

This separate protocol uses the completed original BF16 free-generation capture;
there is no F32 prerequisite. It preserves document-v1/v2 and their admissions.
The exact original receipt SHA, all177 IDs, decoded text and177 score payload
hashes are frozen in [the protocol](lighton4b-bf16-quant-v1.json). Native admission
is empty: the current runtime rejects4B format5, lacks larger-model ownership
and frontend support, and has no admitted Q8/Q4 weight recipe. No actual native
parity is claimed.

Retrieve the completed original capture from its retained hosted artifact into
an empty directory, including `capture.json` and every `scores-NNN.f32` file.
The consumer authenticates retrieved bytes against the frozen receipt/payload
hashes; it never fetches a model or regenerates the reference. Remote artifacts
may be retrieved directly on the hosted worker; no shared-host model is needed.

A future actual producer must write `native.json` with schema
`lighton4b-bf16-quant-native/v1`, precision `q8_0` or `q4_k_m`, this protocol's
SHA256, `original_capture_sha256`, exact `teacher_forced_ids`, `scores`,
`free_generation_independent=true`, independent `generated_ids`, `decoded`,
`stop`, `cap_hit`, `native_forward_executed=true`, `producer_source_sha`,
`producer_run_url` and `weights_sha256`. Every score record has the original
position, file, shape `[1,248320]`, bytes993280 and SHA256. Teacher forcing must
consume the original prompt and original IDs before that position, never its
own free-generation prefix. Independent generation starts afresh with the
frozen image-only prompt, cache, greedy decoding, EOS248046 and cap256.
A separately reviewed native-receipt-SHA256/source/run/weight/precision tuple must first be added to
`native_producers`; arbitrary producer claims and synthetic fixtures fail closed.

On hosted CI, build target `crispembed-diff-lighton4b-document`, then run:

```sh
python tools/lighton4b_bf16_quant.py \
  --original "$ORIGINAL" --native "$NATIVE" \
  --binary "$DIFF_BINARY" --output "$REPORT"
```

The consumer compares every original-prefix score through unchanged actual
`crispembed_diff::Ref` and bounded one-position GGUF pairs. It preserves max/mean
absolute error, RMS, minimum row cosine and both magnitudes, strict0.002/0.99999
flags, first observed failing score position, and complete original/native
IDs/text/EOS. Numerical and decoded parity are separate required results.
The original contains no internal-stage trace; output-position differences do
not identify the earliest internal operator. Observer-off original equivalence,
representative OCR quality, model/publication qualification and intermediate
attribution remain outside this diagnostic.

[Hosted controls](../.github/workflows/lighton4b-bf16-quant-controls.yml) use
synthetic full-vocabulary rows with the actual Ref binary: equal rows, an error
at position1 and tampered bytes; contract negatives reject missing prefixes,
missing scores, independent-generation omission, wrong precision and stops.
Synthetic controls retain `native_execution_admitted=false` and cannot admit
weights or native execution. Large artifacts stay remote.

Actual [hosted41997183136](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41997183136)
at `a5b005b368e6f47530eea01bdc81d07df4aadf44` passes both control methods,
including actualRef equal/error/tamper and contract-negative subcases. The
[receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-controls.json)
retains exact source and executable hashes. Initial41997070623 was cancelled
while queued after the public-path/forged-producer control amendment; no model
ran in either workflow. The initial transient Git push rejection and absent
new-workflow default-branch dispatch404 preceded the successful exact-source
existing `heavy-cpu.yml` dispatch. Native parity remains pending.
