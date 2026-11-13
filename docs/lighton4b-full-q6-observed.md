# Physical full-Q6: improved observed BF16 agreement

The native full-Q6 profile reproduces all three frozen upstream Q6 token lists,
decoded strings and EOS stops. Against the same original BF16 outputs it repairs
the table and improves the form over the historical source-aligned Q4_K_M path.

| Case | Historical Q4 character / word edits | Native Q6 character / word edits | Native Q6 tokens / stop |
|---|---:|---:|---|
| Table |348 /97|0 /0|222 / EOS|
| Form |189 /46|154 /31|182 / EOS|
| Arabic |0 /0|0 /0|6 / EOS|

The unchanged BF16 denominators are table 607 characters/106 words,
form 656/90 and Arabic 12/2. These measure agreement with model outputs, not human
OCR accuracy. The form retains a substantial source-Q6 residual against BF16.
The [small result](../tests/results/lighton4b/full-q6-observed.json) includes
separate character/word counts and the historical comparison.

Execution used [source 0b576d0e3](https://github.com/CrispStrobe/CrispEmbed/commit/0b576d0e34218bb934b1f147b916b0224e857605)
and the [three-case worker](../tools/full-q6-observed/worker.py) on Kaggle CPU:
four threads, independent greedy cap256, composed native vision, no source or
BF16 forward and no repeated Q4 model inference. Ten native controls and 13 Python
tests pass; all 410 native vocabulary rows pass finite/greedy checks. An independent
small-output review recomputes token/text equality, EOS and edit distances. The
whole worker took 1,016.39 seconds including build, downloads and import; this is
not a comparative speed benchmark. The initial test-only convolution fixture
used physical rank instead of native rank; the corrected normal retry passes.
No model transfer or inference occurred in that failed attempt.

The separate `lighton4b-upstream-q6-physical-v1` profile retains 249 Q6_K matrices
and 474 F32 tensors. Import checks all 177 small source parameters against the
existing publisher Q4_K_M source, preserves 297 protected vision payloads,
inversely permutes 96 whole-row tensor layouts and retains 24 tiled-column output
projections. It performs no requantization or full-F32 model construction.
`CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL_Q6=1` requires the existing physical master;
Q4 repacking is rejected and the decoder cache records the precision profile.
The legacy Q4 profile and all defaults remain unchanged.

The retained quantized model is 4,797,156,960 bytes, SHA256
`f491ed0ae1e78d0c4a7c54655158f1fba502df8c56eff06da5287f79cb9a3272`.
It and raw native scores/layers stay remote. Only small reports and logs were
retrieved. No independent full-raw audit or all 249 source-matrix operator replay
was run. Existing source Q6 captures contain decoded responses without hidden
states/full-vocabulary rows, so exact text does not establish numerical/operator
or source-vision parity. These three previously observed cases do not establish
representative OCR quality or broader platform support. No alias, publication,
default, release or application pin changes follow.
