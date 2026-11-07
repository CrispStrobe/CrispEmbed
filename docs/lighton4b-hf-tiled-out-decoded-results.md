# HF tiled-output candidate: complete observed decoded negative

The original-F16-proven tiled-output replacement does not qualify the native HF-scalar candidate. The form remains byte-identical to the prior candidate and fails the pinned llama.cpp CER floor; the table deteriorates versus the prior candidate and reaches the256-token cap without EOS. Arabic remains exact. These are observed development cases, not heldout results. Nothing is approved for shipping.

| Case | New tokens / EOS | New raw CER vs BF16 | New raw WER vs BF16 | Old HF candidate CER / WER | Pinned llama CER / WER |
| --- | --- | --- | --- | --- | --- |
| Form |182 / yes |217/656 (0.330793) |43/90 (0.477778) |217/656;43/90 |189/656;46/90 |
| Table |256 / no, capped |237/607 (0.390445) |70/106 (0.660377) |160/607;47/106 |315/607;88/106 |
| Arabic |6 / yes |0/12 |0/2 |0/12;0/2 |0/12;0/2 |

The primary denominator is the frozen genuine original BF16 transcription. Raw CER uses Unicode codepoints; raw WER uses whitespace words, with no normalization. This measures model agreement, not human-annotated OCR accuracy. The pinned llama table also caps without EOS; its larger edit counts do not excuse the candidate's missing termination. Form WER improves over llama while CER worsens, so neither an aggregate promotion nor a claim that every metric regresses is justified.

Kaggle CPU kernel143174221/version1 executed controller source `b2a3778e424559b7216e1b11f21c7b38b71b02d8` with the exact model-free-passed native binaries from `c89aaa4cb0c33d73d121b83b33d42d4bfcb20204`. The native bundle manifest is `1278cf36f67f1cc6098d465e6c602c4bc2ffad702c185eede4f9deee662a96d0`. Full controller elapsed time was7917.54s; form3221.31s, table4177.19s, Arabic208.47s. The90-minute form/table bounds completed the form that previously exceeded a45-minute budget. Cumulative completed-child maximum RSS was4,529,459,200bytes; this is not an isolated per-case peak. Actual CPU was four logical Intel Xeon2.20GHz CPUs, portable build.

The new4,041,906,080-byte candidate SHA256 is `cc6ce27eded435b8ac278229a8c6ff553f1654d865ff14a1b67c597fa0ca5254`. Complete report SHA256 is `63a586c07b0c2125e448e4f425bc9760ff9b5eb7c3123c7de0a60fde0a790361`. The packer independently hashed the complete model and all723 payloads:699 source payloads remain exact and24 original-F16-proven tiled Q4 GDN output payloads match the publisher. The248 importance entries and33-Q6 schedule remain guarded; there is no requantization. HF gamma/log-A semantics remain unchanged, physical flags are off, and only the matching HF-output activation bridge is enabled. Column order also changes projection reduction order; this is not an isolation of quantized weight bytes alone.

The old candidate's table first score and all70 retained stage hashes reproduce exactly on the current host/binary before packing. All genuine BF16 prepared-array hashes pass for each case. Valid, wrong-proof and gate-off metadata admission controls behave as declared. Independent small-artifact readback validates the source/bootstrap/report chain, all723 payload records, raw IDs/text hashes, independent EOS/cap and all three edit numerators against separately hash-pinned references. Actual large model and score bytes remain remote; a separate read-only consumer audits those bytes and matched causal stages without new inference.

Against the old HF candidate, form all182 IDs and raw text are exact; Arabic all6 IDs/text are exact. Table shares113 selected IDs and first differs at zero-based position113. Later score comparisons require different causal contexts and must be excluded. Retained common-prefill layer boundaries can localize recipe effects, but do not establish parity against an original BF16 model with the same quantized weights. No additional original suboperator stages were captured in this worker.

The table first score versus original unquantized BF16 has RMS error0.6896166873/max3.21875, source maximum magnitude23.0 and native20.375, cosine0.9365590, original-token probability0.5366444. Arabic RMS0.3379991333/max1.78125, magnitudes19.0/18.25, original-token probability0.8716350. These combine quantization and runtime effects, not same-weight parity. Form original BF16 vocabulary rows are unavailable in these mounts; only its authenticated decoded reference is used.

[All raw small outputs, complete payload receipts, source/bundle controls and independent review](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/complete-decode-v1/evidence.json) are preserved with byte hashes. Actual transcriptions are retained for [form](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/complete-decode-v1/lighton-hf-tiled-out-candidate-decode/simple_form/native/generated.txt), [table](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/complete-decode-v1/lighton-hf-tiled-out-candidate-decode/simple_table/native/generated.txt), and [Arabic](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/complete-decode-v1/lighton-hf-tiled-out-candidate-decode/arabic_printed_line/native/generated.txt). Large model/scores remain in Kaggle output; no independently guaranteed90-day retention is claimed. No large model was downloaded locally, and no model-hub upload, original forward, full F32 model, heldout read, release or default change occurred.

## Independent actual-byte and common-input readback

Kaggle artifact-only consumer143195841/version1 passed in184.83s at source `0d90fd0b2a24fc7990cbd8bc052c478bd28bf0f2`. Its sealed protocol requires the exact complete producer report above and C89 native binary manifest. It reads and independently hashes the actual complete4GB model twice and all723 named tensor payloads; all699 protected payloads match the separately authenticated old manifest, all24 replaced payloads match the original-F16-proven publisher hashes, and the474F32/216Q4/33Q6 census is exact. Every current complete vocabulary row is finite and matches its recorded hash and selected argmax. No model execution, weight quantization, compilation or large artifact copy occurs.

Prefill input embeddings and positions are byte-identical to the old native candidate for all three cases. The earliest available changed boundary is decoder layer0. The following errors compare the two different native recipe outputs, with the old HF candidate as reference; these are not original-BF16 same-weight parity results.

| Case / stage | Old / new RMS magnitude | Error RMS | Maximum absolute error |
| --- | --- | --- | --- |
| Form layer0 |0.133937 /0.134039 |0.00139551 |0.0233736 |
| Form layer31 |1.01674 /1.01771 |0.0443830 |2.75315 |
| Form logits |2.38894 /2.35542 |0.0879881 |0.46875 |
| Table layer0 |0.202175 /0.202302 |0.00273095 |0.0423252 |
| Table layer31 |0.947946 /0.947180 |0.0532514 |1.26601 |
| Table logits |1.77071 /1.79014 |0.162988 |1.0 |
| Arabic layer0 |0.112272 /0.112378 |0.00125827 |0.0225974 |
| Arabic layer31 |0.965969 /0.966069 |0.0489850 |0.866588 |
| Arabic logits |2.04972 /2.03887 |0.0749170 |0.5625 |

All32 layer boundaries, final norm, source/candidate maximum magnitudes, bit-pattern and numerical inequality counts are retained in the raw receipt. Greedy comparisons use all182 form positions and all6 Arabic positions because their prefixes remain identical. Table compares only positions0–113 inclusive: selection113 first differs (old303/new430), on a common causal input. At that position error RMS is0.0522773/max0.265625 with old/new maximum magnitude22.125; the subsequent142 candidate rows are excluded from numerical comparison because their prefixes differ. This preserves the decoded negative despite small early boundary errors and unchanged form text.

[The complete readback receipt, tests and logs](../tests/results/qwen35-ocr/lightonocr3-4b-hf-tiled-out/causal-readback-v1/evidence.json) preserve report SHA256 `cb7a0a3e9a5e9a40c24c9e3ea2e9a618c4bfb0b2a973edbf1bd91888a54f09f5`. Historical old controller sources E822/B978, current B2 controller and C89 native bundle remain explicitly distinct. The current worker's old-model platform admission controls do not turn historical captures into a same-worker two-arm experiment. No missing original operator stages or publisher float-checkpoint identity are inferred.
