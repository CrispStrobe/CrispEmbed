# Same-feature physical arithmetic ablations: no quality promotion

The [raw six-arm report](../tests/results/qwen35-ocr/lightonocr3-4b-physical-prefix-ablations/native-ops-ablation.json)
(source `7ccb0c46805239aa9fef874809387a82f2eadda7`, SHA256
`82377eb782a8ad7df2028a3e0a7f2c83f4bf7e8aa35494feca9c7de122a69c95`)
uses the same authenticated actual source features/logits, physical candidate843e,
matched LLAMAFILE ON/four threads, and independent cap2 greedy prefixes. OFF and
all-mode each exactly reproduce38 golden stage hashes before ablations execute.
No upstream or original model is rerun; no weight quantization or recipe changes.

| Mode | Firstscore RMS | Max absolute error | Ordered process seconds |
|---|---:|---:|---:|
| OFF | **0.1023485054** | 0.5765159130 | 144.10 |
| All | 0.1296630878 | 0.7529435158 | 26.18 |
| Quantized layer+head projections | 0.1287035089 | 0.7737276554 | 25.98 |
| RMS only | 0.1023507453 | 0.5765359402 | 143.71 |
| Layer projections only | 0.1269996249 | 0.7347669601 | 31.68 |
| Head only | 0.1039376872 | **0.5721991062** | 157.19 |

None improves firstscore RMS. Layer-projection arithmetic accounts for most of
the measured regression; RMS and head alone are small effects here. A slightly
smaller head-only maximum does not establish improvement. Every arm emits only
IDs27,17251, text `<figure`, cap2/no EOS. This is neither complete decoded-quality
proof nor a basis to select a shipping recipe. Timings include loading/prefill
and use one fixed order, so they are diagnostic only.

Independent review rehashes all six firstscore vectors and verifies argmax;
FP64 RMS recomputation differs by at most one ULP, with exact maximum errors.
No model numerical budget or gate changed. Master stays default-off and the public
loader/alias fences remain in force. Next is source-fed attention/cache/softmax
attribution on actual captured inputs, accounting for upstream F16 K/V storage
and implicit mixed-type matmul conversions. The qualified source-fed GDN result
and actual native prefix closure remain separate component evidence.
