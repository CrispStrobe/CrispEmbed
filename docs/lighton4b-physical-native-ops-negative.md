# Physical native-ops: actual negative and main-path closure

The default-off physical native-ops alternative **does not improve the measured
form firstscore**. Component equality did not compose into closer full-model
scores; do not promote this mode or the current Q4 model. All shipping fences,
aliases and defaults stay unchanged.

The bounded [three-arm smoke](../tests/results/qwen35-ocr/lightonocr3-4b-physical-native-ops-smoke/physical-native-ops-smoke.json)
uses integrated source `285ab5a4e9d669746a3664ec7804783ca06cc6dd`, frozen
physical candidate `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`,
one regenerated upstream image-feature capture, four threads and LLAMAFILE ON.
Fresh source `ac34f72ca` reference, new OFF and new ON each independently generate
at most two tokens. The reference and OFF match **all38 stage hashes and the
firstscore bytes exactly**. Fresh upstream firstscore hash matches the retained
full baseline, closing reference drift.

| Arm | Firstscore RMS vs pinned upstream | Max absolute error | Process seconds | Observed peak RSS bytes |
|---|---:|---:|---:|---:|
| Fresh ac34 reference | 0.1023485054 | 0.5765159130 | 137.50 | 3194314752 |
| New OFF | 0.1023485054 | 0.5765159130 | 137.75 | 3204673536 |
| New ON | **0.1296630878** | **0.7529435158** | 21.63 | 2459381760 |

All arms emit IDs27,17251, text `<figure`, cap2/no EOS. This is a partial prefix,
not a decoded-quality pass. Timing is ordered and includes model loading and
prefill; it is not a balanced throughput benchmark. The measured speed difference
cannot waive the worse score result. Raw smoke report SHA256:
`1694d990aa2b23c17685025134b0c565aff844369ef823a46bfc6a8933a219a1`.
Independent review rehashes the three native firstscore vectors, verifies argmax
and38 OFF/reference hashes; FP64 RMS recomputation differs by at most one ULP.
No model numerical budget changed.

A separate [read-only actual trace closure](../tests/results/qwen35-ocr/lightonocr3-4b-physical-actual-trace/actual-trace-closure.json)
(source `9918eae569bd8003a236b08ab1c44a30db83bab7`) binds actual full-baseline
report `712d4c86…a61e2` and matched-profile primitive report `79dff60f…85f398`.
It independently streams all664 per-case artifact hashes (584443630 bytes) in
provider input, with no model/build or local raw-array download. Actual first4
input and scalar RMS are byte-identical to the authenticated reconstruction;
actual first4 QKV/Z/a/b are byte-identical to corresponding dequantized source
primitives after exact row mapping. Native16-token versus primitive4-token
batching produces no observed difference for these outputs. First16 main layer0
output is exactly reconstructed from captured attention residual plus down.
The actual-trace report SHA256 is
`9b4d53bde6a79ac4fee7188b2b99d973b9908f33b7704c52579c3b8e96f1b0fb`.

Thus the prefix reconstruction is closed to the real native path. Later and
composed arithmetic still needs attribution. Pinned upstream uses fused GDN and
F16 K/V defaults; native recurrence, gated norm, attention/cache/softmax and head
must be isolated on identical captured inputs. Controlled quant-only/RMS-only
and layers/head ablations remain gated preparations, not quality claims. Actual
upstream internal graph intermediates have not been captured by this closure.
