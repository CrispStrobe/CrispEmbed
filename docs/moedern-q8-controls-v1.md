# Hosted MoEdern Q8 model-free controls v1

This caller executes the existing authored fixtures from frozen
[1f357313](https://github.com/CrispStrobe/CrispEmbed/tree/1f35731372cc5c9a60cd2431d58353f4742e25fb).
It changes no runtime, root/standalone CMake, converter, old source guard or model.
The [protocol](moedern-q8-controls-v1.json) closes those source objects plus the
new [driver](../tools/ci-heavy/moedern_q8_controls_v1.py),
[workflow](../.github/workflows/moedern-q8-controls-v1.yml) and stdlib controls.
The existing1f source guard executes independently before any compilation.

Hosted Linux CPU only, two build/compute threads, minimum4GiB available RAM and
8GiB disk,60-minute job limit. These are separate model-free bounds; no old
14GiB/24GiB model guard is changed. There are no model, tokenizer, corpus or
original-reference downloads, optional package installs or neural-model forwards.
Git checkout retrieves source only. Build the existing `moedern-q8-controls` and
`moedern-q8-quantize` targets with `--parallel 2`; require actual LLAMAFILE ON,
NATIVE/BLAS/device backends OFF and contraction OFF for all six MoE library TUs.
Capture compiler identity, complete compile commands/cache, child exit/time/RSS,
logs, binary hashes, plain-symbol exclusion and every authored file manifest.
Linux wait4 RSS is per-child peak, not a concurrent process-tree memory total.

The existing C++ executable covers three typed profile containers with20/20/52
Q8 tensor declarations, nine storage mutations, unset/empty/0 dependency gates,
concurrent Q4/Q8 rejection and six actual direct/expert graphs across128/640/1024
input widths. Its checks use exceptions and remain active in Release. Authored
weights/activations have exact block encodings; their0.002 graph comparison is
component evidence, not a new model tolerance.

The separate actual C-quantizer fixture has **two authored Q8 tensors and seven
critical F32 tensors**. Verify standard header7,2D/3D axes, byte-exact critical
payloads and independent half-scale/integer readback. Do not describe this as a
full20/20/52-model quantization: no actual model is loaded or converted. Store
all structured negatives and command logs even on an early rejection. Small
reports and separate authored binaries/fixtures remain remote for90 days.

A successful run establishes only compilation and authored profile/operator/
storage support. The full-corpus1.5M vector failures and27M/115M router-tie
rejections remain; source data, originals,0.002/0.99999/MaxSim and retrieval gates
are unchanged. Real Q8/Q4 qualification still requires diagnosed original/native
intermediates, all5,483 token/vector rows, all300 rankings and public consumers.
No publication, alias, default change, runtime release or app pin is authorized.
