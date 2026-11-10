# Source-compatible Q6 tiled helper proof

One native GGML build links the actual gated runtime helper. Original form source4/image140 inputs and the original Q6 matrix are unchanged. Every legacy output must reproduce the previous native operator report; every candidate output must reproduce the previous upstream operator report, for all declared batch plans and threads1/4. Full source140, source4 and old native16 closures remain mandatory.

Eligible zero-Q6 controls exercise all three public buffer modes before the original17.2MB range. Both output routes preserve original quantized storage; no model forward, full F32 model or decoded qualification.1800-second worker, two build threads, cgroup-aware4GiB RAM and6GiB disk admission. This proves the helper's QKV arithmetic only; broader tensor-width controls and full decoded A/B remain required.
