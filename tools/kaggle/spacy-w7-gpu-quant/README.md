# W7 actual CUDA quantization and prepared-graph diagnostic

This script uses an actual Kaggle GPU, with early `nvidia-smi`/Torch capability
checks before model downloads. Unsupported P100/Torch draws exit inconclusive;
never repush in a loop to fish for another device. It compiles no CPU/native
project, installs no Torch replacement, and attaches no datasets or secrets.
The shared CrispASR progress helper is cloned at its recorded exact source.

All model weights, uint64 hashing/buckets, storage quantization/dequantization,
affine/maxout/normalization/residual/classifier and prepared parser-state math
execute on CUDA tensors. CPU work is limited to transport, format serialization
and the unchanged frozen-reference comparator. Tensor-device assertions and
actual hardware/software/peak allocation are retained. Raw GPU stage arrays and
GPU-produced GGUFs stay in versioned kernel outputs; retrieve only needed reports.

The accepted immutable F32 model and16-case W1 development oracle are hash-pinned.
The native/W6/new96-case held-out splits are not downloaded or executed. F16,
Q8_0 and protected mixed Q4_0 follow the recorded storage recipes; generated
bytes are compared against the already rejected native CPU candidates. Every
profile uses the unchanged2e-4/2e-5 element budgets and classifier/legal-action
checks. CUDA prepared results do not qualify a native GPU text backend,
quantized model, speed improvement, Mac grammar or app activation. Accepted
Linux F32 remains unchanged.

Render the account and exact source placeholders outside Git before one
authorized submission. Use only the operator's required token file and one
account. Increment the version/run ID for any later justified source repair,
save the old terminal logs first, and do not attach cache/tooling datasets.
