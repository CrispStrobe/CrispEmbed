# LFM2 Encoder paired CPU/CUDA dense API performance

One genuine GPU job measures the previously audited official F16 encoder at
runtime898af2c2, harness13956033, modelbb26e78c and original fixturee3599a94.
All reference files must match the prior CUDA audit's exact hashes. No new
checkpoint, precision, default or independent task-quality claim is introduced.

The isolated worker uses the real Python/C ABI dense encode with two threads.
Each of15 unchanged inputs checks exact IDs and original raw CLS against the
existing cosine0.999/relative0.02 gates, then dense/normalized-CLS consistency.
Two warmups precede seven timed calls; returned vectors must remain bit-exact
within the block. Fresh CPU/CUDA/CUDA/CPU processes balance first/last positions.
Initialization, reference/ID checks and warmups are excluded from encode timing;
Python FFI/output copying and completed native computation are included.

Separate ordinary scheduler probes require CUDA matmul assignment for the GPU
arm and none for CPU. This establishes actual GPU use, not full GPU residency
for every shape. Process RSS is a high-water mark including model initialization
and quality probes; it is neither incremental runtime memory nor VRAM. Retain
all sample timings, block/library/source hashes and failures. Report paired
per-case medians and variation; one device/host is not general GPU speed evidence.

Check active sessions and available quota before using one authorized GPU slot.
Use the standard account-neutral kpush wrapper and private credential setup;
verify pulled script bytes. Do not repush an active kernel or retry GPU draws.
Unsupported legacy GPU/toolkit pairs terminate inconclusive before downloads.
Use the standard harness toolchain/CUDA/cache/job-limit/heartbeat helpers.
Keep all weights, references, build and compiler cache remote; only small JSONs
and selected logs are outputs. Script version is lfm2-encoder-cuda-performance-v1.
