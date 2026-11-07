# Bounded same-input physical text primitive diagnostic

This standalone diagnostic builds against two explicitly pinned ggml trees;
it is never linked or installed in CrispEmbed. Run only on the coordinated Kaggle
CPU worker, after authenticating source commits, complete weight files, captured
input hashes and compiler options. No generation or full F32 model is involved.

Configure twice with `-DGGML_SOURCE=<authenticated-tree>/ggml` for upstream
llama.cpp `6c01da6278ab6c69ed67d9124806b50d2b299ba7`, and the native gitlink
`3a27b6c665b191cb230be367db5c52d81dd3e2ac` itself. The independent builds use
AVX2, no AVX512/BLAS/CUDA/Vulkan, four CPU threads. Preserve both CMake caches,
binary/library hashes and CPU details. Exact runtime input paths are worker-owned.

Invocation: `physical-primitive OP TYPE ROWS TOKENS WEIGHTS INPUT OUTPUT`.
Width is fixed 2560; tokens are 1–16; rows are bounded to8192. `norm f32 1`
computes ggml RMSNorm epsilon1e-6 followed by direct-gamma multiplication.
`direct` uses the stored quantized matrix; `dequant` converts only this bounded
matrix to F32 before the original F32 mul_mat precision. Supported types are
f32, q4_k and q6_k. Inputs and outputs are exact-length LE raw arrays.

Run in this order:

1. Authenticate actual native first16 input and input_norm captures, physical
   layer0 direct-gamma/QKV/Z/a/b raw tensor spans and the transport map.
2. Run upstream and native ggml RMS with identical first4 prefix embeddings,
   then the identical first16 embeddings. Compare each to native scalar norm,
   reporting exact hashes, RMS/max error and input magnitudes separately.
3. Feed the **actual native input_norm** unchanged into both ggml builds, each
   with `direct` and `dequant`, for QKV/Z/a/b. Apply the recorded source-to-native
   whole-row permutation to outputs before comparing to actual native captures.
4. Repeat projections using upstream primitive norm output as a composed control;
   keep it separate from the same-input projection proof.

Physical row permutation is native row i = upstream row P(i), with
P(h)=16*(h%2)+h/2 and128 channels/head. QKV's first4096 Q/K rows are identity;
its last4096 V rows and all Z rows use P, while a/b use32-head P. Quantized
blocks remain unchanged. No conversion or tolerance may hide an earlier mismatch.

The source graph processes a four-token text prefix followed by a140-token image
chunk on the form. A16-token primitive is a same-input component control, **not**
a capture of actual upstream16-token batching. Report first4 and16 results
separately. Direct quantized dots may introduce Q8_K activation conversion;
source-vs-native backend differences and that arithmetic change must be isolated
before proposing any production fix. The firstscore difference alone identifies
neither a cache bug nor a particular operator.

`tools/kaggle/lighton4b_physical_prefix_primitives.py` prepares the operator-only
four-token reconstruction without actual activation captures. It authenticates
the full retained154-ID prompt against both text-file and original NPY hashes,
then uses IDs248045,846,198,248053. To authenticate those four embedding rows it
streams and verifies the **entire521472000-byte quantized embedding tensor**,
retains four rows and deletes the task-owned tensor. It also retrieves the
complete five small layer0 tensors; total weight transfer544675840 bytes. It
never downloads a full model. Source and tensor manifest digests are recorded.

The two builds use two compile workers and four operator threads sequentially;
preflight requires2GiB effective RAM and4GiB disk. Suggested provider budget is
35minutes with45minutes outer retention. This can fit a third isolated CPU
worker on resources, but scheduling remains explicitly coordinated; no execution
is implied. Actual trace comparison has priority once its immutable outputs are
available. Reconstructed scalar norm is labeled reconstruction, never actual
main-path evidence. The scalar norm and upstream ggml norm are fed separately
to both direct and dequantized projections, with no row-layout comparison needed
for the cross-backend same-physical outputs. Later actual-native comparison must
apply the recorded row permutation.
