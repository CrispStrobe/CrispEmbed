# Josev original-first CPU and CUDA head audit

This is a component audit of the original Josev joint head. It does not load or
qualify the native Qwen3.5-2B backbone, expose a model alias, or change runtime
defaults. See [the source contract](josev2b.md),
[head conversion](josev2b-head-conversion.md) and
[artifact preservation](https://github.com/CrispStrobe/CrispEmbed/blob/6b3052fde19a959bbba4e23ce2dbafe7d4d81048/docs/josev2b-head-artifact.md).

The private GPU submission uses `tools/kaggle/josev2b-head/bootstrap.py`. The
operator attaches the already frozen original version5 output as a kernel
source; access references remain private. Because attachment names alone do not
establish immutable identity, the worker selects exactly one `reference.json`
with SHA256 `51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`,
then checks every declared input and all251 original arrays before copying them.
It never invokes the original model or generates replacement gold. Missing,
ambiguous, symlinked, truncated or changed source files reject the run.

Actual supported GPU hardware and available memory must be established before
packages, source/model downloads or builds. This is a genuine CPU/CUDA comparison
on GPU hardware, with no CPU fallback or automatic resource retry. The isolated
Python3.13 tooling consists of four exact archived wheels authenticated by size
and SHA256, installed without dependency resolution or an index. The shared
Kaggle build harness configures CUDA for the observed architecture and caps
compilation at two workers. Build/cache/checkpoint scratch remains outside the
kernel output. No build, model or array batch runs on the shared development host.

The worker preserves original arrays, promotes all122 head tensors losslessly,
checks GGUF readback, and exports byte-exact raw hidden/lexical inputs. It then
uses the actual `crispembed_diff::Ref` harness and independently computed NumPy
errors for all123 ordered head stages on each of CPU and CUDA0. The `0.002`
absolute and `0.99999` minimum-row cosine budgets remain fixed. Ordinary,
observed, repeated and post-failure recovery paths must agree. Plain-library
symbol/string checks keep observer code out of production artifacts. All four
complete typed responses are compared with the pinned original output function.
Numeric or decoded failures remain failures even when every attempted comparison
completed.

Keep the full original/component/native output remotely. Download only the
bootstrap, review and diagnostic logs required for independent review. Record
actual source, GPU, tooling, output hashes, earliest divergence, magnitudes,
final typed responses and limitations before integrating any native path.
