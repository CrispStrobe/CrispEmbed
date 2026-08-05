# CrispEmbed v0.17.5

**If you are on v0.17.4, upgrade.** That release shipped the wrong ggml — see
below — and this one corrects it. Alongside that: a tokenizer correctness fix
for LaBSE-class models, quantization that finally gives BERT attention weights
importance data, a 4.8× faster one-shot CLI, and four build fixes that between
them cost two releases.

## Correction: v0.17.4 shipped ggml v0.10.2, not v0.17.0

v0.17.3 moved the ggml pin to `sync/upstream-v0.17` so that an application
bundling CrispEmbed next to CrispASR gets **one** ggml rather than two sets of
libraries fighting over the same `@rpath` install names.

An unrelated commit — `357dee53`, *"refactor(unicode): single generated
category table"* — carried a stray submodule change that reverted the pin from
`c76aaacb` back to `0714117d` (v0.10.2), silently undoing two deliberate
commits. It landed about an hour before v0.17.4 was tagged.

So **v0.17.3 ships `libggml*.so.0.17.0`, v0.17.4 ships `libggml*.so.0.10.2`,
and v0.17.5 ships 0.17.x again.** If you bundle both libraries, v0.17.4
reinstates exactly the soname collision the pin exists to prevent.

Found by inspecting the published archive rather than the source tree — the
version is in the filename, and the CUDA tarball had `libcrispembed.so.0.17.4`
sitting next to `libggml-base.so.0.10.2`.

## Tokenizer: LaBSE-class WordPiece was converted and loaded as SentencePiece

`is_sentencepiece` kept a `vocab > 100000` heuristic for WordPiece, so
`sentence-transformers/LaBSE` (501k WordPiece) converted as
SentencePiece-with-no-scores, and the runtime routed even an *explicitly*
declared WordPiece GGUF with >100k tokens into the SPM tokenizer.

On the 20-case HuggingFace token-id battery that scored **0/20**: wrapped with
`bos=0`/`eos=2` instead of `[CLS]=101`/`[SEP]=102`, a literal `▁` vocab token
for every space, and UNK for tabs and newlines. No shipped registry model was
affected — the conversion path was audited end to end and found broken before
one shipped. Fixes are self-describing, with absent-key preserving historical
behaviour.

## Quantization: BERT attention q/k/v were quantized with no importance data

The BERT-family load path pre-merges per-layer `attn.{q,k,v}.weight` into one
F32 tensor and never named it, so the imatrix collector filed that matmul's
statistics under ggml's auto `leaf_N` — which matches nothing at quantize time.
Every BERT-family attention q/k/v tensor was therefore quantized **without**
importance: multilingual-e5-small covered only 36 of 74 tensors.

The merged tensor is now named, and the quantizer aliases
`attn.{q,k,v}.weight` to it. The arctic sub-Q8 registry aliases are re-pinned
to re-quants built with the corrected imatrix.

## Faster

- **One-shot CLI init is 4.8× faster.** The cost was a 683 MB Metal
  pipeline-cache archive that ggml opens at startup; it is append-only across
  every binary that ever ran on the machine and bought nothing measurable —
  the first encode was marginally *slower* with it open, because macOS keeps
  its own shader cache underneath. Now capped
  (`CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB`, default 64) and adopted across
  every GPU lane, not just embeddings.
- **The embed one-shot defaults to `min(4, cores)` threads.** `-t1` lost to
  `-t4` on every model and architecture tested (2–3× on the 300M class).
  Embeddings are byte-identical across thread counts; an explicit `-t` still
  wins.
- **DeepSeek-OCR-2 decoding is 1.40× faster and byte-identical**, via a
  persistent decode-step graph.

## OCR

- **DeepSeek-OCR-2 no longer spirals on repeats.** The reference contract
  generates with `no_repeat_ngram_size=20` and the lane had no equivalent, so
  2 of 5 gold pages ran to the 1024-token cap repeating one phrase. The guard
  now applies at the shared argmax site (`DS2_NO_REPEAT_NGRAM=0` restores the
  old behaviour).
- **Dynamic-crop port** for DeepSeek-OCR-2, opt-in via `DS2_CROP_MODE=1`.
- **SmolDocling** runs SigLIP on the GPU backend with split residency.

## Build fixes

Four of these, all caught by dry-running the release workflow rather than by
publishing and finding out:

- **Linux CUDA archives link again.** ggml 0.17 calls the CUDA driver API
  (`cuGetErrorString`, `cuMemCreate`), which 0.10.2 did not. The toolkit ships
  the driver stub as `libcuda.so` while its SONAME — and so the `DT_NEEDED`
  that lands in `libggml-cuda.so` — is `libcuda.so.1`, so the linker looked for
  a filename that did not exist. The stub is now staged under both names with
  `-Wl,-rpath-link` (link-time only; the real driver still provides
  `libcuda.so.1` at runtime).
- **Windows builds again.** The Metal pipeline-cache header guarded only its
  function body, leaving `<dirent.h>` and POSIX helpers to compile on MSVC.
- **`GGML_METAL_EMBED_LIBRARY` follows `GGML_METAL`** on every configure.
- The `clean_exit` CI guard now carries the fix in its own error message, and
  the pattern is documented in `docs/contributing.md` — it blocked four
  separate pushes in one day.

Full technical detail in `HISTORY.md` and `LEARNINGS.md`.
