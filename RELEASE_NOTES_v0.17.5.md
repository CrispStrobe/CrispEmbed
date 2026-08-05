# CrispEmbed v0.17.5

**Everyone on v0.17.4 should upgrade** — it shipped the wrong ggml, and this
corrects it. Also here: a segfault in the Rust OCR bindings and two more
instances of the same defect, a large DeepSeek-OCR-2 accuracy win now on by
default, better quantization for every BERT-family model, and a 4.8× faster
one-shot CLI.

## Fixed: segfault in the Rust OCR bindings (and two more like it)

OCR through the Rust bindings crashed *after* every region had been recognised
successfully — on every engine and every model, which made it look like a model
problem. It was a struct-layout mismatch across the FFI boundary:

| | fields | size |
|---|---|---|
| C `crispembed_ocr_result` | …`text_len`, `orientation_corrected`, `orientation_angle`, `orientation_confidence` | 48 bytes |
| Rust `CrispembedOcrResult` | …`text_len`, `orientation_corrected` | 40 bytes |

Consumers walk the region array with `ptr.offset(i)`, striding 40 bytes through
a 48-byte array. Each element drifts 8 bytes further out of alignment until
`text` is a garbage pointer and `CStr::from_ptr` faults. The crash is in the
read-back, not the recognition.

A `#[repr(C)]` mirror is a **layout** contract, not a field list. Auditing the
rest of the bindings against `crispembed.h` found the same defect twice more:

- **`CrispembedOcrStage` was missing eight fields**, and one of them
  (`model_c`) sits at position 5 — so every field after it was at the wrong
  offset. Rust builds a `Vec` of these and hands `*const` to C, which reads 22
  fields per element: C took `cleanup_enabled`'s bytes as the `model_c`
  pointer and dereferenced it. Worse than the result mirror, and on the input
  side.
- **`CrispembedHparams`** was missing `n_experts` / `n_experts_per_tok`
  (32 bytes against 40). `crispembed_get_hparams` returns a pointer, so nothing
  was over-written, but the contract was broken all the same.

Every field added is documented `0 = default` in the header, so behaviour is
unchanged. All three mirrors were then re-checked by field name and order:
10/10, 10/10, 22/22 exact.

## Correction: v0.17.4 shipped ggml v0.10.2, not v0.17.0

v0.17.3 moved the ggml pin to `sync/upstream-v0.17` so an application bundling
CrispEmbed next to CrispASR gets **one** ggml instead of two sets of libraries
fighting over the same `@rpath` install names.

An unrelated commit — `357dee53`, *"refactor(unicode): single generated
category table"* — carried a stray submodule change that reverted the pin to
v0.10.2, silently undoing two deliberate commits, about an hour before v0.17.4
was tagged.

So **v0.17.3 ships `libggml*.so.0.17.0`, v0.17.4 ships `libggml*.so.0.10.2`,
and v0.17.5 ships 0.17.x again.** If you bundle both libraries, v0.17.4
reinstates exactly the collision the pin exists to prevent. Found by inspecting
the published archive rather than the source tree — the version is in the
filename.

## DeepSeek-OCR-2: dynamic crop is now the default

`DS2_CROP_MODE` defaults ON, matching the reference contract. On the CC0 corpus,
raw CER mean:

| | before | after |
|---|---|---|
| Metal | 0.657 | **0.236** |
| CPU | 0.279 | **0.185** |

The CPU figure beats the reference pipeline's own 0.187. It also fixes the
Metal German page that previously ran to the 1024-token cap. `DS2_CROP_MODE=0`
restores the single-view path.

Two apparent regressions were diagnosed as formatting-only decode drift rather
than vision bugs: one bold-vs-plain near-tie followed by markdown-list
self-conditioning (alnum-content CER flat, and CPU *improves*), and four
inserted colons with the content byte-equal.

Also on this lane:

- **No-repeat-ngram decode guard.** The reference generates with
  `no_repeat_ngram_size=20` and the lane had no equivalent, so 2 of 5 gold
  pages spiralled into the 1024-token cap repeating one phrase. Applied at the
  shared argmax site so both decode arms stay comparable
  (`DS2_NO_REPEAT_NGRAM=0` restores the old behaviour).
- **Persistent decode-step graph: 1.40×, byte-identical.**
- **`DS2_KV_F16`** measured against the F32-KV baseline — KV allocation halves
  (165 → 82.5 MB), decode timing neutral, byte identity holds on most fixtures
  but not all, so it stays opt-in.

## Quantization: BERT attention q/k/v had no importance data

The BERT-family load path pre-merges per-layer `attn.{q,k,v}.weight` into one
F32 tensor and never named it, so the imatrix collector filed that matmul's
statistics under ggml's auto `leaf_N` — matching nothing at quantize time.
Every BERT-family attention q/k/v tensor was quantized **without** importance;
multilingual-e5-small covered only 36 of 74 tensors.

The merged tensor is now named and the quantizer aliases `attn.{q,k,v}.weight`
to it. Both arctic sub-Q8 registry aliases are re-pinned to re-quants built
with the corrected imatrix — measured against full-precision gold on 65 texts:

| | before | after |
|---|---|---|
| `q4_k` + imatrix | 0.9614 mean / 0.9481 min | **0.9937 / 0.9910** |
| `iq4_xs` | 0.9757 / 0.9667 | **0.9867 / 0.9817** |

## Tokenizer: LaBSE-class WordPiece was treated as SentencePiece

`is_sentencepiece` kept a `vocab > 100000` heuristic for WordPiece, so
`sentence-transformers/LaBSE` (501k WordPiece) converted as
SentencePiece-with-no-scores, and the runtime routed even an *explicitly*
declared WordPiece GGUF with >100k tokens into the SPM tokenizer. On the
20-case HuggingFace token-id battery that scored **0/20** — `bos`/`eos` instead
of `[CLS]`/`[SEP]`, a literal `▁` token for every space, UNK for tabs and
newlines. No shipped registry model was affected; the path was audited and
found broken before one shipped.

## Faster

- **One-shot CLI init is 4.8× faster, now on every GPU lane.** The cost was a
  683 MB Metal pipeline-cache archive that ggml opens at startup — append-only
  across every binary that ever ran on the machine, and worth nothing: the
  first encode was marginally *slower* with it open, because macOS keeps its
  own shader cache underneath. Capped via
  `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB` (default 64) and hoisted into
  `crispasr_init_gpu_backend()`, so the ~40 OCR/VLM/SR/NER/denoise engines get
  it too rather than embeddings alone.
- **The embed one-shot defaults to `min(4, cores)` threads.** `-t1` lost to
  `-t4` on every model and architecture tested (2–3× on the 300M class).
  Embeddings are byte-identical across thread counts; an explicit `-t` wins.
- **SmolDocling runs SigLIP on the GPU backend** via predicate-routed split
  residency ported from CrispASR — 88 MiB of vision weights in chunked
  allocations, honouring `--gpu-backend`.

## Build and packaging

Four fixes, all caught by dry-running the release workflow rather than by
publishing and finding out:

- **Linux CUDA archives link again.** ggml 0.17 calls the CUDA driver API
  (`cuGetErrorString`, `cuMemCreate`), which 0.10.2 did not. The toolkit ships
  the driver stub as `libcuda.so` while its SONAME — and so the `DT_NEEDED` in
  `libggml-cuda.so` — is `libcuda.so.1`, so the linker looked for a filename
  that did not exist. The stub is now staged under both names with
  `-Wl,-rpath-link` (link-time only; the real driver still provides
  `libcuda.so.1` at runtime). v0.17.3 shipped with **no** Linux CUDA archive
  because of this.
- **Windows builds again** — the Metal pipeline-cache header guarded only its
  function body, leaving `<dirent.h>` and POSIX helpers to compile on MSVC.
- **`GGML_METAL_EMBED_LIBRARY` follows `GGML_METAL`** on every configure.
- The `clean_exit` CI guard now carries the fix in its own error message, and
  the pattern is documented in `docs/contributing.md` — it blocked four
  separate pushes in one day.

Full technical detail in `HISTORY.md` and `LEARNINGS.md`.
