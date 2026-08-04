# CrispEmbed v0.17.2

**Two embedding-quality fixes that affect models already published** — one for
every GTE v1.5 model, one for every Qwen-family embedder — plus four new
models and a Linux CUDA archive that no longer needs a CUDA toolkit
installed (#42).

Both quality fixes are in the runtime, so **upgrading the binary is enough**:
no reconversion, no re-download of GGUFs.

## Qwen-family embedders were losing newlines at tokenization

`core_bpe::tokenize_simple` split text on whitespace and rejoined the runs with
a single space, so `"a\n\nb"` and `"a b"` produced identical token ids —
newlines and indentation were silently deleted. That degraded **every shipped
Qwen-family embedder**, and it hit hardest exactly where it matters: the
`"Instruct: …\nQuery: "` prompt that every retrieval query uses.

Measured against the f32 HF reference on F2LLM-v2-160M: cosine **0.9803** on a
code snippet and **0.9907** on the family's own instruction prompt, while
newline-free text was unaffected at 1.000000. Confirmed as a tokenizer defect
rather than numerics by feeding HF the whitespace-collapsed text, which
reproduced our pre-normalization magnitudes exactly.

`core_bpe::qwen_pretokenize` now implements the pre-tokenizer regex Qwen2/Qwen3
`tokenizer.json` actually declares — newline runs, punctuation-then-newline,
and one token per digit, none of which the old code had any notion of.
`CRISPEMBED_BPE_LEGACY_WHITESPACE=1` restores the previous behaviour; other
`tokenize_simple` callers (`lfm2_embed`, the OCR engines) are untouched.

## GTE v1.5 models were dropping a per-layer bias

The fused gated-FFN branch never applied `ffn.fc2.bias`. ModernBERT's `mlp.Wo`
has no bias, so this was invisible there — but GTE v1.5's
`GteGatedMLP.down_proj` is `nn.Linear(..., bias=True)`, and the converter has
always written the tensor. **Every GTE v1.5 GGUF has been running with a whole
per-layer bias vector silently discarded.**

Measured on the already-published `gte-base-en-v1.5-q8_0` GGUF against HF,
14 sentences:

| | cos_min | cos_mean |
|---|---|---|
| before | 0.985049 | 0.989352 |
| after | **0.999575** | **0.999669** |

The bias was already inside the shipped files, so they are repaired in place —
**no reconversion and no re-download required**; upgrade the binary and existing
GGUFs improve.

The gated-FFN activation is also self-describing now: the converter writes
`bert.ffn_act` from `config.hidden_act`, and the runtime maps
`silu → ggml_swiglu`, `gelu → ggml_geglu_erf`, `gelu_pytorch_tanh → ggml_geglu`.
HF's `ACT2FN["gelu"]` is the exact erf GELU rather than the tanh approximation,
so GTE v1.5 had been using the wrong flavour. An absent key keeps the historical
per-architecture default, so published GGUFs are byte-for-byte unaffected.

## New models

**F2LLM-v2 family** (`codefuse-ai`, Qwen3Model, Apache-2.0) — all four small
sizes now reachable by name: **80M / 160M / 330M / 0.6B**. Last-token pooling,
L2-normalized, instruction prompt on queries only. The 160M is the strongest
sub-200M German embedder on MTEB(deu, v1). Parity vs the f32 HF reference over
14 mixed German/English/code texts, reporting cosine *and* the
`|mine|/|ref|` pre-normalization magnitudes (cosine is scale-blind), plus a
5-query/10-doc German retrieval check:

| model | f16 | q8_0 | German top-1 |
|---|---|---|---|
| F2LLM-v2-80M | 1.000000 | — | 5/5 |
| F2LLM-v2-160M | 1.000000 | 0.999437 | 5/5 |
| F2LLM-v2-0.6B | 1.000000 | 0.990858 | 5/5 |

**snowflake-arctic-embed-m-v2.0**

305M, Apache-2.0, GTE v1.5 backbone, 768-d CLS, 8192 context. Per-stage parity
against HF is `cos_min 1.000000` at embeddings, all 12 blocks and encoder
output. End-to-end over 14 mixed DE/EN sentences: f16 `cos 1.000000`, q8_0
`cos_min 0.999718`, q4_k `cos_min 0.954179`; German retrieval sanity is 5/5
top-1 agreement with HF at every quant. Registry default is Q8_0 (no imatrix
calibrated yet).

Arctic query prefixes are now wired for the previously shipped v1 and l-v2
entries too: v2.0 uses `"query: "`, v1 the BGE-style instruction, documents take
no prefix in either generation.

## New: a Linux CUDA archive that needs only the driver (#42)

`crispembed-linux-x86_64-cuda.tar.gz` ships no CUDA runtime, and the build
described its requirement as *"a matching CUDA driver (12.x) on the host"*.
That was wrong: a driver provides `libcuda.so.1`, while `libcudart.so.12` and
`libcublas.so.12` come from the CUDA **toolkit**.

```
libggml.so.0      -> libggml-cuda.so.0   (bundled, hard DT_NEEDED)
libggml-cuda.so.0 -> libcudart.so.12     <-- toolkit, not bundled
                  -> libcublas.so.12     <-- toolkit, not bundled
                  -> libcuda.so.1        <-- driver, correctly external
```

Because `libggml.so` hard-links `libggml-cuda.so`, a machine with a driver but
no toolkit fails in the dynamic loader at startup — **exit code 127, no output,
and no fall back to the CPU backend**, since the process never reaches
`main()`. The Windows CUDA zip has always bundled `cudart`/`cublas`, which is
the only reason this never affected Windows.

Rather than change what an existing download URL resolves to, there are now two:

| archive | size | host must provide |
|---------|------|-------------------|
| `crispembed-linux-x86_64-cuda.tar.gz` | 138 MiB | NVIDIA driver **and** CUDA 12.x toolkit runtime |
| `crispembed-linux-x86_64-cuda-bundled.tar.gz` | 705 MiB | NVIDIA driver only |

The first is unchanged in name and contents — existing pins keep working, and
its requirement is now documented accurately. Take the second if you are not
certain the toolkit is installed. `libcuda.so.1` is deliberately not bundled in
either: it is the kernel driver library and must match the host's driver. The
bundled archive carries `NVIDIA-EULA.txt` and `NVIDIA-NOTICE.txt` — those
libraries are NVIDIA's, redistributed under the CUDA Toolkit EULA.

## Packaging is now checked per-archive

`scripts/check-bundled-deps.py` gained `--allow`, and each Linux archive is
verified at packaging time against its own contract: the slim CUDA archive may
require the toolkit runtime from the host, the bundled one only the driver, and
the CPU archives nothing beyond a base glibc system. An archive that would fail
to load on a clean machine now fails the release instead of shipping.

## Known limitation (unchanged)

The Linux archives are built on Ubuntu 24.04 and require **glibc ≥ 2.38** /
**GLIBCXX ≥ 3.4.32** — Ubuntu 24.04+, Debian 13+, RHEL/EL 9+ and current Arch,
but not Ubuntu 22.04 or Debian 12. The bundled NVIDIA libraries need only glibc
2.17–2.27, so they are not what sets this floor; our own build is. Lowering it
needs a manylinux-container build and is tracked separately.

Full technical detail in `HISTORY.md` and `LEARNINGS.md`.
