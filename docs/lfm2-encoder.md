# LFM2.5-Encoder-230M

CrispEmbed runs the official
[LiquidAI GGUFs](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M-GGUF)
directly, using its bidirectional LFM2 backend. The 14-layer backbone combines
ShortConv and grouped-query attention, with 1024-dimensional final features
and a tied 65536-entry masked-LM head. No conversion is needed for these files.

## Usage

```sh
crispembed -m LFM2.5-Encoder-230M-F16.gguf --fill-mask \
  "The capital of France is [MASK]."
crispembed -m LFM2.5-Encoder-230M-F16.gguf --tokens-raw --json "hello world"
crispembed -m LFM2.5-Encoder-230M-F16.gguf --capabilities --json
```

Registry names are `lfm2-encoder-230m` (F16), `lfm2-encoder-230m-f16`,
`lfm2-encoder-230m-q8`, and `lfm2-encoder-230m-q4`. Downloads use the existing
LFM license acknowledgement and SHA-256 verification. Query/document prefixes
are absent because this checkpoint is a general-purpose masked encoder,
not the retrieval-trained LFM2.5-Embedding model.

```python
from crispembed import CrispEmbed

model = CrispEmbed("LFM2.5-Encoder-230M-F16.gguf")
ids, hidden = model.encode_tokens("hello world", normalize=False)
# hidden: [tokens, 1024], including tokenizer-added BOS
ids, normalized = model.encode_tokens("hello world")
positions, logits = model.masked_logits("The capital of France is <|mask|>.")
predictions = model.fill_mask("The capital of France is [MASK].", top_k=5)
# predictions[0]["predictions"][0]["token"] == " Paris"
```

The C ABI exposes `crispembed_encode_tokens_raw`, `crispembed_encode_tokens`,
`crispembed_last_token_ids`, `crispembed_has_masked_lm`,
`crispembed_masked_logits`, and `crispembed_token_bytes`. Returned C buffers
belong to the context; copy results before the next inference call. Python,
Rust, and Dart wrappers copy returned arrays. Rust provides
`encode_tokens_raw`, `masked_logits`, `fill_mask`, and `token_bytes`; Dart
provides `encodeTokens(normalize: false)`, `maskedLogits`, `fillMask`, and
`tokenBytes`.

`masked_logits` accepts the literal `<|mask|>` token. CLI `--fill-mask` and
wrapper `fill_mask` / `fillMask` also translate `[MASK]`. Multiple masks are
predicted jointly in one bidirectional forward, with positions including BOS.
The output is unnormalized logits at mask positions only; the head consumes
raw hidden states. Top-K probabilities use the full vocabulary softmax. A token
may represent incomplete UTF-8 bytes; display decoders replace invalid UTF-8
as Hugging Face does. The GGUF contains 1134 UNUSED padding vocabulary slots;
these decode to empty strings rather than invented `[PADnnn]` text.

Ordinary `encode` still returns the L2-normalized first-token vector. It is
available for experimentation, but this checkpoint has not been trained to
make those vectors suitable for semantic retrieval. It has no ColBERT head.

CPU encoder APIs default on for GGUFs declaring a non-causal LFM2 model and
mask token. `CRISPEMBED_LFM2_ENCODER=0` disables these APIs and restores the
historical ordinary-text tokenizer for bisection. Other LFM2 checkpoints and
GPU encoder graphs require `CRISPEMBED_LFM2_ENCODER=1`. GPU scheduling has not
yet passed the reference suite, so it remains opt-in. Existing pooled and
ColBERT paths remain available. The model metadata declares 128000 positions;
the validation suite reaches 649 tokens and does not establish quality at the
metadata limit. Native token/logit APIs reject sequences exceeding metadata
length rather than silently truncating.

## Reference validation

The reference uses the checkpoint's `Lfm2BidirectionalForMaskedLM` class,
FP32 weights, eager bidirectional attention, and the official symmetric
ShortConv patch. Hooks capture embedding lookup, all 14 layer outputs, final
RMSNorm, raw/normalized first-token vectors, and mask logits. Production API
comparisons also check exact token IDs, vocabulary decoding, normalized/pooled
consistency, repeated mixed calls, and cleared failure outputs.

The 15-case suite includes English, German, French, Japanese/Chinese, Arabic,
Unicode digits, whitespace, adjacent masks, literal special tokens, an empty
string, and a 649-token input. Quantized results are measured against the same
FP32 reference separately from the F16 port-correctness gate.

CPU results (2026-10-03), against Torch 2.11.0 / Transformers 5.13.0.dev0,
FP32 eager reference. Relative error is `||native-reference||₂ / ||reference||₂`,
taking the worst case; cosine is the worst token/mask row across the suite.

| GGUF | Hidden min cosine | Hidden max relative error | Logits min cosine | Logits max relative error | Mask top-1 matches |
|---|---:|---:|---:|---:|---:|
| F16 | 0.99998947 | 0.1813% | 0.99999957 | 0.0976% | 15/15 |
| Q8_0 | 0.94665218 | 7.2592% | 0.99934877 | 3.6796% | 13/15 |
| Q4_0 | 0.73603180 | 79.9290% | 0.93175417 | 56.8220% | 9/15 |

F16 is the default because it passes the port-correctness gate and preserves all
15 decoded mask predictions. Q8_0 and Q4_0 execute correctly but their quantized
weights change predictions; Q4_0 also shows substantial hidden-feature drift.
All three match the reference token IDs and decode all 65536 vocabulary slots.
The F16 per-layer replay passes all 297 checks over 15 inputs, including the
production scheduler path. Live Rust and Dart examples exercise the same C ABI as Python.

To reproduce, keep weights/references outside the repository:

```sh
USE_TF=0 python tools/dump_lfm2_reference.py \
  --model "$LFM2_HF_DIR" --texts-file tests/lfm2_encoder_cases.json \
  --long-context --output "$LFM2_REF_DIR"
python tests/lfm2_encoder_parity.py \
  --model "$LFM2_F16" --refs "$LFM2_REF_DIR" \
  --lib build/libcrispembed.so --require-top1
build/test-lfm2-diff "$LFM2_F16" "$LFM2_REF_DIR/000.gguf" \
  "The capital of France is <|mask|>."
# Quantization quality: report drift and decoded changes without declaring parity.
python tests/lfm2_encoder_parity.py \
  --model "$LFM2_QUANT" --refs "$LFM2_REF_DIR" \
  --lib build/libcrispembed.so --measure-only
```

The converter `models/convert-lfm2-embed-to-gguf.py` also accepts the original
masked encoder checkpoint: it strips `lfm2.` weight prefixes, honors
`block_auto_adjust_ff_dim=false`, and writes special-token/mask metadata. Use
the official GGUFs when a fresh conversion is unnecessary. A fresh F16 conversion
also passes the complete 15-case reference suite and the 20-check first-case
per-layer replay.
