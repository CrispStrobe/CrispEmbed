# CrispEmbed v0.17.19

Patch release, 12 November 2026. Changes since [v0.17.18](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.18),
through `c134f689` (merge of PR #216).

## Default changes (OvisOCR2 / Qwen3.5-OCR)

[PR #213](https://github.com/CrispStrobe/CrispEmbed/pull/213) added the gates and
[PR #215](https://github.com/CrispStrobe/CrispEmbed/pull/215) turned them on by default:

| Variable | New default | Opt out |
|---|---|---|
| `CRISPEMBED_QWEN35_OCR_GRAPH_DECODER` | on (graph decoder) | `=0` restores `decoder_cpu` |
| `CRISPEMBED_QWEN35_GRAPH_MATMUL` (for `qwen35_ocr`) | `native` | `=f32` for F32 matmul copies |
| `CRISPEMBED_QWEN35_VISION_THREADED_ATTN` | on | `=0` restores the scalar loop |

- The full eleven-case OvisOCR2 quality set (1,841 tokens) has exact IDs, text, EOS,
  stop and cap with both matmul modes. The maximum selected-confidence drift is 8.4e-6.
  The threaded vision attention is bit-identical to the scalar loop.
- LightOnOCR-3-0.8B (Qwen3.5-OCR) uses the same path. Its three realistic documents
  are exact, and decode drops from about 3.7-4.1 s/token to 112-116 ms/token on CPU.
- MiniCPM keeps `f32` matmul. The graph-path confidences are recorded as an additive
  reference. The frozen `decoder_cpu` reference is unchanged.

## Qwen3.5-OCR on GPU (opt-in, default off)

[PR #216](https://github.com/CrispStrobe/CrispEmbed/pull/216): `CRISPEMBED_QWEN35_OCR_GPU=1`
builds the `qwen35_ocr` vision encoder and decoder from standard ggml ops only. Each graph
runs on the preferred backend, and the CPU backend is used when no GPU is present.
`CRISPEMBED_QWEN35_OCR_GPU_LOG` prints the node placement, and
`CRISPEMBED_QWEN35_OCR_GPU_STRICT` rejects any fallback to the CPU.
On a Kaggle T4, every node runs on CUDA and the three realistic documents are exact
(about 89 tok/s decode). The full eleven-case set is exact on the CPU backend.

## Tests and CI

- Frozen guards: stale guards were re-pinned from fresh replays as recorded,
  behaviour-neutral amendments, and the delta checks were anchored to their
  declaration commits ([#209](https://github.com/CrispStrobe/CrispEmbed/pull/209),
  [#211](https://github.com/CrispStrobe/CrispEmbed/pull/211),
  [#212](https://github.com/CrispStrobe/CrispEmbed/pull/212)). No frozen value,
  budget or gate changed.
- The plain-environment pytest lane gives a meaningful result: 0 errors and explicit
  skips ([#206](https://github.com/CrispStrobe/CrispEmbed/pull/206)). The consumer
  check now scores OCR output ([#205](https://github.com/CrispStrobe/CrispEmbed/pull/205)).
- Kaggle build kernels refresh the ccache seed automatically when it is stale
  ([#214](https://github.com/CrispStrobe/CrispEmbed/pull/214)).

No default model alias or pub.dev publication changes.
