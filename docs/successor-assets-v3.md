# Successor assets v3: v2 plus an explicit WASM-only platform tolerance

Protocol: [successor-assets-v3.json](successor-assets-v3.json). Contract and
controls: `tools/successor_assets_v3.py`, `tests/test_successor_assets_v3.py`.
Workflow: `.github/workflows/successor-assets-v3.yml`, consumers
`.github/workflows/successor-assets-consumers-v3.yml`.

**Owner decision (2 November 2026).** The browser/WASM build of the published
quantized MiniLM (`all-MiniLM-L6-v2-iq4_xs`, sha `77087a5d…`) may differ from
native by a bounded, attributed amount. This is a documented platform
tolerance, not a parity claim.

## What changes and what does not

- v1 and v2 files, guards and receipts are read-only. The 50 successor files
  present at the adopted producer source `11f2ef7c` are pinned by Git blob;
  files added later, such as new receipts, are allowed.
- Producers, final base `8245fd47`, ggml `a06fc0bb` and the runtime-closure
  producer check are identical to v2.
- The consumer matrix, browser consumer and review are **derived** from the
  frozen v1 files by `DERIVATIONS` in `tools/successor_assets_v3.py`. Exactly
  three workflow lines, two browser-consumer lines and four review lines
  differ. The controls regenerate them and reject any other difference.
  Desktop, binding, mobile, Apple and cross-platform gates therefore stay the
  strict v1 gates (max-abs ≤ 0.002, cosine ≥ 0.99999), byte for byte.
- The browser consumer's numeric decision moves to
  `successor_assets_v3.py browser-gate`. It recomputes max-abs and cosine
  from the retained browser and native vectors and requires the browser's own
  reported values to agree (1e-12).

## The WASM tolerance

| | measured (packaged default) | bound (v3) | strict (unchanged elsewhere) |
|---|---|---|---|
| final pooled vector max-abs | 0.00837269052863121 | **≤ 0.0101** | ≤ 0.002 |
| final pooled vector cosine | 0.9986498388732901 | **≥ 0.99837** | ≥ 0.99999 |

- **Evidence:** the drift is deterministic. It is identical in two independent
  producer/consumer runs:
  [40354251257](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40354251257)
  and [40710490021](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40710490021)
  (producers 40724938611/40725005968 at `11f2ef7c`).
- **Rule:** the measured value plus a 20% margin, rounded outward to three
  significant digits. Max-abs 0.0083727×1.2 → 0.0101; the cosine deficit
  0.00135016×1.2 → 0.00163, so cosine ≥ 0.99837. Tests accept the measured
  values and the exact bound, and reject 0.01011 and a cosine below 0.99837.
- **Scope:** the packaged browser `crispembed-embed-wasm` final 384-d vector
  for this model and the fixture text only. Page errors must be empty, and the
  classical OCR cleanup must stay exact (authored blob cleared, RGB-exact).
  The embedding consumer exposes no tokens or text, so no further task-level
  equality is available there.
- **Attribution:** [v4 live-norm attribution](../tests/results/packaging/wasm-live-norm-v4-attribution.json),
  [layer3 FFN matmul](../tests/results/packaging/wasm-layer3-ffn-matmul-v1.json),
  [layer0 attention](../tests/results/packaging/wasm-layer0-attention-v1.json).
  On identical inputs, every elementwise op is bit-exact. The IQ4_NL and
  IQ4_XS matmuls and `flash_attn_ext` differ by at most about 1e-6 per op
  through kernel accumulation order (tinyBLAS/AVX2 FMA lanes vs generic
  scalar; `vec_dot_f32`/`vec_mad_f32` FMA and lane width; `expf`). Activation
  quantization (Q8_0 FP16 scale steps, Q8_K scales) amplifies these into the
  final drift. No single-op fix exists.
- **Caveat:** those replays held `CRISPEMBED_WASM_Q8_0_ROUND=1`. The bound
  itself comes from the packaged default (gate off), which is what the
  consumer runs.

Drift beyond the bound fails, as does any non-WASM regression. Nothing is
qualified, no release alias is created and the app stays on 0.17.12.

## Dispatch

The v1 consumers require the consumer checkout's runtime closure to equal the
producer source. The run is therefore dispatched from a branch at the adopted
producer source plus the v3 files, none of which are runtime paths:

```
gh workflow run successor-assets-v3.yml --ref <branch at 11f2ef7c + v3 files> -f native_run=40724938611 -f wasm_run=40725005968
```

## Parked optional experiment

A multi-kernel native-order WASM build could remove the drift: IQ4_NL tinyBLAS
order, IQ4_XS AVX2 order, flash `vec_dot_f32`/`vec_mad_f32` with `fmaf` and
the AVX lane order, and a matching `expf`. It is recorded, not scheduled.

## Runs
