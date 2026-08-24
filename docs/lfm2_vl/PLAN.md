# LFM2.5-VL-3B — vision-language OCR backend

Engine: `src/lfm2_vl_ocr.{h,cpp}` · ShortConv decode step: `src/lfm2_shortconv.h`
Guard: `tests/test_lfm2_shortconv.cpp` (hermetic, weight-free, ~10 ms)

## NOW — active work

Branch `feat/lfm2vl-kv-decode`, tip `eec98fc7`.

- **DONE** — the "KV-cached decode is broken" bug is fixed, and it was never the
  KV cache. See §1 below.
- **DONE** — a second bug, in the path that had been made the *default* because
  it was believed correct. See §2.
- **IN FLIGHT** — re-running the full-recompute arm with the §2 fix so the two
  decode paths can be A/B'd against each other before the default is flipped to
  the KV path. Until that lands, the default stays full-recompute and the KV
  path stays behind `LFM2_VL_KV_CACHE=1`.
- **NEXT** — flip the default (pending the A/B); multi-tile NaFlex; registry
  entry + auto-download; README/docs; gate the remaining `[lfm2_vl]` prints
  behind `LFM2_VL_DBG=1`.

## §1 — ShortConv decode reduced over the wrong axis

`build_decode_step_graph` built the depthwise-conv product as `[conv_k, D]`
(flat index `d*K + k`) and then summed the taps with `ggml_view_1d` at byte
offsets `{0, D, 2D}` — which sums flat indices `{i, i+D, i+2D}` and straddles
taps and channels. The output has the correct shape `[D]` and a plausible
magnitude, so every shape check, every allocation, and every downstream op was
happy. 22 of the 30 LFM2 layers are ShortConv, so the decode was garbage from
layer 0.

Symptom: decode step 0 argmax `17177` ('差') instead of `1870` ('son').

This is why the whole KV-cache investigation (cache population, F32 vs F16
cache, view sizes, `ggml_cont` on K/V, sched backend assignment) came up empty
— the KV cache was correct the entire time.

**Fix**: the reduction moved to `lfm2_shortconv::step()`, which transposes the
*kernel* into the window's `[D, K]` layout so the tap axis is the slow axis and
each tap's `D` channel values are one contiguous run.

**Guard** (`tests/test_lfm2_shortconv.cpp`, written before the fix and watched
to fail — `max_abs` 1.07 at D=4, 2.68 at D=2048): checks the decode step against
(a) a scalar reference and (b) the *prefill* `ggml_conv_1d_dw` path's last
column. (b) is the invariant that actually matters, because it is what makes
KV-cached decode agree with full recompute.

| | before | after |
|---|---|---|
| decode step 0 vs scalar ref | max_abs 2.68 | **0.000** (exact) |
| decode step 0 vs prefill conv | max_abs 2.68 | **5.0e-4** (F16 kernel rounding) |
| decode step 0 argmax | 17177 ('差') | **1870 ('son')** ✓ |

## §2 — the full-recompute path never wrote its causal masks

`build_prefill_graph` declares one `causal_mask_<il>` input per attention layer.
The prefill call site wrote them; the no-KV full-recompute call site did not.

An unset ggml input is **not zero** — it is whatever the graph allocator last
left in that buffer. So in the path that had been made the default *because it
was believed correct*, every token after the first was decoded under an
allocator-leftover attention pattern.

That is exactly the observed `Jackpot!`: the first token comes from the real
prefill (correct — "Jack"), and everything after it came from garbage masks.
The dev guide's trap, verbatim: the wrong reference agreed with the wrong C++,
so parity proved nothing.

**Fix**: both callers now go through one `set_prefill_seq_inputs()`.

**Guard**: `declare_input()` records every input a graph declares, and
`audit_graph_inputs()` prints a `BUG:` line for any that nobody wrote. This is
structural — a new input added to `build_prefill_graph` and not to the setter
is reported at the next run rather than silently read as leftovers.

## Measured (Q4_K, `tests/regression/images/cc0/commons_example_receipt.png`, 15 tokens)

`LFM2_VL_KV_CACHE=1`, after both fixes:

```
Jackson-Washington
6640 Ortiz Cove, Markmouth
```

Real receipt text, correct English. Previously `Jackpot!` (recompute, garbage
masks) and `Jack差…` (KV, scrambled conv).

⚠ **Timings from this session are NOT usable.** The box was running another
agent's parallel Rust build throughout; the dev guide's rule 5 (both arms under
identical load, median of ≥3, never a contended box) is not satisfied. For the
record only, contended: vision encoder 299 s, projector 16 s, prefill 273 tok
172 s, KV decode ≈19 s/token. The vision encoder — not the decode — is the
dominant cost on this VPS and is the next real perf target.

## Env gates

| gate | default | meaning |
|---|---|---|
| `CRISPEMBED_ACCEPT_LFM_LICENSE` | off | required; LFM-1.0 is revenue-capped |
| `LFM2_VL_KV_CACHE` | off | KV-cached per-token decode instead of full recompute |
| `LFM2_VL_ZERO_CONV_STATE` | off | debug: zero the ShortConv state cache |
| `LFM2_VL_NO_REPEAT_NGRAM` | 5 | greedy no-repeat n-gram size |
| `LFM2_VL_DBG` | off | diagnostics |
| `LFM2_VL_DIFF_REF` | unset | per-stage diff archive |

## Still open (from the port handover)

- **Multi-tile NaFlex** — currently `smart_resize` to ~512² total pixels, one
  tile. The HF processor sets `do_image_splitting=true`, `use_thumbnail=true`,
  `max_tiles=10`.
- **Registry entry + auto-download** — `examples/cli/model_hashes.h`; mmproj
  auto-discovery (`mmproj-*-{F16,Q8_0,BF16}.gguf` in the same dir) already works.
- **README / `docs/ocr_backend_matrix.md`** — backend table row, the license
  gate, the LFM-1.0 revenue cap.
- **Projector parity** sits at cos 0.958 (F16 drift) while the vision encoder is
  0.999 and `llm_logits_last` is 0.9995. Worth a look, but it is not what was
  breaking the decode.
- **Debug prints** — several unconditional `[lfm2_vl]` `fprintf`s should move
  behind `LFM2_VL_DBG=1`.
