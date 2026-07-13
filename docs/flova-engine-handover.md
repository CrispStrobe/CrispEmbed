# Flova/omr_transformer C++ engine — handover brief (2026-07-13)

Everything for the Flova handwritten-OMR port is done and on `main` **except the
ggml inference engine `src/flova_ocr.cpp`**. Converter, reference oracle, the ABI
header, and the full architecture map all exist and are tested. This brief is
self-contained: a fresh session should be able to write + validate the engine from
it alone. The port is highly de-risked because the encoder is the **same DonutSwin
architecture already implemented in `src/mixtex_ocr.cpp`** (copy it, change config).

## Why this model
Flova/omr_transformer (Apache-2.0) is the **only permissive handwritten/whiteboard
music OMR model** (homr = AGPL; everything else is a dataset/detector, not an
end-to-end model — verified by search 2026-07-13). Monophonic "simple notes" →
LilyPond. Complements the shipped polyphonic printed engines SMT + TrOMR.

## What's already done (on `main`)
- `models/convert-flova-to-gguf.py` — tested; arch `flova_ocr`, 484 tensors, 573 MB
  f32. Maps DonutSwin encoder (naming shared with mixtex) + mBART decoder + the
  75-token vocab + hparams. **Run:** `python models/convert-flova-to-gguf.py --repo Flova/omr_transformer --output flova-f32.gguf`
- `tools/dump_flova_reference.py` — tested; 12-stage F32 oracle. **Run:**
  `USE_TF=0 python tools/dump_flova_reference.py --repo Flova/omr_transformer --image <sample1.png> --output flova_ref.gguf`
  (get samples via `hf_hub_download('Flova/omr_transformer','sample1.png')`).
- `src/flova_ocr.h` — the C ABI (init/free/recognize_raw/recognize_file/run_diff).

## Reference oracle stages (from the dumper, sample1.png)
`pixel_values (3,583,409)` [STRUCTURAL GATE] · `enc_stage0 (3796,256)` ·
`enc_stage1 (962,512)` · `enc_stage2 (247,1024)` · `enc_stage3 (247,1024)` ·
`enc_output (247,1024)` [encoder memory, post final LN] · `ids (40)` ·
`dec_block0..3 (40,1024)` [mBART layer outputs, teacher-forced] · `logits (40,75)`.
sample1 decodes to `c'2 a''8 c''8 r4 c'1 e'8 c'8 c'8 a''8 f'4 a'8 c'8` (model card).

## Hparams (in the GGUF as `flova.*`)
Encoder: patch_size 4, window_size 10, embed_dim 128, hidden_size 1024,
image_h 583, image_w 409, depths [2,2,14,2], num_heads [4,8,16,32].
Decoder: hidden_size 1024, num_layers 4, num_heads 16, ffn_dim 4096,
vocab_size 75, max_position 1536, scale_embedding 1.
Tokens: **decoder_start/bos 56 (`<s>`), eos 54 (`</s>`), pad 55, unk 0.**
image_mean/std [0.5,0.5,0.5]. `tokenizer.tokens` = 75 strings.

## ⚠ LANDMINE — the eos token
`generation_config.json` says `eos_token_id=2` — that is a **stale mBART default**
(id 2 = `.`). The REAL eos is the tokenizer's `</s>` = **54** (stored as
`flova.eos_token`). With 54 the greedy decode stops cleanly and matches the model
card; with 2 it never stops. (Same class as the TexTeller decoder-start trap.)

## ENCODER — DonutSwin (COPY `src/mixtex_ocr.cpp`)
`mixtex_ocr.cpp`'s `run_swin_encoder` + helpers (`window_partition`,
`window_reverse`, `cyclic_shift`, `window_mhsa`, `batch_layernorm`) implement the
**identical** DonutSwin. Copy them verbatim and change only:
- config: `embed_dim 128`, `depths [2,2,14,2]`, `heads [4,8,16,32]`, `window_size 10`,
  `hidden 1024`. rpb_table is `[num_heads, 361]` in ggml ne (361 = (2·10−1)²);
  rpb_index is `[100,100]` (100 = 10²). The scalar `window_mhsa` reads these
  generically — no code change, just the tensors.
- tensor names: `enc.patch.{weight,bias}`, `enc.patch_norm.{weight,bias}`,
  `enc.stage{s}.block{b}.{ln1,ln2,attn.{q,k,v,out},attn.rpb_table,attn.rpb_index,ffn.{up,down}}`,
  `enc.stage{s}.downsample.{norm,reduction}` (s<3).
- **final norm (Flova has it, SMT didn't):** after stage 3, apply `enc.final_norm`
  (LayerNorm) to the [247,1024] sequence → `enc_output`. This is the cross-attn memory.
- Swin block flow (per mixtex, non-preact HF DonutSwin): `res=x; x=LN1(x);` pad to
  window multiple; `if odd block: cyclic_shift(+shift)` (shift=ws/2=5) + build the
  9-region shifted-window attn mask (−100 across regions); window_partition →
  window_mhsa(+rpb+mask) → window_reverse; un-shift; un-pad; `x=res+x; x=res2+FFN(LN2(x))`.
  Downsample = patch-merge (concat 2×2 neighbourhood → LN → Linear reduction 4C→2C).
- **Validation:** feed the ref `pixel_values`, compare `enc_stage0..3` + `enc_output`,
  gate cos ≥ 0.999. (mixtex already has `MIXTEX_DIFF_REF` per-stage compares to copy.)
- Patch embed: Conv2d(3→128, k4 s4) — mixtex does it as a manual scalar conv; reuse.
  DonutSwin pads the input H,W up to a multiple of patch_size before patch embed
  (583→ceil, 409→ceil); verify pH/pW against the ref stage token counts (73×52=3796
  after stage0 downsample ⇒ pre-downsample 146×104; so patch grid is 146×104 → the
  input is padded to 584×416 before /4). Match the ref by construction.

## DECODER — mBART 4-layer (PRE-norm) — write fresh (mixtex decoder is a close start)
Per layer (mBART `normalize_before=True`, unlike BART):
```
res=x; x=self_ln(x);  x=self_attn(x, causal);        x=res+x            # self
res=x; x=cross_ln(x); x=cross_attn(q=x, kv=enc_out); x=res+x           # cross (no mask)
res=x; x=ffn_ln(x);   x=fc2(gelu(fc1(x)));           x=res+x           # ffn
```
Attention: 16 heads, head_dim 64, scale 64^-0.5, standard (q/k/v/out all have bias).
- **Embedding:** `x = embed_tokens(id)·√1024 + embed_positions(pos)`; mBART learned
  positions are **offset by 2**: `embed_positions.weight[pos+2]` (table is [1538,1024]
  = 1536+2). Then `x = layernorm_embedding(x)` (`dec.embed_ln`).
- After 4 layers: `x = final_norm(x)` (`dec.final_norm` = mBART's `layer_norm`), then
  `logits = lm_head(x)` (`dec.lm_head.weight` [75,1024], no bias; NOT tied).
- Tensor names: `dec.embed_tokens.weight`, `dec.embed_positions.weight`,
  `dec.embed_ln.{w,b}`, `dec.layers.{i}.{self,cross}_{q,k,v,out}.{w,b}`,
  `dec.layers.{i}.{self_ln,cross_ln,ffn_ln}.{w,b}`, `dec.layers.{i}.ffn.{up,down}.{w,b}`,
  `dec.final_norm.{w,b}`, `dec.lm_head.weight`.
- **Validation:** teacher-force the ref `ids`, compare `dec_block{0..3}` (the hook is
  the layer OUTPUT, post-residual) + `logits`, gate cos ≥ 0.999.

## Greedy decode + detok
Start `ids=[56]`; each step run decoder over the growing seq (+enc_output), argmax
last-position logits (75), append; stop on eos=54. Cap at ~128. **Detok:** concat
the token strings for ids (skip `<s>/</s>/<pad>/<unk>` = 56/54/55/0), replace `</w>`
→ space, strip. e.g. `c`+`'`+`2</w>`+`a`+`'`+`'`+`8</w>` → `c'2 a''8`.

## Preprocessing (DonutImageProcessor) — implement in `recognize_raw`
Target size (h=583, w=409), mean/std [0.5]. Pipeline (validate `pixel_values` cos>0.99999):
1. **align_long_axis:** if `image_w > image_h` (landscape) → `rot90(img, k=3)` (90° CW).
   (Condition: target is portrait 583>409, so rotate when the input is landscape.)
2. **thumbnail:** shrink (never enlarge) preserving aspect so both dims ≤ (583,409),
   BILINEAR (resample 2).
3. **do_resize** (Donut): resize so the image fits `size` keeping aspect (the shorter
   path — mirror `DonutImageProcessor.resize`; in practice thumbnail already did it).
4. **pad** to (583,409): center-pad? No — Donut pads to (top=..,) — HF pads
   symmetric-ish: `delta_w//2` left, `delta_h//2` top. Match against the ref.
5. rescale ×1/255, normalize `(x−0.5)/0.5` → [−1,1]. Output `(3,583,409)` (CHW).
Getting Donut preprocessing byte-exact is the fiddliest part; validate against the
stored `pixel_values` and iterate. (Deferred path: validate the engine on the ref
`pixel_values` first, like TrOMR did.)

## Validation plan (CPU only)
1. Feed ref `pixel_values` → compare `enc_stage0..3`, `enc_output` (cos ≥ 0.999).
2. Teacher-force ref `ids` → compare `dec_block0..3`, `logits` (cos ≥ 0.999); also
   check per-position argmax agreement (the decode-relevant metric).
3. Greedy decode → compare the LilyPond string to the ref decode AND the model card.
4. Native preprocessing: `pixel_values` cos > 0.99999, then end-to-end decode.
First stage below cos 0.999 = the bug (dev-guide HARD RULE #2).

## Wiring (after the engine validates — mirror `tromr_ocr`/`smt_ocr` exactly)
1. `CMakeLists.txt`: `list(APPEND CRISPEMBED_SOURCES src/flova_ocr.cpp)` +
   `add_executable(test-flova-diff tests/test_flova_diff.cpp)`.
2. `src/crispembed.cpp` dispatcher: `#include "flova_ocr.h"`, enum `OCR_MODEL_FLOVA`,
   `detect_arch`: `if (arch=="flova_ocr") return OCR_MODEL_FLOVA;`, init/free/
   recognize_raw/recognize_gray cases.
3. `examples/cli/main.cpp`: add `flova(music)` to the `--ocr` help.
4. Quantize → q8_0 (encoder is the bulk; if conv/patch weights need keeping, mirror
   the tromr `enc.bb` keep-guard idea — but Flova's Swin is all linear, so standard
   q8_0 should be fine; test decode vs ref). Upload `cstr/flova-omr-GGUF` (f32 + q8_0
   + Apache-2.0 card, attribution to Flova/omr_transformer +
   github.com/UHHRobotics22-23/robot_project) via `hf upload` (token in `../.env`
   HF_TOKEN, account `cstr`). Registry entry in `examples/cli/model_mgr.cpp`.
5. Regression fixture: add a `flova` entry to `tests/regression/manifest.json`
   (sample image + golden LilyPond, `max_cer` lenient). NOTE: LilyPond output has no
   spaces-as-tokens issue like TrOMR — the no-garbage guard is fine.

Then: `crispembed -m flova-q8_0.gguf --ocr score.png` → LilyPond.
