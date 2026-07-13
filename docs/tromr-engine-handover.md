# TrOMR C++ engine — handover brief (2026-07-13)

Everything for the Polyphonic-TrOMR OMR port is done and merged to `main` **except
the ggml inference engine `src/tromr_ocr.cpp`**. This brief is self-contained:
architecture, exact tensor names/shapes, formulas, reuse patterns, validation plan,
and wiring. A fresh session should be able to write + validate the engine from this
alone. The port is fully de-risked (converter, reference oracle, and the SAME-pad
solution all exist and are tested; the model is confirmed accurate on real photos).

## What's already done (on `main`)
- `models/convert-tromr-to-gguf.py` — tested; arch `tromr_ocr`, 261 tensors, 86 MB.
  Pre-standardizes the 40 StdConv2dSame backbone kernels (timm-exact: `F.batch_norm`
  per output channel, eps 1e-6) so the engine runs plain convs. Packs the 3 tokenizer
  vocabs + hparams. **Run:** `python models/convert-tromr-to-gguf.py --repo <tromr dir> --output tromr-f32.gguf`
- `tools/dump_tromr_reference.py` — tested; 23-stage F32 oracle GGUF from the real model.
- `src/tromr_ocr.h` — the C ABI (init/free/get_hparams/recognize_raw/recognize_file/run_diff).
- `flutter/crispembed/lib/src/omr.dart` — `CrispEmbedOmr` (auto-detect; covers TrOMR once wired).
- `models/convert-tromr-to-gguf.py`, dumper, header all committed.

## Environment (scratchpad is ephemeral `/private/tmp/...` — RECREATE it)
```bash
S=<scratch dir>
git clone --depth 1 https://github.com/NetEase/Polyphonic-TrOMR.git $S/Polyphonic-TrOMR   # 86 MB checkpoint is in-repo (not LFS)
~/miniconda3/bin/pip install --no-deps --target=$S/xt029 "x-transformers==0.29.2"          # checkpoint needs OLD x-transformers
```
Python gotchas to run the reference:
- `timm` pulls in `trio` (broken on this mac): `sys.modules["trio"]=types.ModuleType("trio")` BEFORE importing timm.
- `.pth` was saved on CUDA: wrap `torch.load` with `map_location="cpu"`.
- Prepend `$S/xt029` to `sys.path` so the old x-transformers loads (new one fails `strict=True`:
  checkpoint has `.net.`-style FF + LayerNorm `weight/bias`; new expects `.ff.` + `.gamma`).
- `USE_TF=0` for the albumentations/transformers import.
- **Reference dumper:** `USE_TF=0 python tools/dump_tromr_reference.py --repo $S/Polyphonic-TrOMR/tromr --xt $S/xt029 --image $S/Polyphonic-TrOMR/examples/photo1.jpg --output $S/tromr_ref.gguf`

## ⚠ Model-quality note (proven): use the PHOTOS, not the `.png`
`examples/N.png` are 4-channel *reference renderings*; `staff2score.readimg` does
`255 - alpha` on their opaque alpha → an **all-black** image. The real inputs are
`examples/photoN.jpg`. On the photos the model reads clefs/keys/rhythms/**pitches**
correctly (verified vs `examples/N.txt`). This is a real, accurate model — worth the port.

## Hparams (from `config.yaml`; all in the GGUF as `tromr.*`)
channels 1, patch_size 16, max_height 128, max_width 1280, max_seq_len 256,
encoder_dim 256, encoder_depth **4** (ViT blocks), encoder_heads 8,
decoder_dim 256, decoder_depth 4 (→ 12 x-transformers layers), decoder_heads 8,
num_rhythm 260 / pitch 71 / lift 7 / note 2, bos 1, eos 2, pad 0, nonote 0,
backbone_layers [2,3,7], norm_mean 0.7931, norm_std 0.1738.

## Preprocessing (staff2score.readimg + transform) — implement in `recognize_raw`
1. To RGB (for a plain photo jpg: cv2 BGR→RGB; app-supplied RGB bytes are already RGB).
2. Resize: `new_h = 128`, `new_w = int(128/h*w) // 16 * 16` (aspect-preserving, width multiple of 16).
3. `ToGray` (RGB luma) → **`Normalize(mean=0.7931, std=0.1738)`** → 1 channel.
   i.e. `t = (luma/255 - 0.7931) / 0.1738`. **NON-inverted** (dark ink → negative).
   Input tensor: `(1, 1, 128, W)`.

## ENCODER — timm hybrid ViT (validate to `enc_backbone` then `enc_context`)
`ResNetV2(layers=[2,3,7], in_chans=1, preact=False, stem_type='same', conv_layer=StdConv2dSame, channels=(256,512,1024))`

**SAME padding** (asymmetric): for kernel k, stride s, input `in`:
`out = ceil(in/s)`, `total = max((out-1)*s + k - in, 0)`, `lp = total//2`, `rp = total-lp`.
Implement as `ggml_pad_ext(g, x, lp_W, rp_W, lp_H, rp_H, 0,0,0,0)` then `ggml_conv_2d(w, x, s, s, 0,0,1,1)`.
**Reuse `src/ppformulanet_ocr.cpp`** — `ppfn_conv_nopad` + `ggml_pad_ext` do exactly this (see lines ~360-419).
Common cases here: stem 7×7/s2 → pad (2,3); 3×3/s2 → (0,1); 3×3/s1 → (1,1); maxpool 3×3/s2-same → (0,1). 1×1 convs → no pad.

- **Stem** (`stem_type='same'`): StdConv 1→64 k7 s2 (SAME) → `GroupNormAct(32,64)`+ReLU → MaxPool k3 s2 SAME. → /4.
  (`ggml_pool_2d(GGML_OP_POOL_MAX, k=3, s=2, pad computed via pad_ext then p=0)`.)
- **Stage 0** (2 Bottleneck blocks, out 256, mid 64, stride 1); **Stage 1** (3 blocks, out 512, mid 128, stride 2);
  **Stage 2** (7 blocks, out 1024, mid 256, stride 2). Total /16 → (H=8, W=W/16).
- **Bottleneck.forward** (non-preact): `shortcut = downsample(x) if block0 else x`;
  `x = ReLU(GN(conv1(x)))` (1×1) → `x = ReLU(GN(conv2(x)))` (3×3, stride, SAME) → `x = GN(conv3(x))` (1×1, no act);
  `out = ReLU(x + shortcut)`. GroupNorm: **32 groups, eps 1e-5**, per-channel affine.
  `downsample = StdConv 1×1 (stride) → GN (no act)`. mid = out*0.25.
- **HybridEmbed proj**: `Conv2d 1024→256, k1` (**plain Conv2d, NOT pre-standardized**). Flatten → `(N, 256)`, N=8·(W/16).
- **ViT** (`CustomVisionTransformer.forward_features`):
  - prepend `cls_token` [1,1,256] → `(N+1, 256)`.
  - **custom pos-index** (compute on HOST, `ggml_get_rows(pos_embed, ind)` then add):
    `h=H//16 (=8), w=W//16`; `pos_ind = repeat(arange(h)*(1280//16 - w), 'h->(h w)', w=w) + arange(h*w)`;
    `pos_ind = cat([0], pos_ind+1)` (cls at 0). `pos_embed` is `[1, 641, 256]` (641 = 1 + 8·80).
  - 4 blocks (timm standard, pre-norm LayerNorm eps 1e-6, NO mask):
    `x = x + attn(norm1(x))`: qkv `[768,256]`+bias → split → 8-head attn scale `64^-0.5` → proj `[256,256]`+bias.
    `x = x + mlp(norm2(x))`: fc1 `[1024,256]`+bias → **GELU-erf** → fc2 `[256,1024]`+bias.
  - `x = norm(x)` (`encoder.norm`, final LayerNorm). Output `(N+1,256)` = decoder cross-attn memory (`enc_context`).

## DECODER — x-transformers (validate `dec_block0..11` + `logits_*`)
12 layers = `('a','c','f') × 4` (self→cross→ff). **Pre-norm** LayerNorm (eps 1e-5, weight+bias), residual `x = x + block(norm(x))`.
- **Input embedding** (ScoreTransformerWrapper): `x = rhythm_emb(r) + pitch_emb(p) + lift_emb(l) + pos_emb(r)`.
  Token embs = plain lookups (**unscaled**). `pos_emb = emb(arange(L)) * 256^-0.5` (**scaled!**). `project_emb`=Identity.
- **'a' self-attn** (causal): q/k/v = `to_q/k/v(norm(x))` (`[512,256]`, **no bias**), 8 heads dim 64, `dots = q·kᵀ · 64^-0.5`,
  causal mask (upper-tri = -inf), softmax, `out = attn·v`. **attn-on-attn** `to_out`: `Linear(512→512, no bias)` → **SIGLU**
  (nn.GLU: `value · sigmoid(gate)`, value=first-half) → 256.
- **'c' cross-attn** (no mask): q = `to_q(norm(x))`, k/v = `to_k/to_v(enc_context)`. Same attn + attn-on-attn to_out.
- **'f' ff_glu**: `net.0.proj` `[2048,256]`+bias → **GEGLU** (`value · GELU(gate)`, value=first-half → 1024) → `net.3` `[256,1024]`+bias.
- `x = norm(x)` (`decoder.net.norm`). Then 4 heads (Linear+bias): `to_logits_{rhythm,pitch,lift,note}`.

**GLU chunk order** (both): `proj(x).chunk(2, dim=-1)` → `(value, gate)`, result `value * act(gate)`.
Check `ggml_glu` split convention matches (value=first half); ggml GLU ops: `GGML_GLU_OP_SIGLU` (attn-on-attn), `GGML_GLU_OP_GEGLU` (ff). If ggml's default splits the other way, use the op's `swapped` flag or slice manually.

## DECODE loop (greedy — use argmax; the dumper already does)
`out_r=[bos], out_p=[nonote], out_l=[nonote]`. Each step: run decoder over the 3 growing streams
(+ `enc_context`, mask all-True), take `argmax` of the last position of rhythm/pitch/lift logits, append.
Stop when rhythm == eos. (Repo uses stochastic multinomial temp 0.2 — DON'T; argmax is deterministic and the
oracle uses it.) Detokenize each stream via `tromr.{rhythm,pitch,lift}_tokens` (strip `Ġ`→space, drop `[BOS]/[EOS]/[PAD]`).
Output: flexible (user parses later) — e.g. join rhythm stream, or merge `rhythm|pitch` per event.

## Exact tensor names (verbatim in the GGUF)
Encoder backbone: `encoder.patch_embed.backbone.stem.conv.weight [64,1,7,7]`, `.stem.norm.{weight,bias}[64]`;
`...stages.{s}.blocks.{b}.{conv1,conv2,conv3}.weight` + `.{norm1,norm2,norm3}.{weight,bias}`;
`...blocks.0.downsample.conv.weight` + `.downsample.norm.{weight,bias}` (block 0 of each stage).
Stages: 0→(2 blk, mid 64, out 256), 1→(3 blk, mid 128, out 512), 2→(7 blk, mid 256, out 1024).
Proj: `encoder.patch_embed.proj.weight [256,1024,1,1]`, `.bias [256]`.
ViT: `encoder.cls_token [1,1,256]`, `encoder.pos_embed [1,641,256]`,
`encoder.blocks.{0..3}.{norm1,norm2}.{weight,bias}`, `.attn.qkv.{weight[768,256],bias}`, `.attn.proj.{weight[256,256],bias}`,
`.mlp.fc1.{weight[1024,256],bias}`, `.mlp.fc2.{weight[256,1024],bias}`; `encoder.norm.{weight,bias}`.
Decoder: `decoder.net.{rhythm_emb,pitch_emb,lift_emb}.emb.weight`, `decoder.net.pos_emb.emb.weight [256,256]`;
`decoder.net.attn_layers.layers.{0..11}.0.0.{weight,bias}` (pre-norm LN);
attn layers (0,1,3,4,6,7,9,10): `.1.{to_q,to_k,to_v}.weight [512,256]`, `.1.to_out.0.weight [512,512]` (no bias);
ff layers (2,5,8,11): `.1.net.0.proj.{weight[2048,256],bias}`, `.1.net.3.{weight[256,1024],bias}`;
`decoder.net.norm.{weight,bias}`; `decoder.net.to_logits_{rhythm,pitch,lift,note}.{weight,bias}`.

## Codebase reuse
- SAME-pad conv + `ggml_pad_ext`: `src/ppformulanet_ocr.cpp` (`ppfn_conv_nopad`).
- `ggml_group_norm(g, x, 32, 1e-5f)` + per-channel affine (reshape weight/bias to `[1,1,C,1]`, mul/add).
- Attention: `src/math_ocr.cpp` `g_mha` (scaled) / `src/smt_ocr.cpp` `mha_core` (adapt: TrOMR IS scaled `64^-0.5`, causal for self, none for cross).
- `ggml_pool_2d` (MAX), `ggml_gelu_erf`, `ggml_glu` (SIGLU/GEGLU), `ggml_get_rows` (pos-index).
- gguf load / greedy-decode / diff-harness scaffolding: `src/smt_ocr.cpp` (best template — same class).

## Validation plan (CPU only — Metal `set_output` snapshots lie)
1. Feed the ref `input_tensor`. Compare `enc_backbone` (ResNet out) and `enc_context` (ViT out). Gate cos ≥ 0.999.
2. Teacher-force the ref `ids_{rhythm,pitch,lift}`. Compare `dec_block0..11` (⚠ the dumper hooks the *block* output,
   pre-residual — either match that in C++, or change the dumper to capture post-residual) and `logits_{rhythm,pitch,lift,note}`.
3. Greedy decode → compare streams to the ref AND to ground truth (`examples/N.txt`).
First stage below cos 0.999 = the bug (dev-guide HARD RULE #2).

## Landmines
- SAME padding is asymmetric — compute lp/rp per input size (formula above). ggml_pad_ext dim order is (W,H).
- pos_emb is **scaled** by `256^-0.5`; token embs are NOT.
- ViT attn has NO mask; decoder self-attn IS causal; cross-attn NO mask.
- attn-on-attn = SIGLU, ff = GEGLU; verify ggml_glu's value/gate split (value=first half).
- GroupNorm 32 groups eps 1e-5; ViT/ff GELU = **erf** (`ggml_gelu_erf`).
- Quantizer flattens 4D conv weights (like SMT) — either reshape back in-engine (see `smt_ocr` conv helper) or add
  `patch_embed`/`backbone` to the `tools/quantize.cpp` keep-guard. Test q8_0 decode vs ground truth.
- Metal: if SAME-pad conv/pool misbehaves, force the encoder on CPU (env gate) like other conv engines; validate CPU first.

## Wiring (after the engine validates — mirror SMT exactly)
1. `CMakeLists.txt`: `list(APPEND CRISPEMBED_SOURCES src/tromr_ocr.cpp)` + `add_executable(test-tromr-diff tests/test_tromr_diff.cpp)`.
2. `src/crispembed.cpp` dispatcher: `#include "tromr_ocr.h"`, enum `OCR_MODEL_TROMR`, `detect_arch`: `if (arch=="tromr_ocr") return OCR_MODEL_TROMR;`,
   and the init/free/recognize/recognize_gray switch cases (recognize path → `tromr_ocr_recognize_raw`, which does preprocessing internally).
3. `examples/cli/model_mgr.cpp`: registry entry (after upload).
4. Quantize (`crispembed-quantize`) → q8_0. Upload `cstr/tromr-GGUF` (f32 + q8_0 + Apache-2.0 README, attribution to
   NetEase/Polyphonic-TrOMR + arXiv:2308.09370) via `hf upload` (token in `../.env` as `HF_TOKEN`, account `cstr` — see
   [[hf-upload-token-cstr]]). Verify card license + registry auto-download.
5. Dart `CrispEmbedOmr` + CLI `--ocr` auto-cover it once the dispatcher knows `tromr_ocr` — no further binding work.

Then it's done: `crispembed -m tromr-q8_0.gguf --ocr score.jpg` and Dart `CrispEmbedOmr('tromr-q8_0.gguf')`.
