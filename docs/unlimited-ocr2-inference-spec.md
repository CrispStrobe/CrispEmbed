# Unlimited-OCR-2 (rev 04647c7a) — inference spec for a v1 ggml engine

Scope: `UnlimitedOCRForCausalLM.infer(tokenizer, prompt='', image_file=<one image>)` with defaults
(`crop_mode=None`, `temperature=0`, `dash_topm=16`, `page_count_guard=True`, `max_page_tokens=20000`,
`no_repeat_ngram_size=0`, CPU => Triton paths off). File abbreviations:

- `MU` = v2/modeling_unlimitedocr.py
- `MD` = v2/modeling_deepseekv2.py (DASH monkey-patch + the inference router)
- `SB` = v2/_sparse_base_impl.py (base model, rope, page helpers)
- `DA` = v2/dash_attention.py (DashAttention module, entmax, dense attend)
- `CF` = v2/config.json

Summary of the deltas vs v1:

| Area | v1 | v2 |
|---|---|---|
| Vision encoder / projector / newline / separator | — | **unchanged** (deepencoder.py differs only in comments; same tensors) |
| Max tiles | `max_num=32` (v1 MU:174) | `max_crops=16` (MU:1043, 557) |
| Prompt | conversation template `plain` | raw: `BOS + "<PAGE><image>" + "Multi page parsing."` (MU:1073-1078, 571) |
| Main attention RoPE | 1D NEOX theta 1e4 | **2-axis "page2d" NEOX**, 63 token pairs at theta 1e4 (resampled) + 1 page pair (page mod 2). **YaRN/500000 is NOT used on the default path** |
| Attention pattern | full causal (v1 had ring SWA in decode) | **DASH block-sparse** in prefill AND decode: local 128 tokens + ≤12 routed 32-token blocks with additive log-prior bias |
| New tensors / layer | — | `self_attn.dash_router.{q_proj[128,1280], k_proj[128,1280], q_norm.weight[128], k_norm.weight[128], head_cls[128]}` (used) ; `self_attn.indexer.*` (loaded, **unused** numerically on this path) |
| MoE | softmax, greedy top-6, no renorm, scale 1.0 | identical |
| Stopping | HF generate, EOS | custom loop, EOS + PageCountGuard |

---

## 1. Input construction (`_uocr_build_inputs`, MU:531-580)

### 1.1 Prompt string
- `prompt` defaults to `'Multi page parsing.'` when images are given (MU:1073).
- If the prompt has no `<image>`: `prompt = '<PAGE><image>' * n_img + prompt` (MU:1077-1078).
  If a user prompt contains exactly one `<image>` per image it is used **as is** (MU:1081) — then there is
  **no `<PAGE>` marker** and the whole page-aware machinery falls back (see §2.5). Default ⇒ marker present.
- `crop_mode=None` + 1 image ⇒ `cropping=True` (gundam) (MU:541).

### 1.2 Token layout
```
ids = [BOS=0]                                   # MU:571 (bos_id=0, not a tokenizer special)
    + tok("<PAGE>")                              # splits[0], expected == [128827]  (MU:550)
    + IMG * n_img_tokens                         # IMG = 128815                     (MU:564-568)
    + tok("Multi page parsing.")                 # splits[-1]                       (MU:570)
seq_mask = True exactly on the IMG positions     (MU:569)
```
`tokenizer.encode(..., add_special_tokens=False)`. `<PAGE>` must tokenize to the single added token 128827
(CF `page_token_id`), otherwise the page logic silently degrades — verify against the tokenizer.

### 1.3 Views and image-token counts
- `nq = ceil((640/16)/4) = 10`, `nqb = ceil((1024/16)/4) = 16` (MU:546-547).
- Tiling (MU:554-560): if `w<=640 and h<=640` ⇒ ratio (1,1), no tiles. Else `dynamic_preprocess(min_num=2,
  max_num=16, image_size=640)` (MU:229-267): candidate grids `(i,j)` with `2 <= i*j <= 16`, sorted by `i*j`,
  closest aspect ratio, tie ⇒ later (larger) grid wins if `area > 0.5*640*640*i*j`; resize (PIL default bicubic)
  to `(640*i, 640*j)`, crop row-major. No thumbnail.
- Global view: `ImageOps.pad(image, (1024,1024), color=(127,127,127))` (letterbox, centered) (MU:561).
- Normalization: ToTensor then mean=std=0.5 ⇒ `x/255*2-1` (MU:544, 376-398), bf16.
- Token count per page (MU:564-567):
  `N_img = 16*17 + 1  [+ (10*w_t + 1) * (10*h_t)  if w_t>1 or h_t>1]` → 273 for an untiled page.
- `spatial = [[w_t, h_t]]` (MU:563) is passed through to vision; **no position-id/spatial structure is passed
  to the decoder** — `position_ids = arange(S)` (MU:1163); all 2-D positional info is recomputed from token ids.

### 1.4 Embedding order inside the IMG span (MU:731-763, unchanged from v1)
Tokens are all identical IMG ids, features are `masked_scatter`ed in this order (local first!):
```
tiled:   [local grid (10*h_t rows, each 10*w_t feats + image_newline)]
         [global 16 rows, each 16 feats + image_newline]
         [view_seperator]
untiled: [global 16x(16+newline)] [view_seperator]
```
Feature = projector(cat(CLIP[:,1:], SAM.flatten.permute)) exactly as v1.

### 1.5 Native tiling status
`src/unlimited_ocr.cpp` implements §1.3/§1.4 gundam tiling behind `CRISPEMBED_UNLIMITED_OCR2_TILES=1`
(default OFF; variant-2 GGUFs only; only when `w > 640 or h > 640`). With the gate off, or for pages
of at most 640x640, the global-only 273-token path is unchanged. **Status: implemented, not yet
validated against the reference** (the Kaggle native kernel now runs the tiled `para-en`/`table`
cases with the gate on; nothing is qualified until those IDs match).

Parity risks, most likely first:
- **Tile resize.** A port of Pillow's `Resample.c` BICUBIC (a=-0.5, support widened by the
  downscale factor, 22-bit fixed-point coefficients, horizontal pass then vertical, uint8 rounding
  after each pass). Intended to be bit-exact with Pillow, but not yet checked; a Pillow version
  with different rounding would show up as small pixel differences.
- **Position-embedding resize for 640 inputs** (SAM 64→40 grid, CLIP 16→10 grid): torch-style
  bicubic+antialias in fp32, used only for tiles. Summation order differs from torch (~1e-7). The
  existing bilinear SAM interpolation (`UOCR_OPT_SAM_RES`) is unchanged.
- **Grid choice ties.** The reference sorts a Python `set` by `i*j`, so the order among equal
  products is CPython set order; native uses lexicographic order. This only matters for an exact
  float tie between two different grids with the same product.
- **Global view.** The tiled path reuses the existing letterbox resize (non-antialiased
  bicubic). Downscaling to 1024 is exact for the 1024x768 fixtures but may differ from
  `ImageOps.pad` on larger pages.

---

## 2. Main-attention positions and RoPE

Decoder: 12 layers, MHA 10 heads × head_dim 128 (no MLA, `qk_rope_head_dim=0`), `SlidingWindowLlamaAttention`
(SB:3627, 1366). All 12 layers are sparse/DASH layers (`sparse_attention_freq` all 1, CF) ⇒ every layer has
an `indexer` and a `dash_router` (SB:1384, MD:80-82).

### 2.1 Which RoPE is used
`sparse_main_rope_mode='page2d'` ⇒ `rotary_emb_2d_main = DeepseekV2IndexerRotary2D(head_dim=128, base_tok,
base_page, page_pairs, tok_freq_mode, page_mod)` (SB:1413-1425) with the parameters read from these keys:

| param | key | value |
|---|---|---|
| base_tok | `sparse_rope_tok_theta` (SB:1419) | **10000.0** |
| base_page | `sparse_rope_page_theta` | 256.0 |
| page_pairs | `sparse_rope_page_pairs` | 1 |
| tok_freq_mode | `sparse_main_rope_tok_freq_mode` | `resample` |
| page_mod | `sparse_rope_page_mod` | 2 |

**`rope_theta=500000` and YaRN (`rope_scaling`) only feed `self.rotary_emb` (HF LlamaRotaryEmbedding,
SB:1374) which is used only in the fallback (§2.5).** No mscale/attention_factor on the page2d path
(SB:2059-2061: plain cos/sin). `sparse_rope_page_span=0` ⇒ no span normalisation (SB:2569).

### 2.2 Frequencies (SB:2018-2061)
```
half = 64; n_tok = half - page_pairs = 63
inv_tok[i]  = 10000 ** (-i / 63),        i = 0..62     # 'resample' (SB:2032); NOT i*2/128
inv_page[0] = 256 ** (0/1) = 1.0                         # SB:2033
ang[0:63]   = tok_pos * inv_tok
ang[63]     = (page_pos mod 2) * 1.0                     # SB:2052-2056
emb = cat(ang, ang)   -> [128];  cos = cos(emb), sin = sin(emb)   (fp32, then cast to activation dtype)
```
Applied with HF `apply_rotary_pos_emb` (`rotate_half`) ⇒ **split-half NEOX pairing**: dim j pairs with j+64,
pair j<63 is the token axis, pair 63 (dims 63 and 127) is the page axis. Same q and k rotation; K is cached
already rotated (SB:1438-1441, MD:183-186).

### 2.3 Coordinates in prefill (`_build_page_rope_pos` SB:2511-2556, `_page_rope_offsets` SB:2558-2607)
Built once from layer 0's indexer and handed to all layers (MU:667-693).
```
img  = ids == 128815;  mk = ids == 128827
cum  = cumsum(mk)                                  # inclusive, 1-based
last_img = index of last IMG token
n_pages  = cum[last_img]                           # markers up to last image (1 for single page)
in_prefill[t] = t <= last_img
page[t] = (in_prefill ? cum[t] : cum[t] - n_pages).clamp(min=0)
tok[t]  = (# of positions u<=t with page[u]==page[t]) - 1      # running count PER PAGE BUCKET
pf_cnt[p] = # of in_prefill positions with page p                # saved for decode
n_pg = max(page)+1
```
Single page example (`S = 2 + N_img + T_instr`):

| token | page_pos | tok_pos |
|---|---|---|
| BOS (t=0) | 0 | 0 |
| `<PAGE>` (t=1) | 1 | 0 |
| IMG k (t=2+k) | 1 | 1+k |
| instruction token j (after last image) | 0 | 1+j |

`pf_cnt = [1, 1+N_img]`, `n_pg = 2`. Note the instruction shares bucket 0 with BOS and continues its count.

### 2.4 Coordinates in decode (`_main_rope_coords` SB:1479-1489)
Per-layer state seeded at prefill end (SB:1465-1477): `page = page_pos[-1]` (=0, the instruction),
`tok = tok_pos[-1]`. For each fed token `x` (the token just generated):
```
if x == 128827:  page = min(page+1, n_pg-1);  tok = pf_cnt[page]
else:            tok  = tok + 1
```
So output text of page k continues page k's prefill bucket (`<PAGE>` gets tok = 1+N_img, the next text token
2+N_img, ...). Text generated before the first `<PAGE>` stays in bucket 0. Page is clamped to `n_pg-1`
(a 2nd generated `<PAGE>` on a 1-page doc stays on page 1, tok resets to pf_cnt[1]).

### 2.5 Fallback (1D YaRN) — only if the prompt has no `<PAGE>`/no image
`_build_page_rope_pos` returns None (SB:2542-2552) ⇒ `self.rotary_emb(v, position_ids)` = HF
`LlamaRotaryEmbedding` with yarn (transformers 4.52 `_compute_yarn_parameters`), dim 128, base 500000:
```
pos_freqs = 500000 ** (arange(0,128,2)/128)
inv_extra = 1/pos_freqs ; inv_inter = 1/(4*pos_freqs)
low  = floor(128*ln(32768/(32*2π)) / (2 ln 500000)) = 24
high = ceil (128*ln(32768/(1 *2π)) / (2 ln 500000)) = 42      (clamped to [0,127])
ramp[i] = clamp((i-low)/(high-low), 0, 1), i=0..63 ;  extra_factor = 1 - ramp
inv_freq = inv_inter*(1-extra_factor) + inv_extra*extra_factor
cos,sin = cos/sin(pos*inv_freq) duplicated (NEOX) * attention_factor(=1.0)
```
(low/high computed by me from the formula; verify against transformers 4.52 when implementing.) The
router then uses `indexer.rotary_emb_1d` (SB:2337-2340, 1935-1981) = the same yarn rope at dim 128, theta
`sparse_rope_theta`=500000. Positions are plain `arange` (prefill) / `S+len(gen)-1` (decode, MU:1188).
The default infer path never takes this branch; implement it only for prompts with user-supplied `<image>`.

---

## 3. DASH in PREFILL (q_len > 1)

Entry: `_patched_forward` (MD:216-238) → `_dash_attn_forward` (MD:146-213) → `_infer_prefill_attend`
(MD:433-451) because `configure_dash_inference(on=True)` sets `_INFER['on']` (MD:296-335, MU:1154).

### 3.1 Inference parameters (MD:308-328)
Read from a snapshot of the **checkpoint** config (`_CKPT_SP`, MD:56-72):
```
stage0 = env DASH_STAGE0 or sp.dash_stage0 = "head_cls"
K      = sp.dash_topk_blocks = 12            # hard read budget (NOT sparse_topk_blocks=16)
topm   = dash_topm = 16                       # entmax window
idx_scale = dash_scale_per_layer[l] / sqrt(128) = 128/11.3137 = 11.3137   (all layers)
floor  = dash_floor_per_layer[l] = 1e-4
sigma  = dash_sigma = 1.0
block_size = 32, local_blocks = 4, local_window = 128 tokens (DA:899-901)
main scale = 1/sqrt(128) (DA:937)
```
Also sets `_PAGE_SCOPE_PREFILL = True` (MD:333). `dash_router='entmax_topk_perlayer'`,
`sparse_topk_blocks=16`, `sparse_init_block=1`, `dash_prior_floor`, `dash_scaling_factor` are **not used**
on this path (no "init/sink block" is forced; only the local window and routed blocks).

### 3.2 Main q/k/v
```
h  = input_layernorm(x)                               # RMSNorm, as v1
q  = Wq h ; k = Wk h ; v = Wv h   -> [10,S,128]       # no bias, no qk-norm (MD:162-164)
q,k = rope_page2d(q,k, tok_pos, page_pos)            # §2
cache.append(k, v)                                    # full cache, no ring (MU:1149-1153)
```

### 3.3 Router index q/k (`_router_qk`, MD:365-377) — uses `dash_router.*`, NOT `indexer.*`
```
hs = nan_to_num(h)                                    # h = the layer's normed input (same as attention input)
iq = q_norm(Wq_r hs)   ik = k_norm(Wk_r hs)           # [S,128] each, 1 head
   q_norm/k_norm = Gemma RMSNorm: x*rsqrt(mean(x^2)+1e-6) * (1 + w), fp32, clamp ±1e4 (DA:840-855)
iq,ik = NEOX rope with the router's cos/sin (cos[..., :128] == whole thing)
iq,ik = fp32, nan->0
```
Router rope (`_dash_indexer_rope`, MD:126-143): `sparse_rope_mode=page2d` and `index_dim(128) ==
indexer.head_dim(128)` ⇒ `indexer.rotary_emb_2d(x, tok_pos, page_pos)` built from `sparse_rope_tok_theta=1e4,
page_theta=256, page_pairs=1, sparse_rope_tok_freq_mode=resample, page_mod=2` (SB:2330-2347) — **numerically
identical to the main page2d rope**; reuse the same cos/sin table. (cos/sin are cast to the activation dtype,
bf16 in the reference, before being applied.)
`ik` for all prefill tokens is stored per layer in `cache._dash_idx_k` (MD:441-444) — needed for decode.

### 3.4 Block scores, stage0 = head_cls (`_block_scores_infer`, MD:381-417)
```
nblk = ceil(S/32); pad ik with zero rows to nblk*32, real[] marks valid rows
for each block c:
   a_j   = clamp( (ik_j · head_cls) / sqrt(128), ±1e4 )        j in block, valid only
   w_j   = softmax_j(a_j) over valid j  (max over valid; denom clamp_min 1e-6)
   kbar_c = Σ_j w_j ik_j                                       # [nblk,128]
score[i,c] = clamp( (iq_i · kbar_c) / sqrt(128), ±1e4 )
keep[i,c]  = (c <= floor(pos_i/32) - 4) and block c has ≥1 real row
score[i,c] = -1e9 (_SCORE_NEG) where !keep                    # "not remote"
```
`pos_i` = the 1-D `position_ids` (= i in prefill), **not** the page coordinates.
Remote ⇔ the whole block ends before `32*(qblk-4)+31`. Because local_window=128 is token-level, every key
`<= pos` is covered by remote-candidate blocks ∪ local window (no gap).

### 3.5 Page-aware candidate domain (`_dash_page_ids` MD:96-105 → `_build_page_aware_ids` SB:2404-2502;
`_dash_block_cand` MD:108-123 → `_build_kl_candidate_mask` SB:2706-2737 → SB:2629-2649)
`image_bucket_constraint=True`:
```
cum = cumsum(ids==PAGE); last_img; prompt_pages = cum[last_img]; marker_mode = prompt_pages>0
img_page[t] = IMG ? (marker_mode ? cum[t]-1 : (running_img_idx // 273)) : -1
raw_page = cum-1 ; resp_page = raw_page - prompt_pages
q_page[t] = t<=last_img ? raw_page[t] : resp_page[t]      # PAGE_SCOPE_PREFILL=1
q_page[t] = -1 if q_page[t] < 0
cand[i,c] = (q_page[i] >= 0) and any_{j in block c} img_page[j] == q_page[i]
if not any_c cand[i,c]:  cand[i,:] = True                 # per-query fallback
```
Single page: BOS → -1 (all blocks), `<PAGE>`/IMG → page 0 ⇒ only blocks containing page-0 image tokens,
instruction text → -1 ⇒ all blocks. (`dash_image_only` is not consulted in prefill; the candidate set is
image-only through `_build_kl_candidate_mask` regardless.)

### 3.6 Routing (`_infer_route_select`, MD:338-362), per query row
```
remote = score > -5e8
cand   = remote & cand_page ;  if !any(cand): cand = remote
x      = cand ? score*idx_scale : -1e4                         # nan/inf -> -1e4
M      = min(topm=16, nblk)
xv, xi = topk(x, M)            (descending)
prior  = entmax15(xv)          over the M entries:
           X = (xv - max(xv)) / 2
           sort desc; rho=1..M; mean=cumsum(X)/rho; mean_sq=cumsum(X^2)/rho
           ss = rho*(mean_sq-mean^2); delta=(1-ss)/rho; tau=mean-sqrt(max(delta,0))
           support = max(1, #(tau <= Xsorted)); tau* = tau[support-1]
           prior = max(X - tau*, 0)^2                     (DA:772-800)
K'     = min(12, M);   keep first K' slots of (xi, prior)
valid  = cand[xi]                                          # non-candidates / non-remote slots invalid
logp   = log(max(prior, 1e-4))
mu     = mean(logp over valid slots)  (count clamped ≥1)
bias   = valid ? (logp - mu)/sigma : 0                     # NOT clamped in the inference path
```
Selected blocks = `xi[valid]` (≤12), each with a scalar bias shared by all heads and all 32 tokens.
Entmax-zero blocks still get read with bias `log(1e-4) - mu`. Ties in topk among -1e4 entries are irrelevant
(they are invalid).

### 3.7 Attention over the selection (`DashAttention._forward_dense`, DA:1230-1260; CPU path)
Mask, not gather. For query i, key j (both absolute indices):
```
local_ij = (j <= i) and (j > i - 128)
sel_ij   = block(j) is a valid selected block of row i
keep_ij  = (j <= i) and (local_ij or sel_ij)
m_ij     = local_ij ? 0 : (sel_ij ? bias[i, block(j)] : -1e9)          # local wins on overlap
out_i    = softmax_j( q_i·k_j / sqrt(128) + m_ij ) · v                   # per head, one softmax over the union
```
The additive mask is cast to the q dtype (bf16 in the reference) before SDPA (DA:1257) — bias is bf16-rounded.
On CUDA the Triton e2e kernel (DA:1207-1227) implements the same read set/bias; on CPU `q.is_cuda` is False
(MD:447) so dense is the reference. Then `o_proj`, residual, post-norm, MLP/MoE exactly as v1 (SB:3693-3716).

Early queries (pos < 128) have no remote block ⇒ pure causal over the local window (= full causal there).

---

## 4. DASH in DECODE (q_len == 1)

Dispatch (MD:239-241): if `_INFER` and `layer_idx in cache._dash_idx_k` ⇒ `_dash_decode_forward` (MD:484-531).
**No ring buffer**: `_ring_window` (CF 128) and `sliding_window` are set to None for the whole infer
(MU:1149-1153); the KV cache keeps every token.

Per step, with fed token `x` at absolute position `pos = S + n_gen - 1` (MU:1188-1193):

1. `q,k,v` projections; coords from the state machine §2.4 (advanced once per layer per step; all layers
   carry identical state); rope page2d; `k_all,v_all = cache.append(k,v)` (length `L = pos+1`).
2. Router: `iq, ik_new = _router_qk(h, same cos/sin)`; `ik = concat(cached ik, ik_new)` and store back
   (MD:515-517). Block summaries `kbar` are **recomputed over all L router keys every step** (including the
   partial last block; padding rows excluded) — implement incrementally: full blocks are immutable.
3. `score[c]` = §3.4 with `qblk = pos // 32`, remote iff `c <= qblk - 4`.
4. Candidate mask (`_uocr_dash_decode_allow`, MU:482-528), used because `router.image_only` is True
   (MD:521-526):
   - recorded at prefill (MU:499-513): per prefill block `bmin/bmax` = min/max image page of image tokens in
     the block (−1 if none), `pp` = markers up to the last image (1), `mcum` = total prompt markers (1).
     image page = `cum-1` when `cum>0` (else `(img_run-1)//273`).
   - each decode step: `mcum += (x == 128827)`; `qp = mcum - 1 - pp` (−1 if <0);
     `allow[c] = (c < n_prefill_blocks) and bmin[c] >= 0 and bmin[c] <= qp <= bmax[c]`;
     if `qp < 0` or no such block ⇒ `allow[:] = True` (open row). Blocks made of generated tokens are only
     candidates in the open-row case.
   - Before the first generated `<PAGE>` is fed, qp = −1 ⇒ all remote blocks are candidates. From the step
     where `<PAGE>` itself is the fed token, qp = 0 ⇒ only page-0 image blocks.
5. `_infer_route_select` (§3.6) with this candidate row.
6. `_decode_attend` (MD:455-481), fp32:
```
keep_j = (j <= pos) and ( j > pos-128  or  block(j) selected&valid )
m_j    = (j > pos-128) ? 0 : bias[block(j)]       (local wins)
out    = softmax_j( q·k_j / sqrt(128) + m_j ) · v_j
```
   i.e. the read set = last 128 tokens (including the current one) ∪ ≤12 remote blocks (each fully in the
   past, so causal is a no-op for them). Newly generated tokens are attended only through the local window
   unless they fall in a selected remote block (only possible in open-row mode or never once a page is set,
   since generated blocks are not in `allow`).

---

## 5. Generation loop (`_uocr_generate`, MU:1125-1205)

```
ids = input_ids; S = len(ids)
expected_pages = count(ids == 128827)          # 1 for one page (MU:1141); guard on
prefill: model(ids, position_ids=arange(S)), keep only last hidden (MU:1162-1165)
for step in range(max_new_tokens):             # = max_length(131072) - S   (MU:1089)
    logits = lm_head(final_norm(h_last)).float()        # untied lm_head
    if ngram_size>0: no-repeat-ngram (default size 0 => OFF; window default 100) (MU:1168-1169, 608-623)
    page_count_guard(logits)                             # MU:583-605, below
    nxt = argmax(logits)                                 # temperature 0
    if nxt == eos (tokenizer.eos_token_id, =1): stop, finish='stop'  (EOS not appended)
    gen.append(nxt)
    pos = S + len(gen) - 1
    if pos + 1 >= 131072: stop
    h_last = model([nxt], position_ids=[pos])           # decode step §4
text = tokenizer.decode(gen, skip_special_tokens=False)  # <PAGE> tokens kept in the text
```
PageCountGuard state `{marker_cum, scanned, page_len}` over **generated** ids only (MU:583-605):
```
update: new tail = gen[scanned:]; if tail contains PAGE: marker_cum += count; page_len = #tokens after last PAGE in tail
        else page_len += len(tail)
if marker_cum > expected:            logits = -inf except EOS
elif 20000 <= page_len:              logits = -inf except (PAGE if marker_cum < expected else EOS)
elif marker_cum < expected:          logits[EOS] = -inf
```
So for one page: EOS is banned until the model emits `<PAGE>`; after a 2nd `<PAGE>` only EOS is allowed; a
page longer than 20000 tokens is force-terminated. Post-processing for `save_results` only (MU:1207-1228):
split on `<PAGE>`, strip, `<|ref|>/<|det|>` handling, coordinates /999 — same as v1.

---

## 6. Other numerically relevant points

- RMSNorm (input/post/final): standard, weight × normalized, eps 1e-6, fp32 then cast (SB:216-230) — as v1.
  Router norms differ: **Gemma (1+w)** form (DA:840-855, SB:1918-1932).
- MoE (SB:522-657, 681-748): layer 0 dense MLP (inter 6848); layers 1-11: 64 routed experts (inter 896),
  softmax gate in fp32, greedy top-6 unsorted, `norm_topk_prob=False`, `routed_scaling_factor=1.0`,
  + 2 shared experts as one MLP of inter 1792. SiLU-GLU. Identical to v1.
- Embedding: `embed_tokens` 129280×1280; image features overwrite IMG positions — as v1.
- `max_position_embeddings` 131072 (v1 32768); irrelevant to page2d.
- Main attention scale `1/sqrt(128)`, no attention bias, no softcap.
- The `indexer.{q_proj,k_proj,q_norm,k_norm}` tensors are loaded but not used by the DASH inference path;
  only its helper methods (page ids, rope tables) are used. They can be skipped in the converter
  (or kept for a future "stage0=indexer" mode, which would use block-max of token-level `iq·ik` with the
  *router* projections, MD:418-430 — note no 1/sqrt(d) there).

---

## 7. Pseudo-code for the engine (per DASH layer)

```
// tables built once per request
tok_pos[S], page_pos[S], pf_cnt[], n_pg        // §2.3
cos,sin[S][128]  from inv = [10000^(-i/63) i<63 ; 1.0], angle = [tok*inv_tok, (page&1)*1.0], dup halves
q_page[S], img_page[S]                          // §3.5 (prefill), bmin/bmax/pp/mcum (§4)

prefill(layer):
  h = rmsnorm(x)
  q,k,v = proj(h); rope_neox(q,k,cos,sin); cache_kv.append(k,v)
  iq = gemma_rms(Wqr h)*(1+wq); ik = gemma_rms(Wkr h)*(1+wk); rope_neox(iq,ik,cos,sin)
  cache_ik.append(ik)
  kbar[c] = Σ softmax_j(ik_j·head_cls/√128) ik_j
  for each query i:
     s[c] = (c <= i/32-4) ? clamp(iq_i·kbar[c]/√128) : NEG
     cand[c] = remote && pageCand(i,c); if none: cand = remote
     x[c] = cand ? s[c]*11.3137 : -1e4
     top16 -> entmax15 -> first 12 -> valid=cand -> bias=(log(max(p,1e-4)) - mean_valid)/1
     attn_i = softmax over {j in (i-128, i]} (bias 0) ∪ {j in selected valid blocks} (bias_b)
  out = o_proj(attn); residual; mlp/moe as v1

decode(layer, fed token x, pos):
  advance (tok,page) state (§2.4); cos/sin for that single coordinate
  same projections/rope; append k,v,ik; recompute/extend kbar
  s[c] for qblk=pos/32; cand = allow (§4.4); route (§3.6); attend (§4.6)
```
Complexity note: a sparse implementation should gather the ≤12 blocks (≤384 keys) + 128 local keys per
query instead of the dense masked softmax; results are identical up to the masked -1e9 entries
(exp underflows to exactly 0).

---

## 8. Ambiguities / risks (ranked)

1. **Rope theta**: the page2d path uses theta 1e4 with 63 *resampled* pairs + 1 page pair; the 500000/YaRN
   config is a decoy for the default path. Using YaRN here will break everything. YaRN is only in the
   no-`<PAGE>` fallback (and its low/high values above should be re-verified against transformers 4.52).
2. **Decode state machine** depends on the *fed* token; the engine must update rope page/tok, `mcum` and
   the guard from the same token stream. Page clamping at `n_pg-1`; tok resets to `pf_cnt[page]` on `<PAGE>`.
3. **Candidate mask asymmetry**: prefill uses `_build_kl_candidate_mask` (image-only always, page scope on),
   decode uses `_uocr_dash_decode_allow` (prefill blocks only, bmin/bmax range test — a block whose min/max
   page range *spans* qp counts even if the exact page is absent; irrelevant for 1 page).
4. **bf16 rounding**: reference casts rope cos/sin to bf16 before applying, the router norms output bf16
   before rope, and the prefill additive bias mask is bf16 (DA:1257) whereas decode is fp32 (MD:475-481).
   Routing (top-k) can flip on near ties — expect small divergences vs. a pure-fp32 engine.
5. **Entmax over top-M only** (M=16) then hard K=12: entmax renormalises over the window, so the bias depends
   on M; `dash_topm` must be 16, K from `dash_topk_blocks` (12), not `sparse_topk_blocks` (16).
6. `_SCORE_NEG` (−1e9) vs `_ENTMAX_NEG` (−1e4): non-candidates enter entmax as −1e4 (scaled scores of real
   candidates are within ±1.1e5 in principle, clamp ±1e4 before scaling). Keep these exact sentinels.
7. `<PAGE>` must be a single token (128827) in the tokenizer; BOS is hard-coded 0, EOS from tokenizer (1).
8. Head_cls summaries are recomputed each decode step including the partial current block; but that block
   is never remote (c > qblk-4), so only full blocks matter — safe to cache.
9. `sparse_init_block=1` is NOT honoured by this path (no forced sink block). `dash_image_only` only gates the
   decode candidate mask.
10. Decode `position_ids` (1D) are still used for block remoteness and the local window; page coords only for
    rope.
