"""Memory-efficient BidirLM-Omni text+image parity test.

Loads HF BidirLMOmniVisionModel + BidirLMOmniTextModel separately, skipping
the audio_tower that BidirLMOmniModel loads, and reproduces the multimodal
forward manually:

  1. Vision tower → image_embeds + deepstack (3 slabs).
  2. Text encoder with masked_scatter image splice + position_ids from a
     local port of BidirLMOmniModel.get_rope_index.
  3. CrispEmbed encode_with_image_ids on the same inputs.

This avoids the ~10 GB BidirLMOmniModel RAM peak that
tests/test_bidirlm_image_text.py hits and runs in 3–4 min on a 16 GB Mac —
practical for iteration. Pass criteria mirror the full test: cosine
≥ 0.99 against bf16 for q8_0+ quants; q4_k settles at ~0.94 vs bf16
(see LEARNINGS.md "q4_k quantization cosine ceiling").

Usage:
  python tests/test_bidirlm_image_text_lite.py \
      --gguf "$CRISPEMBED_CACHE_DIR/bidirlm-omni-2.5b-q4_k.gguf" \
      --image "$CRISPEMBED_BIDIRLM_IMAGE" \
      --ref-dtype bf16
"""

import os, sys, argparse
os.environ.setdefault("HF_HOME", "/Volumes/backups/ai/huggingface-hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Volumes/backups/ai/huggingface-hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "python")

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from crispembed._binding import CrispEmbed
from crispembed.image import preprocess_image

p = argparse.ArgumentParser()
p.add_argument("--gguf", default="/tmp/bidirlm-vision-out/bidirlm-omni-2.5b-q4_k.gguf")
p.add_argument("--image", default="/tmp/cat.jpg")
p.add_argument("--ref-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
p.add_argument("--model", default="BidirLM/BidirLM-Omni-2.5B-Embedding")
args = p.parse_args()
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.ref_dtype]

print(f"Loading config + tokenizer ({args.model}) ...", flush=True)
config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

print("Loading shard...", flush=True)
sd = {}
try:
    raw = load_file(hf_hub_download(repo_id=args.model, filename="model.safetensors"))
except Exception:
    import json
    idx = json.load(open(hf_hub_download(repo_id=args.model, filename="model.safetensors.index.json")))["weight_map"]
    raw = {}
    for shard in set(idx.values()):
        raw.update(load_file(hf_hub_download(repo_id=args.model, filename=shard)))
print(f"  {len(raw)} tensors loaded", flush=True)

# --- Build vision tower from visual.* tensors ---
print("Building vision tower...", flush=True)
vis_cls = get_class_from_dynamic_module("modeling_bidirlm_omni.BidirLMOmniVisionModel", args.model)
vis = vis_cls(config.vision_config).to(DTYPE); vis.eval()
vis_sd = {k[len("visual."):]: v.to(DTYPE) for k, v in raw.items() if k.startswith("visual.")}
vis.load_state_dict(vis_sd, strict=False)
del vis_sd

# --- Build text encoder from language_model.* tensors ---
print("Building text encoder...", flush=True)
text_cls = get_class_from_dynamic_module("modeling_bidirlm_omni.BidirLMOmniTextModel", args.model)
tm = text_cls(config.text_config).to(DTYPE); tm.eval()
tm_sd = {k[len("language_model."):]: v.to(DTYPE) for k, v in raw.items() if k.startswith("language_model.")}
tm.load_state_dict(tm_sd, strict=False)
del tm_sd, raw

# --- Preprocess + tokenize ---
pv_np, gt_np = preprocess_image(args.image, model_name=args.model)
spatial_merge = config.vision_config.spatial_merge_size
n_merged = (gt_np[0,0]*gt_np[0,1]*gt_np[0,2]) // (spatial_merge**2)
pv = torch.from_numpy(pv_np).to(DTYPE)
gt = torch.from_numpy(gt_np).to(torch.long)

prompt = "a photograph of a cat"
prefix_ids = tok.encode(prompt, add_special_tokens=False)
ids = [config.vision_start_token_id] + [config.image_token_id]*int(n_merged) + [config.vision_end_token_id] + prefix_ids
input_ids = torch.tensor([ids], dtype=torch.long)
attn = torch.ones_like(input_ids)
print(f"input_ids len={len(ids)}, n_merged={int(n_merged)}, image_token_id={config.image_token_id}", flush=True)

# --- 1. Vision tower forward ---
print("Vision forward...", flush=True)
with torch.no_grad():
    image_embeds, deepstack_image_embeds = vis(pv, grid_thw=gt)
print(f"  image_embeds={tuple(image_embeds.shape)}, deepstack list len={len(deepstack_image_embeds)}", flush=True)

# --- 2. Build inputs_embeds with image splice ---
inputs_embeds = tm.embed_tokens(input_ids)  # (1, T, hidden)
image_mask = (input_ids == config.image_token_id)
expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
inputs_embeds = inputs_embeds.masked_scatter(expanded, image_embeds.to(inputs_embeds.dtype))

# --- 3. Build position_ids via the omni model's helper ---
# Reproduce get_rope_index inline (same logic as BidirLMOmniModel.get_rope_index).
def build_rope(input_ids, grid_thw, attn_mask, spatial_merge,
                image_token_id, vision_start_token_id):
    position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1],
                              dtype=torch.long, device=input_ids.device)
    image_index = 0
    for i, ids_row in enumerate(input_ids):
        ids_row = ids_row[attn_mask[i] == 1]
        input_tokens = ids_row.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images = (ids_row == image_token_id).sum().item()
        for _ in range(remain_images):
            ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens else len(input_tokens) + 1
            t, h, w = grid_thw[image_index]
            image_index += 1
            llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item() // spatial_merge, w.item() // spatial_merge
            text_len = ed_image - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h*llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed_image + llm_grid_t * llm_grid_h * llm_grid_w
            break  # only one image in this test
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(len(input_tokens) - st).view(1, -1).expand(3, -1) + st_idx)
        position_ids[..., i, attn_mask[i] == 1] = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    return position_ids

pos_ids = build_rope(input_ids, gt, attn, spatial_merge, config.image_token_id, config.vision_start_token_id)
print(f"  position_ids: {tuple(pos_ids.shape)}", flush=True)

# --- 4. Text encoder forward with deepstack injection ---
print("Text forward (with deepstack)...", flush=True)
visual_pos_masks = image_mask
with torch.no_grad():
    out = tm(input_ids=None,
              attention_mask=attn,
              position_ids=pos_ids,
              inputs_embeds=inputs_embeds,
              visual_pos_masks=visual_pos_masks,
              deepstack_visual_embeds=deepstack_image_embeds)
h = out.last_hidden_state[0].float()
m = attn[0].float()
hf_pooled = ((h * m[:, None]).sum(0) / m.sum().clamp(min=1)).cpu().numpy()
hf_pooled = hf_pooled / (np.linalg.norm(hf_pooled) + 1e-12)

# --- 5. CrispEmbed reference ---
print("CrispEmbed encode_with_image_ids...", flush=True)
ce = CrispEmbed(args.gguf, n_threads=4)
ce_v = ce.encode_with_image_ids(ids, pv_np, gt_np)

cos = float(np.dot(hf_pooled, ce_v))
print()
print(f"cosine(HF lite, CrispEmbed) = {cos:.6f}  (ref dtype={args.ref_dtype})")
print(f"  threshold for {args.ref_dtype}: {0.99 if args.ref_dtype != 'fp32' else 0.93}")
sys.exit(0 if cos > (0.99 if args.ref_dtype != 'fp32' else 0.93) else 1)
