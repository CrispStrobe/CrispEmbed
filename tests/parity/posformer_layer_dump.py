#!/usr/bin/env python3
"""Dump PosFormer decoder layer-by-layer intermediates for diff testing.

Runs pure L2R greedy decode one step at a time, dumping:
- After each decoder layer: hidden state, cross-attention weights
- ARM inputs/outputs when active
- Final logits per step
"""

import sys
import os
import torch
import numpy as np
from functools import partial
from einops import rearrange

sys.path.insert(0, '/mnt/volume1/PosFormer-fresh')
from PIL import Image
from Pos_Former.model.posformer import PosFormer
from Pos_Former.datamodule import vocab

OUT_DIR = "/mnt/volume1/posformer_ref"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = torch.load(
        '/mnt/volume1/PosFormer-fresh/lightning_logs/version_0/checkpoints/best.ckpt',
        map_location='cpu', weights_only=False
    )
    hp = ckpt['hyper_parameters']
    model = PosFormer(
        d_model=hp['d_model'], growth_rate=hp['growth_rate'],
        num_layers=hp['num_layers'], nhead=hp['nhead'],
        num_decoder_layers=hp['num_decoder_layers'],
        dim_feedforward=hp['dim_feedforward'], dropout=0.0,
        dc=hp['dc'], cross_coverage=hp['cross_coverage'],
        self_coverage=hp['self_coverage']
    )
    sd = {k[6:]: v for k, v in ckpt['state_dict'].items()
          if k.startswith('model.') and 'posdecoder' not in k}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Use em_0 (21 tokens, diverges at step 6)
    img_path = '/mnt/storage/crohme_test_images/off_image_test/18_em_0_0.bmp'
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    if img_np.mean() > 0.5:
        img_np = 1.0 - img_np
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    img_mask = torch.zeros(1, img_t.shape[2], img_t.shape[3], dtype=torch.long)

    # Encoder
    with torch.no_grad():
        feature, mask = model.encoder(img_t, img_mask)
    print(f"Encoder: {feature.shape}")

    # Save encoder output flattened
    feat_flat = rearrange(feature, "b h w d -> b (h w) d")
    feat_flat.numpy().astype(np.float32).tofile(f"{OUT_DIR}/enc_output.f32")

    D = hp['d_model']
    V = len(vocab)
    decoder = model.decoder

    # Hook into decoder layers to capture intermediates
    dec_model = decoder.model  # TransformerDecoder
    arm_module = dec_model.arm

    # We'll instrument each decoder layer to dump intermediates
    layer_outputs = {}  # will be populated per forward call

    SOS = vocab.SOS_IDX
    EOS = vocab.EOS_IDX
    tokens = [SOS]
    max_steps = 10  # only first 10 steps for debugging

    for step in range(max_steps):
        input_ids = torch.LongTensor([tokens])

        # Hook into self-attention output, cross-attention output, FFN output per layer
        hooks = []
        step_data = {}

        def make_layer_hook(layer_idx):
            def hook(module, input, output):
                # output is (tgt, attn) tuple
                tgt, attn = output
                # tgt shape: [l, b, d] (seq-first)
                last_hidden = tgt[-1, 0, :].detach().numpy().copy()
                step_data[f'layer{layer_idx}_output'] = last_hidden
                if attn is not None:
                    # attn shape: [b*nhead, t, n_enc]
                    last_attn = attn[:, -1, :].detach().numpy().copy()
                    step_data[f'layer{layer_idx}_ca_weights'] = last_attn
            return hook

        def make_sa_hook(layer_idx):
            """Hook after self-attention + LN (norm1)"""
            def hook(module, input, output):
                # norm1 output: [l, b, d]
                step_data[f'layer{layer_idx}_after_sa'] = output[-1, 0, :].detach().numpy().copy()
            return hook

        def make_ca_hook(layer_idx):
            """Hook after cross-attention + LN (norm2)"""
            def hook(module, input, output):
                # norm2 output: [l, b, d]
                step_data[f'layer{layer_idx}_after_ca'] = output[-1, 0, :].detach().numpy().copy()
            return hook

        for li, layer in enumerate(dec_model.layers):
            h = layer.register_forward_hook(make_layer_hook(li))
            hooks.append(h)
            h2 = layer.norm1.register_forward_hook(make_sa_hook(li))
            hooks.append(h2)
            h3 = layer.norm2.register_forward_hook(make_ca_hook(li))
            hooks.append(h3)

        # Hook ARM
        arm_data = []
        if arm_module is not None:
            orig_forward = arm_module.forward
            def arm_hook(prev_attn, key_padding_mask, h, curr_attn, tgt_vocab):
                result = orig_forward(prev_attn, key_padding_mask, h, curr_attn, tgt_vocab)
                # Save the ARM output for the last timestep only
                # result shape: [b*nhead, t, n_enc]
                last_result = result[:, -1, :].detach().numpy().copy()
                arm_data.append({
                    'bias': last_result,
                    'prev_attn_last': prev_attn[:, -1, :].detach().numpy().copy() if prev_attn.dim() == 3 else prev_attn.detach().numpy().copy(),
                    'curr_attn_last': curr_attn[:, -1, :].detach().numpy().copy() if curr_attn.dim() == 3 else curr_attn.detach().numpy().copy(),
                })
                return result
            arm_module.forward = arm_hook

        with torch.no_grad():
            logits_all, _ = decoder(feature, mask, input_ids)

        # Restore ARM
        if arm_module is not None:
            arm_module.forward = orig_forward

        # Remove hooks
        for h in hooks:
            h.remove()

        # Get logits for last position
        step_logits = logits_all[0, -1, :].numpy()
        best = int(logits_all[0, -1, :].argmax())
        token_str = vocab.idx2word.get(best, f"<{best}>")

        # Dump everything
        step_logits.astype(np.float32).tofile(f"{OUT_DIR}/step{step:03d}_logits.f32")

        for key, val in step_data.items():
            if isinstance(val, np.ndarray):
                val.astype(np.float32).tofile(f"{OUT_DIR}/step{step:03d}_{key}.f32")

        for ai, ad in enumerate(arm_data):
            ad['bias'].astype(np.float32).tofile(f"{OUT_DIR}/step{step:03d}_arm{ai}_bias.f32")

        # Top-5
        top5_idx = np.argsort(step_logits)[-5:][::-1]
        top5_str = ", ".join([
            f"{vocab.idx2word.get(int(i), '?')}({step_logits[i]:.2f})"
            for i in top5_idx
        ])
        arm_str = f" ARM calls={len(arm_data)}" if arm_data else ""
        print(f"Step {step}: {best}='{token_str}' | top5: {top5_str}{arm_str}")

        if best == EOS or best == vocab.PAD_IDX:
            print(f"  [EOS at step {step}]")
            break
        tokens.append(best)

    result = ' '.join([vocab.idx2word[t] for t in tokens[1:]])
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
