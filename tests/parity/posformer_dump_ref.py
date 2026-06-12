#!/usr/bin/env python3
"""Dump PosFormer intermediate values for diff testing against C++."""

import sys
import torch
import numpy as np

sys.path.insert(0, '/mnt/volume1/PosFormer-fresh')
from PIL import Image
from Pos_Former.model.posformer import PosFormer
from Pos_Former.datamodule import vocab

OUT_DIR = "/mnt/volume1/posformer_ref"

def main():
    import os
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

    # Use test image
    img_path = '/mnt/storage/crohme_test_images/off_image_test/18_em_1_0.bmp'
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    if img_np.mean() > 0.5:
        img_np = 1.0 - img_np
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    print(f"Image shape: {img_t.shape}")
    print(f"Image range: [{img_t.min():.3f}, {img_t.max():.3f}]")

    # Save input
    img_np.astype(np.float32).tofile(f"{OUT_DIR}/input_image.f32")
    with open(f"{OUT_DIR}/input_meta.txt", "w") as f:
        f.write(f"{img_t.shape[3]} {img_t.shape[2]}\n")  # W H

    # 1. Encoder output
    img_mask = torch.zeros(1, img_t.shape[2], img_t.shape[3], dtype=torch.long)
    with torch.no_grad():
        feature, mask = model.encoder(img_t, img_mask)
    print(f"Encoder output: {feature.shape}")  # [1, n_enc, D]
    feature.numpy().astype(np.float32).tofile(f"{OUT_DIR}/encoder_output.f32")
    mask_np = mask.numpy()
    mask_np.astype(np.int32).tofile(f"{OUT_DIR}/encoder_mask.i32")

    # 2. Greedy decode with intermediate dumps
    # We'll hook into the decoder to capture layer-by-layer outputs and ARM values
    D = hp['d_model']
    nhead = hp['nhead']
    head_dim = D // nhead

    # Get word embedding and position encoding
    word_embed = model.decoder.word_embed
    pos_enc = model.decoder.pos_enc

    dec_model = model.decoder.model  # TransformerDecoder
    proj = model.decoder.proj  # output projection

    # Manual greedy decode to dump intermediates
    sos_token = 0  # <sos>
    eos_token = 1  # <eos>
    max_steps = 200

    tokens = []
    prev_token = sos_token

    # Pre-compute stuff
    memory = feature.transpose(0, 1)  # [n_enc, 1, D]
    memory_key_padding_mask = mask  # [1, n_enc]

    h = feature.shape[1]  # This isn't right - need spatial h
    # Get spatial dimensions from encoder
    enc_out_before_reshape = None
    # Hook into encoder to get spatial dims
    # Actually, the encoder's feature has already been flattened. Let me get h from the model.
    # The encoder stores self.feature and self.mask in forward()
    # Let me just check the model architecture
    print(f"Feature shape: {feature.shape}, mask shape: {mask.shape}")

    # The height h is stored implicitly. For DenseNet with 3 blocks and 2 transitions:
    # Each transition halves the spatial dims. Input h -> h/4 (after 2 halvings)
    # But with padding in conv layers it depends on the actual image dims.
    # Let's compute it from the feature count and the mask dimensions.
    n_enc = feature.shape[1]
    # mask is [1, h*w] where h and w are spatial dims of encoder output
    # We need to figure out h. Let me trace through the encoder.

    # Actually, let me hook into the encoder to get intermediate shapes
    # For now, let me use the mask to figure out spatial dims
    # The DenseNet encoder with growth_rate=24, 16 layers per block, 3 blocks
    # Initial conv: stride 2 -> /2
    # MaxPool: stride 2 -> /2
    # Transition 1: AvgPool(2) -> /2
    # Transition 2: AvgPool(2) -> /2
    # Total spatial reduction: /16
    # But there might be different padding...

    # Actually let me just get it from the encoder itself
    # Run encoder again with a hook to capture spatial dims
    spatial_h = None
    spatial_w = None

    def hook_fn(module, input, output):
        nonlocal spatial_h, spatial_w
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            spatial_h = output.shape[2]
            spatial_w = output.shape[3]

    # The last conv in the encoder is feat_proj (1x1 conv)
    # which is encoder.feature_proj
    handle = model.encoder.feature_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model.encoder(img_t, img_mask)
    handle.remove()

    if spatial_h is None:
        # Fallback: compute from n_enc
        # Try common aspect ratios
        for h_try in range(1, n_enc + 1):
            if n_enc % h_try == 0:
                w_try = n_enc // h_try
                if abs(h_try / w_try - img_t.shape[2] / img_t.shape[3]) < 0.5:
                    spatial_h = h_try
                    spatial_w = w_try
                    break

    print(f"Spatial dims: h={spatial_h}, w={spatial_w}, n_enc={n_enc}")
    with open(f"{OUT_DIR}/spatial_dims.txt", "w") as f:
        f.write(f"{spatial_h} {spatial_w}\n")

    # Now do greedy decode with hooks on ARM
    arm_module = dec_model.arm  # AttentionRefinementModule

    arm_calls = []
    arm_orig_forward = arm_module.forward

    def arm_hook_forward(prev_attn, key_padding_mask, h, curr_attn, tgt_vocab):
        result = arm_orig_forward(prev_attn, key_padding_mask, h, curr_attn, tgt_vocab)
        arm_calls.append({
            'prev_attn': prev_attn.detach().numpy().copy(),
            'curr_attn': curr_attn.detach().numpy().copy(),
            'result': result.detach().numpy().copy(),
            'tgt_vocab': tgt_vocab.detach().numpy().copy(),
        })
        return result

    arm_module.forward = arm_hook_forward

    # Run beam search with beam_size=1 (greedy)
    with torch.no_grad():
        hyps = model.beam_search(img_t, img_mask, beam_size=1, max_len=200,
                                  alpha=1.0, early_stopping=True, temperature=1.0)

    result_str = ' '.join([vocab.idx2word[t] for t in hyps[0].seq if t > 2])
    print(f"PyTorch result: {result_str}")
    print(f"Tokens: {[t for t in hyps[0].seq if t > 2]}")
    print(f"ARM was called {len(arm_calls)} times")

    with open(f"{OUT_DIR}/pytorch_result.txt", "w") as f:
        f.write(result_str + "\n")
        f.write(str([t for t in hyps[0].seq if t > 2]) + "\n")

    # Save ARM intermediate values
    for i, call in enumerate(arm_calls):
        call['prev_attn'].astype(np.float32).tofile(f"{OUT_DIR}/arm_call{i}_prev_attn.f32")
        call['curr_attn'].astype(np.float32).tofile(f"{OUT_DIR}/arm_call{i}_curr_attn.f32")
        call['result'].astype(np.float32).tofile(f"{OUT_DIR}/arm_call{i}_result.f32")
        with open(f"{OUT_DIR}/arm_call{i}_meta.txt", "w") as f:
            f.write(f"prev_attn shape: {call['prev_attn'].shape}\n")
            f.write(f"curr_attn shape: {call['curr_attn'].shape}\n")
            f.write(f"result shape: {call['result'].shape}\n")

    print(f"\nAll reference data saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
