#!/usr/bin/env python3
"""Step-by-step PosFormer decoder dump for diff testing.
Runs greedy decode manually, dumping logits at each step."""

import sys
import os
import torch
import numpy as np
from functools import partial

sys.path.insert(0, '/mnt/volume1/PosFormer-fresh')
from PIL import Image
from einops import rearrange
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

    # Use em_0 (diverges between C++ and PyTorch)
    img_path = '/mnt/storage/crohme_test_images/off_image_test/18_em_0_0.bmp'
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    if img_np.mean() > 0.5:
        img_np = 1.0 - img_np
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    img_mask = torch.zeros(1, img_t.shape[2], img_t.shape[3], dtype=torch.long)

    print(f"Image: {img_path}")
    print(f"Shape: {img_t.shape}")

    # Encoder
    with torch.no_grad():
        feature, mask = model.encoder(img_t, img_mask)
    print(f"Encoder output: {feature.shape}")

    # Save encoder output (flatten h,w -> n_enc)
    feat_flat = rearrange(feature, "b h w d -> b (h w) d")
    feat_flat.numpy().astype(np.float32).tofile(f"{OUT_DIR}/enc_output.f32")
    print(f"Encoder flat: {feat_flat.shape}")

    h_enc = feature.shape[1]

    # Manual greedy decode
    decoder = model.decoder
    D = hp['d_model']
    V = len(vocab)

    # Step through one token at a time
    SOS = vocab.SOS_IDX  # 1
    EOS = vocab.EOS_IDX  # 2
    PAD = vocab.PAD_IDX  # 0

    tokens = [SOS]
    max_steps = 30

    for step in range(max_steps):
        input_ids = torch.LongTensor([tokens])
        with torch.no_grad():
            logits, attn = decoder(feature, mask, input_ids)
        # logits: [1, len, V]
        step_logits = logits[0, -1, :].numpy()  # last position
        step_logits.astype(np.float32).tofile(f"{OUT_DIR}/step{step:03d}_logits.f32")

        best = int(logits[0, -1, :].argmax())
        token_str = vocab.idx2word.get(best, f"<{best}>")

        # Also dump top-5 tokens
        top5 = torch.topk(logits[0, -1, :], 5)
        top5_str = ", ".join([
            f"{vocab.idx2word.get(int(idx), f'<{int(idx)}>')}({val:.2f})"
            for val, idx in zip(top5.values, top5.indices)
        ])
        print(f"Step {step}: token={best} '{token_str}' | top5: {top5_str}")

        if best == EOS or best == PAD:
            print(f"  [EOS at step {step}]")
            break
        tokens.append(best)

    result = ' '.join([vocab.idx2word[t] for t in tokens[1:]])
    print(f"\nFinal: {result}")
    print(f"Tokens: {tokens[1:]}")

    with open(f"{OUT_DIR}/step_decode_result.txt", "w") as f:
        f.write(result + "\n")
        f.write(str(tokens[1:]) + "\n")


if __name__ == "__main__":
    main()
