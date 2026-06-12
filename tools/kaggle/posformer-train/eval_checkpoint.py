#!/usr/bin/env python3
"""Evaluate a PosFormer checkpoint on CROHME test with full beam search."""

import sys, os, torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/mnt/volume1/PosFormer-fresh')
from Pos_Former.model.posformer import PosFormer
from Pos_Former.datamodule import vocab

CKPT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/volume1/posformer-training/crohme/latest.ckpt'
TEST_CAPTION = '/mnt/storage/Pytorch-HMER/test_caption.txt'
TEST_IMG_DIR = '/mnt/volume1/crohme_test_images/off_image_test'

# Load checkpoint
print(f"Loading {CKPT}...")
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
hp = ckpt.get('hyper_parameters', {})
print(f"Epoch: {ckpt.get('epoch', '?')}, step: {ckpt.get('global_step', '?')}")
print(f"Hparams: d={hp.get('d_model')}, beam={hp.get('beam_size')}")

model = PosFormer(
    d_model=hp['d_model'], growth_rate=hp['growth_rate'],
    num_layers=hp['num_layers'], nhead=hp['nhead'],
    num_decoder_layers=hp['num_decoder_layers'],
    dim_feedforward=hp['dim_feedforward'], dropout=0.0,
    dc=hp['dc'], cross_coverage=hp['cross_coverage'],
    self_coverage=hp['self_coverage'],
)
# Load weights (strip 'model.' prefix from LitPosFormer)
sd = {}
for k, v in ckpt['state_dict'].items():
    if k.startswith('model.') and 'posdecoder' not in k:
        sd[k[6:]] = v
model.load_state_dict(sd, strict=False)
model.eval()

# Load GT
gt = {}
with open(TEST_CAPTION) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gt[parts[0]] = parts[1]

# Evaluate with different beam sizes
for beam_size in [1, 10]:
    match = 0
    total = 0
    errors = 0

    for key, truth in sorted(gt.items()):
        bmp = os.path.join(TEST_IMG_DIR, f'{key}_0.bmp')
        if not os.path.exists(bmp):
            continue

        img = Image.open(bmp).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.mean() > 0.5:
            img_np = 1.0 - img_np
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        img_mask = torch.zeros(1, img_t.shape[2], img_t.shape[3], dtype=torch.long)

        try:
            with torch.no_grad():
                hyps = model.beam_search(
                    img_t, img_mask, beam_size=beam_size,
                    max_len=200, alpha=1.0, early_stopping=True, temperature=1.0)
            pred = ' '.join([vocab.idx2word[t] for t in hyps[0].seq if t > 2])
        except Exception as e:
            errors += 1
            pred = ''

        norm_pred = ' '.join(pred.split())
        norm_truth = ' '.join(truth.split())
        total += 1
        if norm_pred == norm_truth:
            match += 1

        if total % 100 == 0:
            print(f"  beam={beam_size}: {match}/{total} ({100*match/total:.1f}%) ...", flush=True)

    print(f"\nbeam_size={beam_size}: {match}/{total} = {100*match/total:.1f}% (errors: {errors})")
    print()
