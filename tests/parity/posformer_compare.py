#!/usr/bin/env python3
"""Compare PosFormer PyTorch vs C++ outputs on multiple images."""

import sys
import os
import subprocess
import torch
import numpy as np

sys.path.insert(0, '/mnt/volume1/PosFormer-fresh')
from PIL import Image
from Pos_Former.model.posformer import PosFormer
from Pos_Former.datamodule import vocab

POSFORMER_BIN = "/mnt/volume1/CrispEmbed-build/test-posformer"
POSFORMER_MODEL = "/mnt/storage/models/posformer-hw-f32.gguf"
LD_PATH = "/mnt/volume1/CrispEmbed-build/ggml/src"
IMAGE_DIR = "/mnt/storage/crohme_test_images/off_image_test"
GT_FILE = "/mnt/storage/Pytorch-HMER/test_caption.txt"


def load_gt():
    gt = {}
    with open(GT_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gt[parts[0]] = parts[1]
    return gt


def run_cpp(bmp_path):
    env = {**os.environ, "LD_LIBRARY_PATH": LD_PATH}
    try:
        r = subprocess.run(
            [POSFORMER_BIN, POSFORMER_MODEL, bmp_path],
            capture_output=True, text=True, timeout=120, env=env,
        )
        return r.stdout.strip()
    except:
        return "[ERROR]"


def main():
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

    gt = load_gt()

    test_images = [
        '18_em_0_0', '18_em_1_0', '18_em_2_0', '18_em_3_0',
        '18_em_4_0', '18_em_5_0', '18_em_10_0', '18_em_11_0',
        '18_em_15_0', '18_em_16_0',
        '20_em_0_0', '20_em_1_0', '20_em_10_0',
    ]

    match_count = 0
    total = 0

    for img_file in test_images:
        bmp_path = os.path.join(IMAGE_DIR, f"{img_file}.bmp")
        if not os.path.exists(bmp_path):
            continue

        # C++ output
        cpp_result = run_cpp(bmp_path)

        # PyTorch output
        img = Image.open(bmp_path).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.mean() > 0.5:
            img_np = 1.0 - img_np
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        img_mask = torch.zeros(1, img_t.shape[2], img_t.shape[3], dtype=torch.long)

        with torch.no_grad():
            hyps = model.beam_search(img_t, img_mask, beam_size=1, max_len=200,
                                      alpha=1.0, early_stopping=True, temperature=1.0)
        py_result = ' '.join([vocab.idx2word[t] for t in hyps[0].seq if t > 2])

        # Ground truth key (strip _0 suffix)
        gt_key = img_file[:-2] if img_file.endswith('_0') else img_file
        truth = gt.get(gt_key, "???")

        match = (cpp_result == py_result)
        match_count += match
        total += 1

        status = "MATCH" if match else "DIFF"
        gt_match_cpp = "Y" if cpp_result == truth else "n"
        gt_match_py = "Y" if py_result == truth else "n"

        print(f"\n{'='*70}")
        print(f"Image: {img_file}  [{status}]")
        print(f"  GT:     {truth}")
        print(f"  C++:    {cpp_result}  (gt={gt_match_cpp})")
        print(f"  PyTorch:{py_result}  (gt={gt_match_py})")

    print(f"\n{'='*70}")
    print(f"Match: {match_count}/{total}")


if __name__ == "__main__":
    main()
