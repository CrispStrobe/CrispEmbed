#!/usr/bin/env python3
"""HMER parity test: PyTorch reference vs CrispEmbed GGUF.

Runs the original PyTorch HMER model on a synthetic test image (or a
user-provided image), dumps the encoder output and decoded token sequence,
then (optionally) compares against the CrispEmbed C++ inference.

Usage:
    # Step 1: Generate PyTorch reference output
    python tests/test_hmer_parity.py \
        --model-dir /mnt/storage/Pytorch-HMER/model \
        --dict /mnt/storage/Pytorch-HMER/dictionary.txt \
        --dump /mnt/storage/models/hmer-reference.npz

    # Step 2 (after C++ build): Compare
    python tests/test_hmer_parity.py \
        --gguf /mnt/storage/models/hmer-hw-f32.gguf \
        --reference /mnt/storage/models/hmer-reference.npz
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def load_dict(dict_path):
    tokens = {}
    with open(dict_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tokens[int(parts[-1])] = parts[0]
    return tokens


def create_test_image(width=128, height=64):
    """Create a synthetic test image with some stroke-like patterns."""
    img = np.ones((height, width), dtype=np.float32)  # white background

    # Draw a simple "2" shape using horizontal/vertical strokes
    cx, cy = width // 2, height // 2
    # Top horizontal
    img[cy - 15, cx - 10:cx + 10] = 0
    img[cy - 14, cx - 10:cx + 10] = 0
    # Right vertical (top half)
    img[cy - 15:cy, cx + 9] = 0
    img[cy - 15:cy, cx + 8] = 0
    # Middle horizontal
    img[cy, cx - 10:cx + 10] = 0
    img[cy + 1, cx - 10:cx + 10] = 0
    # Left vertical (bottom half)
    img[cy:cy + 15, cx - 10] = 0
    img[cy:cy + 15, cx - 9] = 0
    # Bottom horizontal
    img[cy + 14, cx - 10:cx + 10] = 0
    img[cy + 15, cx - 10:cx + 10] = 0

    return img


def run_pytorch_reference(model_dir, dict_path, test_image=None, dump_path=None):
    """Run PyTorch model and return encoder output + decoded tokens."""
    # Filter broken local torch
    _orig = sys.path[:]
    sys.path = [p for p in sys.path if '.local' not in p]
    import torch
    import torch.nn.functional as F
    sys.path = _orig

    # Add HMER repo to path for imports
    hmer_dir = str(Path(model_dir).parent)
    if hmer_dir not in sys.path:
        sys.path.insert(0, hmer_dir)

    from Densenet_torchvision import densenet121
    from Attention_RNN import AttnDecoderRNN

    # Load models
    enc_files = sorted(Path(model_dir).glob("encoder_*.pkl"))
    dec_files = sorted(Path(model_dir).glob("attn_decoder_*.pkl"))
    assert enc_files and dec_files, "Checkpoint files not found"

    encoder = densenet121()
    decoder = AttnDecoderRNN(256, 112, dropout_p=0.5)

    # Load weights (strip module. prefix)
    enc_sd = torch.load(str(enc_files[0]), map_location='cpu', weights_only=False)
    dec_sd = torch.load(str(dec_files[0]), map_location='cpu', weights_only=False)

    enc_sd_clean = {k.replace('module.', ''): v for k, v in enc_sd.items()}
    dec_sd_clean = {k.replace('module.', ''): v for k, v in dec_sd.items()}

    encoder.load_state_dict(enc_sd_clean, strict=False)
    decoder.load_state_dict(dec_sd_clean, strict=False)

    encoder.eval()
    decoder.eval()

    # Load dictionary
    worddicts_r = load_dict(dict_path)

    # Prepare input image
    if test_image is None:
        test_image = create_test_image()

    H, W = test_image.shape
    print(f"Test image: {W}x{H}")

    # Preprocessing: 2-channel (gray + mask), batch dim, /255 already done
    gray_tensor = torch.from_numpy(test_image).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    mask_tensor = torch.ones_like(gray_tensor)  # all 1s (no padding)
    input_tensor = torch.cat([gray_tensor, mask_tensor], dim=1)  # (1, 2, H, W)

    print(f"Input tensor: {input_tensor.shape}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Encoder
    with torch.no_grad():
        enc_output = encoder(input_tensor)  # (1, 1024, H', W')

    print(f"Encoder output: {enc_output.shape}")
    enc_h = enc_output.shape[2]
    enc_w = enc_output.shape[3]

    # Decoder (greedy, CPU)
    batch_size = 1
    maxlen = 48
    hidden_size = 256
    output_area = enc_w
    dense_input = enc_h

    decoder_input = torch.LongTensor([111])  # <sos>
    decoder_hidden = torch.zeros(1, 1, hidden_size)  # zeros for deterministic test
    decoder_attention = torch.zeros(1, 1, dense_input, output_area)
    attention_sum = torch.zeros(1, 1, dense_input, output_area)

    # Masks (no padding, all valid)
    h_mask = [dense_input]
    w_mask = [output_area]

    tokens = []
    with torch.no_grad():
        for step in range(maxlen):
            # Monkey-patch the decoder's forward to work on CPU (it has cuda() calls)
            # Instead, manually run the forward pass
            embedded = decoder.module.embedding(decoder_input).view(1, 256) if hasattr(decoder, 'module') else decoder.embedding(decoder_input).view(1, 256)
            dec = decoder.module if hasattr(decoder, 'module') else decoder

            hidden_view = decoder_hidden.view(1, hidden_size)
            st = dec.gru1(embedded, hidden_view)
            hidden1 = dec.hidden(st).view(1, 1, 1, 256)

            enc_trans = enc_output.permute(0, 2, 3, 1)  # (1, H', W', 1024)

            decoder_attention = dec.conv1(decoder_attention)
            attention_sum = attention_sum + decoder_attention
            attn_sum_trans = attention_sum.permute(0, 2, 3, 1)  # (1, H', W', 1)

            enc_out1 = dec.ua(enc_trans)      # (1, H', W', 256)
            attn_sum1 = dec.uf(attn_sum_trans) # (1, H', W', 256)

            et = hidden1 + enc_out1 + attn_sum1
            et_t = et.permute(0, 3, 1, 2)     # (1, 256, H', W')
            et_t = dec.conv_tan(et_t)

            # Mask (all 1s)
            et_mask = torch.ones(1, 1, dense_input, output_area)
            et_t = et_t * et_mask
            et_t = dec.bn1(et_t)
            et_t = torch.tanh(et_t)
            et_t = et_t.permute(0, 2, 3, 1)   # (1, H', W', 256)
            et_scalar = dec.v(et_t).squeeze(3)  # (1, H', W')

            # Softmax
            et_exp = torch.exp(et_scalar)
            et_exp = et_exp * torch.ones(1, dense_input, output_area)
            et_sum = et_exp.sum(dim=1).sum(dim=1)  # (1,)
            alpha = et_exp / (et_sum.unsqueeze(1).unsqueeze(2) + 1e-8)
            alpha_4d = alpha.unsqueeze(1)  # (1, 1, H', W')

            # Context
            ct = (alpha_4d * enc_output).sum(dim=2).sum(dim=2)  # (1, 1024)

            # GRU2
            hidden_next = dec.gru(ct, st)
            decoder_hidden = hidden_next.view(1, 1, hidden_size)

            # Output
            h2 = dec.hidden2(hidden_next)
            e2 = dec.emb2(embedded)
            c2 = dec.wc(ct)
            output = F.log_softmax(dec.out(h2 + e2 + c2), dim=1)

            topv, topi = torch.max(output, 1)
            tok = topi.item()

            if tok == 0:  # <eol>
                break
            tokens.append(tok)
            decoder_input = topi

    # Detokenize
    latex_tokens = [worddicts_r.get(t, f'<{t}>') for t in tokens]
    latex_str = ' '.join(latex_tokens)

    print(f"\nDecoded tokens ({len(tokens)}): {tokens}")
    print(f"LaTeX: {latex_str}")

    # Save reference
    if dump_path:
        np.savez(dump_path,
                 input_image=test_image,
                 encoder_output=enc_output.numpy(),
                 tokens=np.array(tokens, dtype=np.int32),
                 latex=latex_str)
        print(f"\nReference saved to {dump_path}")

    return enc_output.numpy(), tokens, latex_str


def main():
    p = argparse.ArgumentParser(description="HMER parity test")
    p.add_argument("--model-dir",
                   help="PyTorch model directory (encoder/decoder .pkl)")
    p.add_argument("--dict",
                   help="Path to dictionary.txt")
    p.add_argument("--dump",
                   help="Save reference output to .npz")
    p.add_argument("--image",
                   help="Optional: path to a grayscale test image (BMP/PNG)")
    p.add_argument("--gguf",
                   help="Path to GGUF model (for C++ comparison)")
    p.add_argument("--reference",
                   help="Path to reference .npz from --dump")
    args = p.parse_args()

    if args.model_dir and args.dict:
        test_img = None
        if args.image:
            from PIL import Image
            img = Image.open(args.image).convert('L')
            test_img = np.array(img, dtype=np.float32) / 255.0

        run_pytorch_reference(args.model_dir, args.dict,
                              test_image=test_img,
                              dump_path=args.dump)
    elif args.reference:
        ref = np.load(args.reference, allow_pickle=True)
        print(f"Reference: {ref['tokens']} → {ref['latex']}")
        print(f"Encoder shape: {ref['encoder_output'].shape}")
        # TODO: compare against C++ output
    else:
        p.print_help()


if __name__ == "__main__":
    main()
