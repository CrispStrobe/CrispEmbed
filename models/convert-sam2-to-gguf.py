#!/usr/bin/env python3
"""Convert a SAM 2.1 checkpoint (image mode) to GGUF for src/sam2.cpp.

Keeps what single-image segmentation needs: the Hiera image encoder, the FPN
neck, the prompt encoder and the mask decoder. The video-memory parts
(memory encoder and attention, object pointers) are dropped.

    python models/convert-sam2-to-gguf.py --checkpoint sam2.1_hiera_tiny.pt --output sam2.1-hiera-tiny.gguf [--dtype f16]

Needs torch, numpy and the gguf package; not the SAM 2 source. Tested with
sam2.1_hiera_tiny.pt; the architecture numbers below are those of that model
and are written as metadata so the runtime does not assume them.

Tensor layout (ggml ne, i.e. the numpy shape reversed):
  linear weights      [in, out]                 (torch [out, in] unchanged)
  1x1 convolutions    [in, out]                 (torch [out, in, 1, 1] squeezed)
  patch embedding     [7, 7, 3, 96]             (torch [out, in, kh, kw] unchanged; ggml_conv_2d layout)
  transposed convs    [in, out, kw, kh]         (torch [in, out, kh, kw] permuted to [kh, kw, out, in])
  position embedding  hiera.pos_small [7, 7, 96], hiera.pos_window [8, 8, 96], hiera.pos_interp [7, 256]:
                      the trunk's bicubic resize to the 256 x 256 token grid as a matrix, so the runtime
                      computes the embedding as interp @ small @ interp^T + tiled window, without bicubic code.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ARCH = "sam2"

# SAM 2.1 Hiera-tiny (configs/sam2.1/sam2.1_hiera_t.yaml).
HPARAMS = {
    "image_size": 1024,
    "embed_dim": 96,
    "num_heads": 1,
    "stages": [1, 2, 7, 2],
    "global_att_blocks": [5, 7, 9],
    "window_spec": [8, 4, 14, 7],
    "q_pool": 3,
    "neck_dim": 256,
    "fpn_top_down_levels": [2, 3],
    "decoder_depth": 2,
    "decoder_heads": 8,
    "decoder_mlp_dim": 2048,
    "num_mask_tokens": 4,
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def rename(key):
    for old, new in (
        ("image_encoder.trunk.", "hiera."),
        ("image_encoder.neck.convs.", "neck."),
        ("sam_prompt_encoder.", "prompt."),
        ("sam_mask_decoder.", "dec."),
    ):
        if key.startswith(old):
            key = new + key[len(old):]
    key = key.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
    # GGUF tensor names are limited to 63 characters.
    for old, new in SHORT:
        key = key.replace(old, new)
    return key


SHORT = (
    ("dec.transformer.", "dec.tf."),
    ("cross_attn_token_to_image", "t2i"),
    ("cross_attn_image_to_token", "i2t"),
    ("final_attn_token_to_image", "final_t2i"),
    ("output_hypernetworks_mlps", "hyper"),
    ("iou_prediction_head", "iou_head"),
    ("pe_layer.positional_encoding_gaussian_matrix", "gauss"),
)


def wanted(key):
    if key == "no_mem_embed":
        return True
    if key.startswith("sam_prompt_encoder.mask_downscaling"):
        return False  # mask prompts are not offered
    if key.startswith("sam_mask_decoder.pred_obj_score_head"):
        return False  # object score: video tracking only
    return key.startswith(("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder."))


def bicubic_matrix(source, target):
    """F.interpolate(..., mode="bicubic") along one axis as a target x source matrix."""
    eye = torch.eye(source)[None, None]
    return F.interpolate(eye, size=(target, source), mode="bicubic")[0, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32", help="type of linear and convolution weights")
    args = parser.parse_args()

    import gguf

    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("model", state)
    writer = gguf.GGUFWriter(str(args.output), ARCH)
    writer.add_string("general.name", "SAM 2.1 Hiera-tiny (image mode)")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source", "facebookresearch/sam2, " + args.checkpoint.name)
    for key, value in HPARAMS.items():
        if isinstance(value, list):
            writer.add_array(f"{ARCH}.{key}", value)
        else:
            writer.add_uint32(f"{ARCH}.{key}", value)
    writer.add_array(f"{ARCH}.image_mean", MEAN)
    writer.add_array(f"{ARCH}.image_std", STD)
    writer.add_float32(f"{ARCH}.stability_delta", 0.05)
    writer.add_float32(f"{ARCH}.stability_threshold", 0.98)

    grid = HPARAMS["image_size"] // 4
    count = 0
    for key, tensor in state.items():
        if not wanted(key):
            continue
        name = rename(key)
        t = tensor.detach().float()
        if name == "hiera.pos_embed":
            writer.add_tensor("hiera.pos_small", t[0].permute(1, 2, 0).contiguous().numpy())  # [7,7,96] -> ne [96,7,7]
            writer.add_tensor("hiera.pos_interp", bicubic_matrix(t.shape[-1], grid).numpy())  # ne [7, 256]
            count += 2
            continue
        if name == "hiera.pos_embed_window":
            writer.add_tensor("hiera.pos_window", t[0].permute(1, 2, 0).contiguous().numpy())  # ne [96, 8, 8]
            count += 1
            continue
        if t.dim() == 4 and t.shape[-2:] == (1, 1):
            t = t[:, :, 0, 0]
        elif name.startswith("dec.output_upscaling.") and t.dim() == 4:
            t = t.permute(2, 3, 1, 0)  # [in, out, kh, kw] -> [kh, kw, out, in]: ne [in, out, kw, kh]
        elif t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        array = t.contiguous().numpy()
        heavy = array.ndim >= 2 and min(array.shape) > 1 and "pos_" not in name and "token" not in name and "embed" not in name
        if args.dtype == "f16" and heavy:
            array = array.astype(np.float16)
        writer.add_tensor(name, array)
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{args.output}: {count} tensors, {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
