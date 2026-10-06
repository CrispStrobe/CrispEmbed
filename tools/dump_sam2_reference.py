#!/usr/bin/env python3
"""Dump SAM 2.1 reference tensors (PyTorch on the CPU) for tests/test_sam2_diff.cpp.

    python tools/dump_sam2_reference.py --sam2-source SAM2_CHECKOUT --checkpoint sam2.1_hiera_tiny.pt \
        --image photo.png --point 888,884 [--point X,Y:LABEL ...] [--box x0,y0,x1,y1] --output ref.gguf

Runs on the CPU on purpose: PyTorch 2.7 on Apple MPS computes Hiera's pooled
query wrongly (max_pool2d on a strided view), so MPS is not a reference.

Tensors (float32, memory order as the runtime keeps them: channel-last maps):
  input_image             [1,3,1024,1024]   resized and normalised, as the encoder takes it
  hiera.patch             [256,256,96]      patch embedding plus position embedding
  hiera.block.NN          [H,W,C]           output of every trunk block
  neck.N                  [H,W,256]         FPN outputs, finest first (before conv_s0/conv_s1)
  image_embed             [64,64,256]       last level plus no_mem_embed
  high_res_0, high_res_1  [256,256,32], [128,128,64]
  point_coords            [N,2]             in the 1024 frame, box corners first
  point_labels            [N]               as float
  mask_logits             [4,256,256]       all four output tokens, raw
  iou                     [4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_point(text):
    xy, _, label = text.partition(":")
    x, y = (float(v) for v in xy.split(","))
    return x, y, int(label) if label else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam2-source", type=Path, required=True, help="checkout of facebookresearch/sam2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--point", action="append", default=[], help="X,Y[:LABEL] in photo pixels; label 1 object, 0 background")
    parser.add_argument("--box", help="x0,y0,x1,y1 in photo pixels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import gguf
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, str(args.sam2_source.resolve()))
    from sam2.build_sam import build_sam2

    torch.set_grad_enabled(False)
    model = build_sam2(args.config, str(args.checkpoint), device="cpu", apply_postprocessing=False).eval()
    size = model.image_size
    rgb = np.array(Image.open(args.image).convert("RGB"))
    height, width = rgb.shape[:2]
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)[None]
    image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = (image - mean) / std

    tensors = {"input_image": image.numpy()}
    trunk = model.image_encoder.trunk
    hooks = [trunk.patch_embed.register_forward_hook(
        lambda m, i, o: tensors.__setitem__("hiera.patch", (o + trunk._get_pos_embed(o.shape[1:3]))[0].numpy()))]
    for n, block in enumerate(trunk.blocks):
        hooks.append(block.register_forward_hook(lambda m, i, o, n=n: tensors.__setitem__(f"hiera.block.{n:02d}", o[0].numpy())))
    out = model.image_encoder(image)
    for n, feature in enumerate(out["backbone_fpn"]):
        tensors[f"neck.{n}"] = feature[0].permute(1, 2, 0).contiguous().numpy()
    backbone = model.forward_image(image)
    _, feats, _, sizes = model._prepare_backbone_features(backbone)
    feats[-1] = feats[-1] + model.no_mem_embed
    maps = [f.permute(1, 2, 0).reshape(1, -1, s[0], s[1]) for f, s in zip(feats, sizes)]
    embed, high0, high1 = maps[2], maps[0], maps[1]
    tensors["image_embed"] = embed[0].permute(1, 2, 0).contiguous().numpy()
    tensors["high_res_0"] = high0[0].permute(1, 2, 0).contiguous().numpy()
    tensors["high_res_1"] = high1[0].permute(1, 2, 0).contiguous().numpy()

    coords, labels = [], []
    if args.box:
        x0, y0, x1, y1 = (float(v) for v in args.box.split(","))
        coords += [[x0, y0], [x1, y1]]
        labels += [2, 3]
    for text in args.point:
        x, y, label = parse_point(text)
        coords.append([x, y])
        labels.append(label)
    if not coords:
        raise SystemExit("give at least one --point or a --box")
    scaled = torch.tensor(coords, dtype=torch.float32) / torch.tensor([width, height], dtype=torch.float32) * size
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    sparse, dense = model.sam_prompt_encoder(points=(scaled[None], label_tensor[None]), boxes=None, masks=None)
    logits, iou, _, _ = model.sam_mask_decoder.predict_masks(
        image_embeddings=embed, image_pe=model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, repeat_image=False, high_res_features=[high0, high1])
    tensors["point_coords"] = scaled.numpy()
    tensors["point_labels"] = label_tensor.float().numpy()
    tensors["mask_logits"] = logits[0].numpy()
    tensors["iou"] = iou[0].numpy()

    writer = gguf.GGUFWriter(str(args.output), "sam2-reference")
    writer.add_string("sam2.reference.image", args.image.name)
    writer.add_uint32("sam2.reference.width", width)
    writer.add_uint32("sam2.reference.height", height)
    for name, array in tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{args.output}: {len(tensors)} tensors; iou {np.round(tensors['iou'], 4).tolist()}")


if __name__ == "__main__":
    main()
