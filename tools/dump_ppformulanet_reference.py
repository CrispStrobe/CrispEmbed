#!/usr/bin/env python3
"""Dump per-layer reference activations for PPFormulaNet-S encoder.

Loads the PyTorch HGNetv2+MBart model, runs inference on a test image,
captures intermediate activations at every architectural boundary via
forward hooks, and writes to a GGUF tensor archive.

The C++ crispembed_diff harness then loads that GGUF and compares each
tensor against what the ggml encoder graph produces. The first layer
where cos_min drops below 0.999 is where the bug lives.

Stages captured:
  input_image          (3, 384, 384)   F32 preprocessed RGB input
  stem_output          (48, H, W)      F32 after StemBlock
  stage{0-3}_output    (C, H, W)       F32 after each HG_Stage
  enc_output           (N, 2048)       F32 encoder flatten output
  proj_output          (N, 384)        F32 after enc_to_dec_proj
  dec_embed            (384,)          F32 first decoder token embedding
  dec_layer_{0,1}      (384,)          F32 after each decoder layer (step 0)
  logits_step0         (V,)            F32 logits at first decode step

Usage:
    python tools/dump_ppformulanet_reference.py \
        --checkpoint /mnt/storage/models/ppformulanet-s/checkpoints/formulanet.pt \
        --config /mnt/storage/models/ppformulanet-s/config.json \
        --tokenizer-dir /mnt/storage/models/ppformulanet-s/unimernet_tokenizer \
        --output /tmp/ppfn-ref.gguf

    python tools/dump_ppformulanet_reference.py \
        --checkpoint /mnt/storage/models/ppformulanet-s/checkpoints/formulanet.pt \
        --config /mnt/storage/models/ppformulanet-s/config.json \
        --tokenizer-dir /mnt/storage/models/ppformulanet-s/unimernet_tokenizer \
        --image /path/to/formula.png \
        --output /tmp/ppfn-ref.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


# ---------------------------------------------------------------------------
# HGNetv2 model definition (minimal, for loading weights)
# We need the actual model to run forward hooks.
# This reimplements just enough to load formulanet.pt and run inference.
# ---------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, use_act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.use_act = use_act

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.use_act:
            x = F.relu(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 1, use_act=False)  # pointwise
        self.conv2 = ConvBNAct(out_ch, out_ch, kernel_size, groups=out_ch)  # depthwise

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.stem1 = ConvBNAct(in_ch, mid_ch, 3, stride=2)
        self.stem2a = ConvBNAct(mid_ch, mid_ch // 2, 2)
        self.stem2b = ConvBNAct(mid_ch // 2, mid_ch, 2)
        self.stem3 = ConvBNAct(mid_ch * 2, mid_ch, 3, stride=2)
        self.stem4 = ConvBNAct(mid_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HG_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n_layers=6, kernel_size=3,
                 light_block=False, residual=False):
        super().__init__()
        Layer = LightConvBNAct if light_block else ConvBNAct
        self.layers = nn.ModuleList([
            Layer(in_ch if i == 0 else mid_ch, mid_ch, kernel_size)
            for i in range(n_layers)
        ])
        total_ch = in_ch + n_layers * mid_ch
        self.aggregation_squeeze_conv = ConvBNAct(total_ch, out_ch // 2, 1)
        self.aggregation_excitation_conv = ConvBNAct(out_ch // 2, out_ch, 1)
        self.residual = residual

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.residual:
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n_blocks, n_layers=6,
                 kernel_size=3, downsample=True, light_block=False):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = ConvBNAct(in_ch, in_ch, 3, stride=2,
                                        groups=in_ch, use_act=False)
        self.blocks = nn.Sequential(*[
            HG_Block(
                in_ch if i == 0 else out_ch, mid_ch, out_ch,
                n_layers, kernel_size, light_block,
                residual=(i > 0),
            )
            for i in range(n_blocks)
        ])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetv2Config(PretrainedConfig):
    model_type = "my_hgnetv2"

    def __init__(self, stem_channels=None, stage_config=None,
                 hidden_size=2048, pretrained="", freeze=False, **kwargs):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels or [3, 32, 48]
        self.stage_config = stage_config or {}
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.freeze = freeze


class HGNetv2(PreTrainedModel):
    config_class = HGNetv2Config
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.stem = StemBlock(*config.stem_channels)
        self.stages = nn.ModuleList()
        for k in sorted(config.stage_config.keys()):
            args = config.stage_config[k]
            # args: [in_ch, mid_ch, out_ch, n_blocks, n_layers, kernel, downsample, light]
            in_ch, mid_ch, out_ch, n_blocks, n_layers, ks = args[:6]
            ds = bool(args[6]) if len(args) > 6 else True
            light = bool(args[7]) if len(args) > 7 else False
            self.stages.append(HG_Stage(in_ch, mid_ch, out_ch, n_blocks,
                                         n_layers, ks, ds, light))

    def forward(self, pixel_values, **kwargs):
        x = self.stem(pixel_values)
        for stage in self.stages:
            x = stage(x)
        out = x.flatten(2).transpose(1, 2)
        return BaseModelOutput(last_hidden_state=out)


# Register the custom model type
from transformers import AutoConfig, AutoModel
try:
    AutoConfig.register("my_hgnetv2", HGNetv2Config)
    AutoModel.register(HGNetv2Config, HGNetv2)
except Exception:
    pass  # Already registered


# ---------------------------------------------------------------------------
# Image preprocessing (matches UniMERNet's processor)
# ---------------------------------------------------------------------------

UNIMERNET_MEAN = 0.7931
UNIMERNET_STD = 0.1738


def preprocess_image(image_path, size=384):
    """Load and preprocess image for PPFormulaNet-S (UniMERNet pipeline).

    1. Resize maintaining aspect ratio to fit within size×size
    2. Pad with black to fill size×size
    3. Convert to grayscale, replicate to 3ch
    4. Normalize with UniMERNet mean/std
    """
    from PIL import Image

    img = Image.open(image_path).convert("L")  # grayscale

    # Resize preserving aspect ratio
    img.thumbnail((size, size), Image.BILINEAR)
    w, h = img.size

    # Pad to size×size (center, black padding)
    padded = Image.new("L", (size, size), 0)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    # Convert to float [0, 1], replicate to 3ch
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=-1)  # HWC

    # Normalize with UniMERNet mean/std
    arr = (arr - UNIMERNET_MEAN) / UNIMERNET_STD

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr


def create_test_image(size=384):
    """Create synthetic test image: gray 0.8 with dark bar."""
    gray = np.ones((size, size), dtype=np.float32) * 0.8
    gray[size // 2 - 2:size // 2 + 2, size // 4:3 * size // 4] = 0.1

    # Replicate to 3ch
    rgb = np.stack([gray, gray, gray], axis=-1)

    # Normalize with UniMERNet mean/std
    arr = (rgb - UNIMERNET_MEAN) / UNIMERNET_STD

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PPFormulaNet-S reference dumper")
    p.add_argument("--checkpoint", required=True, help="Path to formulanet.pt")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer dir")
    p.add_argument("--image", default=None, help="Test image (default: synthetic)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    args = p.parse_args()

    # Load checkpoint first to detect architecture from weights
    print(f"Loading weights: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Detect vocab_size, d_model, n_decoder_layers from weights
    vocab_size = sd["decoder.model.decoder.embed_tokens.weight"].shape[0]
    d_model = sd["decoder.model.decoder.embed_tokens.weight"].shape[1]
    max_pos = sd["decoder.model.decoder.embed_positions.weight"].shape[0]
    dec_ffn = sd["decoder.model.decoder.layers.0.fc1.weight"].shape[0]
    n_dec = 0
    while f"decoder.model.decoder.layers.{n_dec}.self_attn.q_proj.weight" in sd:
        n_dec += 1

    # Detect per-stage config from weight keys
    # stage_config[k] = [in_ch, mid_ch, out_ch, n_blocks, n_layers, kernel, downsample, light]
    stage_defs = {
        # in_ch, mid_ch, out_ch, n_blocks, n_layers, kernel from PPFormulaNet-S spec
        "stage1": [48, 48, 128, 1, 6, 3],
        "stage2": [128, 96, 512, 1, 6, 3],
        "stage3": [512, 192, 1024, 3, 6, 5],
        "stage4": [1024, 384, 2048, 1, 6, 5],
    }
    stage_config = {}
    for si, sname in enumerate(["stage1", "stage2", "stage3", "stage4"]):
        base = stage_defs[sname]
        has_ds = f"encoder.stages.{si}.downsample.conv.weight" in sd
        is_light = f"encoder.stages.{si}.blocks.0.layers.0.conv1.conv.weight" in sd
        stage_config[sname] = base + [1 if has_ds else 0, 1 if is_light else 0]

    print(f"Detected: vocab={vocab_size}, d_model={d_model}, dec_layers={n_dec}, "
          f"max_pos={max_pos}, ffn={dec_ffn}")
    for k, v in stage_config.items():
        print(f"  {k}: in={v[0]} mid={v[1]} out={v[2]} blocks={v[3]} ds={v[6]} light={v[7]}")

    # Build config
    enc_cfg = {
        "model_type": "my_hgnetv2",
        "stem_channels": [3, 32, 48],
        "stage_config": stage_config,
        "hidden_size": 2048,
    }
    dec_cfg = {
        "model_type": "mbart",
        "vocab_size": vocab_size,
        "max_position_embeddings": max_pos - 2,  # MBart adds 2 internally
        "d_model": d_model,
        "decoder_layers": n_dec,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": dec_ffn,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 0,
        "forced_eos_token_id": 2,
        "is_decoder": True,
        "add_cross_attention": True,
        "scale_embedding": True,
        "tie_word_embeddings": False,
    }
    cfg = {
        "model_type": "vision-encoder-decoder",
        "encoder": enc_cfg,
        "decoder": dec_cfg,
        "decoder_start_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 1,
    }

    # Build model
    print("Building VisionEncoderDecoderModel...")
    vec_config = VisionEncoderDecoderConfig(**cfg)
    model = VisionEncoderDecoderModel(vec_config)

    # Load weights
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Model loaded successfully")

    # Prepare input
    if args.image:
        img = preprocess_image(args.image)
        print(f"Image: {args.image} -> {img.shape}")
    else:
        img = create_test_image()
        print(f"Synthetic test image: {img.shape}")

    pixel_values = torch.from_numpy(img).unsqueeze(0).float()

    # Register forward hooks
    captures = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, BaseModelOutput):
                t = output.last_hidden_state
            elif isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            if isinstance(t, torch.Tensor):
                captures[name] = t.detach().float().cpu().numpy()
        return hook_fn

    hooks = []

    # Encoder hooks
    enc = model.encoder
    hooks.append(enc.stem.register_forward_hook(make_hook("stem_output")))
    for si, stage in enumerate(enc.stages):
        hooks.append(stage.register_forward_hook(make_hook(f"stage{si}_output")))
        if stage.downsample is not None:
            hooks.append(stage.downsample.register_forward_hook(make_hook(f"stage{si}_downsample")))
        for bi, block in enumerate(stage.blocks):
            hooks.append(block.register_forward_hook(make_hook(f"stage{si}_block{bi}_output")))

    # Full encoder output
    hooks.append(enc.register_forward_hook(make_hook("enc_output")))

    # enc_to_dec_proj
    hooks.append(model.enc_to_dec_proj.register_forward_hook(make_hook("proj_output")))

    # Decoder layer hooks
    for li, layer in enumerate(model.decoder.model.decoder.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"dec_layer_{li}")))

    # Run encoder
    print("\nRunning encoder...")
    with torch.no_grad():
        enc_out = model.encoder(pixel_values)
        enc_hidden = enc_out.last_hidden_state  # (1, N, 2048)
        print(f"  Encoder output: {enc_hidden.shape}")

        proj_out = model.enc_to_dec_proj(enc_hidden)  # (1, N, 384)
        print(f"  Projected: {proj_out.shape}")

    # Run one decoder step with per-layer hooks
    print("Running decoder step 0...")
    bos = 0  # decoder_start_token_id
    decoder_input_ids = torch.tensor([[bos]], dtype=torch.long)

    # Hook every decoder sublayer for step 0
    dec_mod = model.decoder.model.decoder
    dec_hook_list = []
    for li, layer in enumerate(dec_mod.layers):
        dec_hook_list.append(layer.self_attn_layer_norm.register_forward_hook(
            make_hook(f"dec_self_attn_ln_{li}")))
        dec_hook_list.append(layer.encoder_attn_layer_norm.register_forward_hook(
            make_hook(f"dec_cross_attn_ln_{li}")))
        dec_hook_list.append(layer.final_layer_norm.register_forward_hook(
            make_hook(f"dec_ffn_ln_{li}")))
    dec_hook_list.append(dec_mod.layer_norm.register_forward_hook(
        make_hook("dec_final_ln")))

    with torch.no_grad():
        dec_out = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=proj_out,
            use_cache=False,
            return_dict=True,
        )
        logits = dec_out.logits  # (1, 1, V)
        print(f"  Logits: {logits.shape}")
        top5 = torch.topk(logits[0, 0], 5)
        print(f"  Top-5 tokens: {top5.indices.tolist()} scores: {[f'{s:.3f}' for s in top5.values.tolist()]}")

    for h in dec_hook_list:
        h.remove()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Store results
    captures["input_image"] = img.astype(np.float32)
    captures["logits_step0"] = logits[0, 0].detach().float().cpu().numpy()

    # Run full greedy decode for comparison
    print("\nRunning greedy decode...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            max_length=256,
            num_beams=1,
            do_sample=False,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Output: {decoded[:200]}")
    captures["generated_ids"] = generated[0].numpy().astype(np.int32)

    # Write GGUF
    print(f"\nWriting GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "ppfn_ref")
    writer.add_string("general.name", "ppformulanet-s-reference")
    writer.add_string("ppfn.ref.checkpoint", args.checkpoint)
    writer.add_string("ppfn.ref.decoded", decoded[:512])

    for name, arr in sorted(captures.items()):
        # Squeeze batch dim if present
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.dtype == np.int32:
            writer.add_tensor(name, arr)
        else:
            writer.add_tensor(name, arr.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB, {len(captures)} tensors)")
    for name in sorted(captures.keys()):
        arr = captures[name]
        print(f"  {name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
