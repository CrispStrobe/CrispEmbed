#!/usr/bin/env python
"""Benchmark the EasyOCR Python reference at the same stage boundaries as native.

Uses the repository's Miniconda/PyTorch environment; this is intentionally a
reference-only tool and is not a production dependency.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_model(repo, checkpoint, charset, generation):
    sys.path.insert(0, repo)
    import torch
    try:
        from easyocr.model import model as gen1_model
        from easyocr.model import vgg_model as gen2_model
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise
        import importlib.util
        import types
        tv = types.ModuleType("torchvision")
        tv.__version__ = "0.0"
        tv.models = types.SimpleNamespace()
        transforms = types.ModuleType("torchvision.transforms")
        transforms.ToTensor = object
        tv.transforms = transforms
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = transforms
        root = Path(repo) / "easyocr"
        pkg = types.ModuleType("easyocr")
        pkg.__path__ = [str(root)]
        model_pkg = types.ModuleType("easyocr.model")
        model_pkg.__path__ = [str(root / "model")]
        sys.modules["easyocr"] = pkg
        sys.modules["easyocr.model"] = model_pkg

        def load(name):
            full = f"easyocr.model.{name}"
            spec = importlib.util.spec_from_file_location(full, root / "model" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
            setattr(model_pkg, name, module)
            return module

        load("modules")
        gen1_model = load("model")
        gen2_model = load("vgg_model")
    chars = "".join(Path(charset).read_text(encoding="utf-8").splitlines())
    pkg = gen1_model if generation == 1 else gen2_model
    net = pkg.Model(input_channel=1, output_channel=512 if generation == 1 else 256,
                    hidden_size=512 if generation == 1 else 256, num_class=len(chars) + 1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()})
    net.eval()
    return torch, net, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easyocr-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--charset", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--generation", type=int, choices=(1, 2), required=True)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=20)
    args = ap.parse_args()
    torch, net, chars = load_model(args.easyocr_repo, args.checkpoint, args.charset, args.generation)
    image = np.asarray(Image.open(args.image).convert("L"), dtype=np.uint8)
    height = 64
    resized_w = min(args.width, max(1, int(np.ceil(height * image.shape[1] / image.shape[0]))))
    prep = []
    forward = []
    decode = []
    result = ""
    with torch.no_grad():
        for _ in range(args.repetitions):
            start = time.perf_counter()
            resized = cv2.resize(image, (resized_w, height), interpolation=cv2.INTER_LINEAR)
            arr = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            padded = np.zeros((1, height, args.width), dtype=np.float32)
            padded[:, :, :resized_w] = arr
            if resized_w < args.width:
                padded[:, :, resized_w:] = arr[:, resized_w - 1:resized_w]
            inp = torch.from_numpy(padded[None])
            text = torch.zeros((1, args.width // 10 + 1), dtype=torch.long)
            prep.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            logits = net(inp, text)
            forward.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            best = logits.softmax(2).argmax(2)[0].tolist()
            out = []
            previous = 0
            for token in best:
                if token and token != previous:
                    out.append(chars[token - 1])
                previous = token
            result = "".join(out)
            decode.append((time.perf_counter() - start) * 1000)
    mean = lambda values: sum(values) / len(values)
    print(f"easyocr-reference-benchmark repetitions={args.repetitions} text={result} "
          f"preprocess_ms={mean(prep):.3f} graph_ms={mean(forward):.3f} "
          f"decode_ms={mean(decode):.3f} total_ms={mean([a+b+c for a,b,c in zip(prep, forward, decode)]):.3f}")


if __name__ == "__main__":
    main()
