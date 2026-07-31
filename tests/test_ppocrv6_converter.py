#!/usr/bin/env python3
"""Structural guardrails for PP-OCRv6 GGUF conversion and quantization policy."""

import argparse
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.models:
        reader = GGUFReader(str(path))
        kv = {k: v for k, v in reader.fields.items()}
        arch = bytes(kv["general.architecture"].parts[-1]).decode("utf-8")
        assert arch == "ppocrv6", (path, arch)
        names = {t.name for t in reader.tensors}
        assert any(n.startswith("det.") or n.startswith("rec.") for n in names)
        for tensor in reader.tensors:
            name = tensor.name
            assert len(name) < 64, name
            if (name.endswith(".bias") or "normalization" in name or "squeeze_excitation" in name
                    or "token_conv" in name or ".se1." in name or ".se2." in name or ".dw." in name):
                assert tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16), name
        print(f"PASS {path.name}: {len(reader.tensors)} tensors")


if __name__ == "__main__":
    main()
