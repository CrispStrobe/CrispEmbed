#!/usr/bin/env python3
"""Inspect a Paddle PIR inference artifact without importing Paddle.

PP-LCNet's current Hugging Face export is ``inference.json`` plus a binary
``inference.pdiparams`` stream.  This tool extracts the persistable parameter
names/shapes and layer structure from the JSON graph so a native GGUF
converter can be built and reviewed without putting Paddle in the runtime.
It deliberately does not decode or copy the weight stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def parameter_inventory(program):
    out = {}
    for node in walk(program):
        if node.get("#") != "p":
            continue
        args = node.get("A", [])
        name = args[-1] if args and isinstance(args[-1], str) else None
        outputs = node.get("O", {})
        if not name or not isinstance(outputs, (dict, list)):
            continue
        output = outputs if isinstance(outputs, dict) and "TT" in outputs else next(
            (x for x in outputs if isinstance(x, dict) and "TT" in x), {})
        tt = output.get("TT", {})
        dtypes = tt.get("D", []) if isinstance(tt, dict) else []
        shape = dtypes[1] if len(dtypes) > 1 and isinstance(dtypes[1], list) else None
        if shape is not None:
            out[name] = {"shape": shape, "dtype": dtypes[0].get("#") if dtypes and isinstance(dtypes[0], dict) else None}
    return out


def layer_inventory(program):
    layers = []
    for node in walk(program):
        if node.get("#") not in {"1.conv2d", "1.depthwise_conv2d", "1.batch_norm_"}:
            continue
        attrs = node.get("A", [])
        struct = next((a.get("AT", {}).get("D") for a in attrs
                       if isinstance(a, dict) and a.get("N") == "struct_name"), None)
        if struct:
            layers.append({"op": node["#"], "struct_name": struct})
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    graph_path = args.model_dir / "inference.json"
    doc = json.loads(graph_path.read_text())
    program = doc.get("program", {})
    report = {
        "model": doc.get("Global", {}).get("model_name", args.model_dir.name),
        "parameters": parameter_inventory(program),
        "layers": layer_inventory(program),
        "preprocess": doc.get("PreProcess", {}),
        "postprocess": doc.get("PostProcess", {}),
        "weights": str(args.model_dir / "inference.pdiparams"),
        "weights_decoded": False,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
