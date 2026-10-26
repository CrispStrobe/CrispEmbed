#!/usr/bin/env python3
"""Convert the reviewed MI-GAN 256 Places2 state dictionary losslessly to F32.

Requires a completed original capture and exactly its checkpoint SHA. All state
keys, learned filters, fixed masks, noise buffers and noise strengths survive.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dump_migan_reference import inventory, load_original
from migan_reference_contract import CASES, FULL_STAGE_CASES, fixture_identity, load_contract, read_f32_gguf, sha256, write_f32_gguf


def convert(source_dir, checkpoint, checkpoint_sha256, original_report, output):
    report = json.loads(Path(original_report).read_text())
    if report.get("schema") != "crispembed-migan-original-v1" or report.get("complete") is not True:
        raise ValueError("a complete original-module capture must precede conversion")
    if report["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("original capture checkpoint differs")
    contract = load_contract()
    if report["source_commit"] != contract["original_source"]["commit"]:
        raise ValueError("original capture source differs")
    if report["contract_sha256"] != sha256(Path(__file__).resolve().parents[1] / "docs/migan-reference-contract.json"):
        raise ValueError("original capture contract differs")
    if [case["id"] for case in report["cases"]] != list(CASES):
        raise ValueError("original capture cases differ")
    for case in report["cases"]:
        if any(case[key] != value for key, value in fixture_identity(case["id"]).items()):
            raise ValueError("original capture fixture identity differs")
        if case["known_pixels_bit_exact"] is not True or (case["id"] in FULL_STAGE_CASES and case["hook_neutrality_bit_exact"] is not True):
            raise ValueError("original capture neutrality/composition control failed")
        directory = Path(original_report).parent / case["id"]
        for row in case["stages"]:
            if Path(row["file"]).name != row["file"] or sha256(directory / row["file"]) != row["sha256"]:
                raise ValueError("original captured stage file differs")
        for filename, key in (("generated.png", "generated_png_sha256"), ("composited.png", "composited_png_sha256"),
                              ("reference.gguf", "reference_gguf_sha256")):
            if sha256(directory / filename) != case[key]:
                raise ValueError("original captured output differs")
    if report["rejected_controls"] != ["nonbinary", "mask_size", "rgb_channels", "wrong_resolution"]:
        raise ValueError("original negative controls are incomplete")
    import torch

    _, state = load_original(source_dir, checkpoint, checkpoint_sha256)
    if inventory(state) != report["state"]:
        raise ValueError("original captured tensor inventory differs")
    tensors = [(name, tuple(tensor.shape), tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
               for name, tensor in sorted(state.items())]
    metadata = {"general.architecture": "migan", "general.name": "MI-GAN-256-Places2-experimental-F32",
                "general.file_type": 0, "general.license": "MIT", "migan.image_size": 256,
                "migan.source.commit": contract["original_source"]["commit"],
                "migan.checkpoint.sha256": checkpoint_sha256,
                "migan.input.contract": "RGB8;binary-mask8;255=fill;fixed256",
                "migan.tensor.layout": "original contiguous OIHW; GGUF dimensions reversed; scalar=[1]"}
    output = Path(output)
    if output.exists():
        raise ValueError("refuse to overwrite an existing converted artifact")
    write_f32_gguf(output, tensors, metadata)
    actual_metadata, actual = read_f32_gguf(output)
    if actual_metadata != metadata or len(actual) != len(tensors):
        raise AssertionError("GGUF metadata/tensor inventory differs")
    for name, shape, data in tensors:
        if actual[name] != (shape or (1,), data):
            raise AssertionError(f"GGUF byte/shape mapping differs: {name}")
    receipt = {"schema": "crispembed-migan-conversion-v1", "checkpoint_sha256": checkpoint_sha256,
               "gguf_sha256": sha256(output), "gguf_bytes": output.stat().st_size,
               "tensor_count": len(tensors), "all_tensor_bytes_exact": True,
               "original_report_sha256": sha256(original_report), "torch_version": torch.__version__,
               "native_qualified": False, "published": False}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source_dir, args.checkpoint, args.checkpoint_sha256, args.original_report, args.output)


if __name__ == "__main__":
    main()
