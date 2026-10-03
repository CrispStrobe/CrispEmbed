#!/usr/bin/env python3
"""Exercise mixed matrix precision on a small real GGUF, without model weights."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from gguf import GGUFReader, GGUFWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizer", default="build/crispembed-quantize")
    args = ap.parse_args()
    quantizer = str(Path(args.quantizer).resolve())
    with tempfile.TemporaryDirectory(prefix="crispembed-quant-override-") as directory:
        root = Path(directory)
        source, baseline, mixed = root / "source.gguf", root / "baseline.gguf", root / "mixed.gguf"
        rng = np.random.default_rng(42)
        data = {name: rng.normal(size=shape).astype(np.float32) for name, shape in {
            "token_embd.weight": (16, 256), "blk.0.ffn_gate.weight": (16, 256),
            "blk.0.ffn_down.weight": (16, 256), "blk.0.shortconv.in_proj.weight": (16, 256),
            "blk.0.shortconv.conv.weight": (256, 3), "blk.0.ffn_norm.weight": (256,),
            "untouched.weight": (16, 256), "narrow.weight": (16, 64), "float32.weight": (16, 256),
        }.items()}
        writer = GGUFWriter(str(source), "lfm2")
        writer.add_name("override-fixture")
        for name, values in data.items():
            writer.add_tensor(name, values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        def run(output, *rules):
            return subprocess.run([quantizer, str(source), str(output), "q4_k", *rules],
                                  capture_output=True, text=True)

        p = run(baseline)
        assert p.returncode == 0, p.stderr
        p = run(mixed, "--tensor-type", "*.ffn_*.*=q8_0",
                "--tensor-type", "blk.0.ffn_down.weight=f16",
                "--tensor-type", "blk.?.shortconv.*=f16",
                "--tensor-type", "token_embd.weight=f16",
                "--tensor-type", "narrow.weight=q6_k", "--tensor-type", "float32.weight=f32")
        assert p.returncode == 0, p.stderr
        before = {t.name: t for t in GGUFReader(baseline).tensors}
        after = {t.name: t for t in GGUFReader(mixed).tensors}
        for name, expected in {
            "token_embd.weight": "F16", "blk.0.ffn_gate.weight": "Q8_0",
            "blk.0.ffn_down.weight": "F16", "blk.0.shortconv.in_proj.weight": "F16",
            "blk.0.shortconv.conv.weight": "F32", "blk.0.ffn_norm.weight": "F32",
            "untouched.weight": "Q4_K", "narrow.weight": "Q8_0", "float32.weight": "F32",
        }.items():
            assert after[name].tensor_type.name == expected, (name, after[name].tensor_type)
        for name in ["blk.0.shortconv.conv.weight", "blk.0.ffn_norm.weight", "untouched.weight"]:
            np.testing.assert_array_equal(after[name].data, before[name].data)
        for name in ["token_embd.weight", "blk.0.ffn_down.weight", "blk.0.shortconv.in_proj.weight"]:
            np.testing.assert_array_equal(after[name].data, data[name].astype(np.float16))
        np.testing.assert_array_equal(after["float32.weight"].data, data["float32.weight"])
        # A typo must fail before truncating a destination, rather than silently
        # producing the wrong precision mix. Invalid types fail at parsing.
        protected = root / "protected.gguf"
        protected.write_bytes(b"existing-output")
        for spec, message in [("missing.*=f16", "matches no tensor"), ("*=invalid", "Invalid tensor override")]:
            p = run(protected, "--tensor-type", spec)
            assert p.returncode != 0 and message in p.stderr
            assert protected.read_bytes() == b"existing-output"
        print("PASS: stored matrix types, wildcard/precedence, norm/conv preservation, "
              "F16/F32 values, Q6 fallback and rejected overrides without output truncation")


if __name__ == "__main__":
    main()
