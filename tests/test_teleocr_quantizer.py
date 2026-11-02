#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise TeleOCR quantizer provenance on small authored GGUF tensors.

No model download or neural inference is used. The native quantizer must
preserve protected bytes, record actual types, retain ordered/escaped policy
overrides, and distinguish requested versus successfully loaded calibration.
The metadata change must remain scoped to the TeleOCR architecture.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def fixture(path: Path, architecture: str):
    import gguf
    import numpy as np

    rng = np.random.default_rng(724)
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_string("teleocr.precision_recipe", "f16")
    writer.add_uint32("general.file_type", 1)
    for name in ("l.blk.0.attn_q.weight", "l.blk.0.ffn_up.weight", 'l.blk.0.quoted"weight',
                 "l.embed_tokens.weight", "v.blk.0.attn_qkv.weight", "v.patch_embed.weight"):
        writer.add_tensor(name, rng.uniform(-0.4, 0.4, (4, 256)).astype(np.float16))
    writer.add_tensor("l.blk.0.attn_norm.weight", np.ones(256, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def importance_fixture(path: Path):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(path, "imatrix")
    writer.add_uint64("count.l.blk.0.attn_q.weight", 1)
    writer.add_tensor("l.blk.0.attn_q.weight", np.ones(256, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> int:
    import gguf
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantizer", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    quantizer = args.quantizer.resolve(strict=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="teleocr-quantizer-") as temporary:
        root = Path(temporary)
        source, generic, importance = root / "source.gguf", root / "generic.gguf", root / "imatrix.gguf"
        fixture(source, "teleocr")
        fixture(generic, "qwen2vl")
        importance_fixture(importance)
        base_overrides = ["v.*=f16", "l.embed_tokens.weight=f16"]
        variants = [
            ("protected-q4", source, base_overrides, [], None, {"Q4_K": 3, "F16": 3, "F32": 1}),
            ("ordered-overrides-and-flags", source,
             ["v.*=f16", "l.blk.*=q8_0", 'l.blk.0.quoted"weight=f16'],
             ["--decoder-f16", "--decoder-attn-q8", "--ppocrv6-q8-head"], None,
             {"Q8_0": 2, "F16": 4, "F32": 1}),
            ("loaded-imatrix", source, base_overrides, ["--imatrix", str(importance)], "sum",
             {"Q4_K": 3, "F16": 3, "F32": 1}),
            ("missing-imatrix-is-recorded", source, base_overrides,
             ["--imatrix", str(root / "missing.gguf")], "merged", {"Q4_K": 3, "F16": 3, "F32": 1}),
            ("other-architecture-unchanged", generic, base_overrides, [], None,
             {"Q4_K": 3, "F16": 3, "F32": 1}),
            ("q5-provenance", source, base_overrides, [], None, {"Q5_K": 3, "F16": 3, "F32": 1}),
            ("q6-provenance", source, base_overrides, [], None, {"Q6_K": 3, "F16": 3, "F32": 1}),
            ("iq4-nl-provenance", source, base_overrides, [], None, {"IQ4_NL": 3, "F16": 3, "F32": 1}),
            ("iq4-xs-provenance", source, base_overrides, [], None, {"IQ4_XS": 3, "F16": 3, "F32": 1}),
        ]
        baseline = None
        for name, model, overrides, flags, imatrix_mode, expected in variants:
            output = root / (name + ".gguf")
            target = {"q5-provenance": "q5_k", "q6-provenance": "q6_k",
                      "iq4-nl-provenance": "iq4_nl", "iq4-xs-provenance": "iq4_xs"}.get(name, "q4_k")
            command = [str(quantizer), str(model), str(output), target, *flags]
            for rule in overrides:
                command.extend(["--tensor-type", rule])
            environment = os.environ.copy()
            environment.pop("CRISPEMBED_QUANT_IMATRIX_QKV", None)
            if imatrix_mode:
                environment["CRISPEMBED_QUANT_IMATRIX_QKV"] = imatrix_mode
            run = subprocess.run(command, capture_output=True, text=True, timeout=30, env=environment)
            row = {"name": name, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
            try:
                assert run.returncode == 0, "native quantizer failed"
                reader = gguf.GGUFReader(output)
                tensors = {t.name: t for t in reader.tensors}
                original = {t.name: t for t in gguf.GGUFReader(model).tensors}
                assert tensors.keys() == original.keys()
                counts = dict(Counter(t.tensor_type.name for t in tensors.values()))
                assert counts == expected, (counts, expected)
                for tensor_name, tensor in tensors.items():
                    old = original[tensor_name]
                    assert np.array_equal(tensor.shape, old.shape), tensor_name
                    if tensor.tensor_type == old.tensor_type:
                        assert np.array_equal(tensor.data, old.data), tensor_name
                if model == generic:
                    assert reader.fields["teleocr.precision_recipe"].contents() == "f16"
                    assert not any(k.startswith("teleocr.quantization.") for k in reader.fields)
                    for tensor_name, tensor in tensors.items():
                        assert np.array_equal(tensor.data, baseline[tensor_name].data)
                else:
                    recipe = json.loads(reader.fields["teleocr.precision_recipe"].contents())
                    assert recipe["tool"] == "crispembed-quantize"
                    assert recipe["requested_type"].lower() == target
                    expected_file_type = {"q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_S,
                                          "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_S,
                                          "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
                                          "iq4_nl": gguf.LlamaFileType.MOSTLY_IQ4_NL,
                                          "iq4_xs": gguf.LlamaFileType.MOSTLY_IQ4_XS}[target]
                    assert reader.fields["general.file_type"].contents() == expected_file_type.value
                    assert recipe["tensor_overrides"] == [
                        {"pattern": rule.rsplit("=", 1)[0], "type": rule.rsplit("=", 1)[1]}
                        for rule in overrides]
                    assert recipe["decoder_f16"] == ("--decoder-f16" in flags)
                    assert recipe["decoder_attn_q8"] == ("--decoder-attn-q8" in flags)
                    assert recipe["ppocrv6_q8_head"] == ("--ppocrv6-q8-head" in flags)
                    assert recipe["imatrix_requested"] == ("--imatrix" in flags)
                    assert recipe["imatrix_vectors_loaded"] == (1 if name == "loaded-imatrix" else 0)
                    assert recipe["imatrix_qkv_mode"] == (imatrix_mode or "direct")
                    type_counts = reader.fields["teleocr.quantization.tensor_type_counts"].contents()
                    assert sum(type_counts) == 7
                    assert {gguf.GGMLQuantizationType(i).name: n for i, n in enumerate(type_counts) if n} == counts
                    assert reader.fields["teleocr.quantization.tensor_count"].contents() == 7
                    converted = 2 if name == "ordered-overrides-and-flags" else 3
                    assert reader.fields["teleocr.quantization.converted_tensor_count"].contents() == converted
                    assert reader.fields["teleocr.quantization.copied_tensor_count"].contents() == 7 - converted
                    assert reader.fields["teleocr.quantization.imatrix_applied_count"].contents() == (
                        1 if name == "loaded-imatrix" else 0)
                    row["recipe"] = recipe
                if name == "protected-q4":
                    baseline = tensors
                row.update(passed=True, tensor_types=counts,
                           output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            except (AssertionError, KeyError, ValueError) as error:
                row.update(passed=False, error=str(error))
            rows.append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} {name}", flush=True)
    report = {"kind": "teleocr-quantizer-provenance-regression", "model_weights_used": False,
              "quantizer_sha256": hashlib.sha256(quantizer.read_bytes()).hexdigest(),
              "passed": all(row["passed"] for row in rows), "checks": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
