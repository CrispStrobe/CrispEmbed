#!/usr/bin/env python3
"""Bounded native recurrence parity against pinned official Transformers code.

No checkpoint download. Complements the real LightOnOCR boundary probe with
single-token and multi-batch/chunk-boundary cases at the model's head width.
"""

import json
import os
import subprocess
from pathlib import Path


def main():
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
    )

    if transformers.__version__ != "5.5.4":
        raise ValueError("Requires official transformers==5.5.4")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(3508)
    repo = Path(__file__).resolve().parents[2]
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"])
    build = scratch / "qwen35-delta-build"
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "transformers": transformers.__version__, "torch": torch.__version__,
              "native_ocr_parity": "not implemented", "cases": []}
    try:
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF",
                        "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-delta", "-j", "4"], check=True)
        for batch, tokens in [(1, 1), (2, 65)]:
            label = f"b{batch}-t{tokens}"
            writer = gguf.GGUFWriter(str(out / (label + ".gguf")), "qwen35_delta_reference")
            state = None
            for phase, length in [("prefill", tokens), ("cached", 1)]:
                q, k, v = [torch.randn(batch, length, 16, 128, generator=generator) for _ in range(3)]
                g = -torch.nn.functional.softplus(torch.randn(batch, length, 16, generator=generator))
                beta = torch.sigmoid(torch.randn(batch, length, 16, generator=generator))
                fn = torch_chunk_gated_delta_rule if phase == "prefill" else torch_recurrent_gated_delta_rule
                result, next_state = fn(q, k, v, g=g, beta=beta, initial_state=state,
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
                for name, value in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                                    ("output", result), ("final_state", next_state)]:
                    writer.add_tensor(f"{phase}.delta.{name}", np.ascontiguousarray(value.numpy()))
                state = next_state
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            native = subprocess.run([str(build / "test-qwen35-delta"), str(out / (label + ".gguf"))],
                                    capture_output=True, text=True)
            (out / (label + ".log")).write_text(native.stdout + native.stderr)
            print(native.stdout + native.stderr)
            report["cases"].append({"batch": batch, "prefill_tokens": tokens, "heads": 16,
                                     "width": 128, "exit_code": native.returncode, "log": native.stdout + native.stderr})
            native.check_returncode()
        report["result"] = "native recurrence outputs/states and controls passed; full model pending"
    except BaseException as error:
        report["result"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "delta-parity.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "summary.md").write_text("# Qwen3.5 native delta parity\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
