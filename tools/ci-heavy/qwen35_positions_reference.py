#!/usr/bin/env python3
"""Exact still-image RoPE parity with pinned original Transformers methods, no weights."""
import json
import os
import subprocess
from pathlib import Path


def main():
    import torch
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    if transformers.__version__ != "5.5.4":
        raise ValueError("Original Transformers 5.5.4 required")
    # Actual official model with unallocated weights: positions do not use weights.
    config = Qwen3_5Config()
    with torch.device("meta"):
        model = Qwen3_5Model(config)
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    repo = Path(__file__).resolve().parents[2]
    cases = [
        ("text-one", [0], []), ("text-seventeen", [0] * 17, []),
        ("image-only", [1] * 6, [[1, 4, 6]]),
    ]
    for h, w in [(2, 2), (4, 6), (6, 4), (16, 16), (2, 32), (32, 2)]:
        for prefix, suffix in [(1, 1), (7, 5)]:
            cases.append((f"grid-{h}x{w}-{prefix}-{suffix}", [0] * prefix + [1] * (h * w // 4) + [0] * suffix,
                          [[1, h, w]]))
    cases += [
        ("two-images", [0] * 3 + [1] * 6 + [0] * 2 + [1] * 8 + [0] * 4, [[1, 4, 6], [1, 8, 4]]),
        ("three-images", [1] + [0] + [1] * 6 + [0] * 2 + [1] * 6, [[1, 2, 2], [1, 4, 6], [1, 6, 4]]),
        ("text-context-bound", [0] * 4090, []),
    ]
    rows, records = [str(len(cases))], []
    for index, (name, types, grids) in enumerate(cases):
        ids = torch.zeros((1, len(types)), dtype=torch.long)
        mm = torch.tensor([types], dtype=torch.long)
        grid = torch.tensor(grids, dtype=torch.long).reshape(-1, 3)
        axes, delta = model.get_rope_index(ids, mm, image_grid_thw=grid)
        # Exact original generation prefill logic: text chronology + 3D positions.
        generation = torch.cat([torch.arange(len(types)).reshape(1, 1, -1), axes], dim=0)
        cached_tokens = 1 + index % 4
        cached = (torch.arange(len(types), len(types) + cached_tokens).reshape(1, 1, -1) + delta).expand(3, -1, -1)
        rows.append(f"{name} {len(types)} {len(grids)} {int(delta.item())} {cached_tokens}")
        for values in [types, grid.flatten().tolist(), generation.flatten().tolist(), cached.flatten().tolist()]:
            rows.append(" ".join(map(str, values)))
        records.append({"name": name, "tokens": len(types), "grids": grids, "delta": int(delta.item()),
                        "cached_tokens": cached_tokens})
    fixture = out / "official-positions.txt"
    fixture.write_text("\n".join(rows) + "\n")
    build = scratch / "positions-build"
    subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF",
                    "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-positions", "-j", "4"], check=True)
    result = subprocess.run([str(build / "test-qwen35-positions"), str(fixture)], capture_output=True, text=True)
    (out / "native-positions.log").write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr)
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "reference": "Transformers 5.5.4 original Qwen3_5Model.get_rope_index/get_vision_position_ids",
              "scope": "unpadded batch=1 still images; no model weights or OCR-quality claim", "cases": records,
              "passed": result.returncode == 0}
    (out / "positions.json").write_text(json.dumps(report, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
