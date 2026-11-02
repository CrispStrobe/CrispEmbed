#!/usr/bin/env python3
"""Recipe helper for the qwen2vl_ocr Q4_K recipe sweep (docs/qwen2vl-q4k-recipes.md).

    python tools/qwen2vl_quant_census.py args MODEL RECIPE      # quantizer argv tail, one per line
    python tools/qwen2vl_quant_census.py census GGUF OUT.json   # stored bytes by type and component
    python tools/qwen2vl_quant_census.py summary MODEL DIR      # sweep table from DIR/*.cosine.json

Recipes live in tools/qwen2vl_quant_recipes.json.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "tools/qwen2vl_quant_recipes.json"


def recipe(model, name):
    return json.loads(RECIPES.read_text())[model][name]


def component(name):
    if name.startswith("v.merger") or name == "v.patch_embed.weight":
        return "vision_host"
    if name.startswith("v."):
        return "vision"
    if name in ("l.embed_tokens.weight", "l.lm_head.weight"):
        return "embedding"
    m = re.match(r"l\.blk\.\d+\.(\w+)\.weight$", name)
    if m:
        return "decoder." + m.group(1)
    return "other"


def census(path):
    from gguf import GGUFReader  # gguf==0.17.1 in the sweep job

    import hashlib

    data = hashlib.sha256()  # tensor names, types and bytes in file order (metadata excluded)
    by_type = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    by_comp = defaultdict(lambda: defaultdict(lambda: {"tensors": 0, "bytes": 0}))
    for t in GGUFReader(path).tensors:
        ty = t.tensor_type.name
        data.update(f"{t.name}:{ty}:".encode())
        data.update(t.data.tobytes())
        by_type[ty]["tensors"] += 1
        by_type[ty]["bytes"] += int(t.n_bytes)
        c = by_comp[component(t.name)][ty]
        c["tensors"] += 1
        c["bytes"] += int(t.n_bytes)
    return {"file_bytes": Path(path).stat().st_size, "tensor_data_sha256": data.hexdigest(), "by_type": dict(by_type),
            "by_component": {k: dict(v) for k, v in by_comp.items()}}


def summary(model, d):
    d = Path(d)
    rows = []
    for cos in sorted(d.glob("*-cpu.cosine.json")):
        name = cos.name[: -len("-cpu.cosine.json")]
        c = json.loads(cos.read_text())
        cen = d / f"{name}.census.json"
        size = json.loads(cen.read_text())["file_bytes"] if cen.exists() else None
        per = {k: e.get("final_output_min_cosine") for k, e in c["cases"].items()}
        worst = min(per.items(), key=lambda kv: kv[1] if kv[1] is not None else -1)
        rows.append({"recipe": name, "file_bytes": size, "final_cos_min": c.get("final_output_min_cosine"),
                     "worst_case": worst[0], "complete": c.get("final_output_complete"), "per_case": per,
                     "worst_stage": c.get("worst_stage")})
    out = {"schema": "crispembed.qwen2vl.q4k-sweep.v1", "model": model, "bar": 0.94, "recipes": rows}
    (d / "sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| recipe | MiB | final cos min (worst case) | > 0.94 | per case |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["final_cos_min"] or 0.0)):
        mib = f"{r['file_bytes'] / 2**20:.0f}" if r["file_bytes"] else "?"
        fo = r["final_cos_min"] or 0.0
        pc = " ".join(f"{k}={v:.4f}" for k, v in r["per_case"].items() if v is not None)
        print(f"| {r['recipe']} | {mib} | {fo:.6f} ({r['worst_case']}) | {'yes' if fo > 0.94 and r['complete'] else 'no'}"
              f" | {pc} |")


def main():
    cmd = sys.argv[1]
    if cmd == "args":
        r = recipe(sys.argv[2], sys.argv[3])
        print(r["type"])
        for rule in r.get("tensor_type", []):
            print("--tensor-type")
            print(rule)
    elif cmd == "env":
        for k, v in recipe(sys.argv[2], sys.argv[3]).get("env", {}).items():
            print(f"{k}={v}")
    elif cmd == "census":
        Path(sys.argv[3]).write_text(json.dumps(census(sys.argv[2]), indent=1) + "\n")
    elif cmd == "summary":
        summary(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
