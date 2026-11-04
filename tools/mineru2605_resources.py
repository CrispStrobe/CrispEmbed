#!/usr/bin/env python3
"""Weight-free selection and conservative disk preflight for one hosted MinerU case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def selection(case_id, tiers, max_tokens, repeat, document_helpers):
    protocol = json.loads((ROOT / "docs/mineru25-pro-2605-protocol.json").read_text())
    if case_id not in protocol["cases"]:
        raise ValueError("one frozen case per dispatch is required; 'all' and case lists are not supported")
    names = tiers.split(",")
    if not names or len(names) != len(set(names)) or not set(names) <= protocol["precision_bars_exclusive"].keys():
        raise ValueError("unknown or duplicate precision tier")
    if not 0 <= max_tokens <= 8192 or not 2 <= repeat <= 100 or document_helpers not in ("0", "1"):
        raise ValueError("invalid token/repeat/helper arguments")
    return protocol, names


def resource_plan(tiers, max_tokens):
    manifest = json.loads((ROOT / "models/mineru25-pro-2605-source.json").read_text())
    source_bytes = sum(row["size"] for row in manifest["files"].values())
    # Every source tensor is BF16. Twice the whole source weight file plus
    # 64 MiB exceeds any single F32/F16/Q8/Q4 GGUF in this exact converter.
    gguf_upper = 2 * manifest["files"]["model.safetensors"]["size"] + 64 * 1024 ** 2
    positions = max_tokens or 8192  # 0 selects the source total-context policy.
    logits = (1 + len(tiers)) * positions * 151936 * 4
    # Original 65-array archive: <=8192 patches and <=8192 text positions.
    # 3 GiB covers 33*1280 vision channels, 26*896 decoder channels, input
    # patches/IDs/positions and merger/logits. Native extra dumps are small.
    stage_bytes = 3 * GIB + len(tiers) * 64 * 1024 ** 2
    return {"schema": "crispembed.mineru2605.resources.v1", "cases_per_dispatch": 1,
            "tiers": tiers, "positions_upper": positions,
            "logit_bytes_upper": logits, "stage_and_extra_dump_bytes_upper": stage_bytes,
            "scratch_bytes_required": source_bytes + gguf_upper * (1 + any(t != "f32" for t in tiers)) + 2 * GIB,
            "output_bytes_required": logits + stage_bytes + 2 * GIB,
            "available_memory_bytes_required": 12 * GIB,
            "policy": "one case; F32 plus one temporary candidate GGUF; raw arrays retained directly in uploaded output",
            "qualification": False, "resource_bounds_are_not_runtime_measurements": True}


def disk_capacity(plan, locations):
    """Sum requirements when scratch/output share a filesystem, not two free-space checks."""
    groups = {}
    for role in ("scratch", "output"):
        location = locations[role]
        group = groups.setdefault(location["device"], {"roles": [], "required_bytes": 0,
                                                      "free_bytes": location["free_bytes"]})
        group["roles"].append(role)
        group["required_bytes"] += plan[role + "_bytes_required"]
        group["free_bytes"] = min(group["free_bytes"], location["free_bytes"])
    return [{**row, "sufficient": row["free_bytes"] >= row["required_bytes"]} for row in groups.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tiers", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--document-helpers", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol, tiers = selection(args.case, args.tiers, args.max_tokens, args.repeat, args.document_helpers)
    row = protocol["cases"][args.case]
    if hashlib.sha256((ROOT / row["image"]).read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("frozen case image bytes changed")
    for directory in (args.scratch, args.out):
        directory.mkdir(parents=True, exist_ok=True)
    case_output = args.out / args.case
    if any(args.scratch.iterdir()) or (case_output.exists() and any(case_output.iterdir())):
        raise ValueError("task scratch and case evidence must be fresh; retain previous results separately")
    plan = resource_plan(tiers, args.max_tokens)
    plan["case"] = args.case
    plan["filesystems"] = disk_capacity(plan, {role: {"device": path.stat().st_dev,
        "free_bytes": shutil.disk_usage(path).free} for role, path in (("scratch", args.scratch), ("output", args.out))})
    memory = {}
    if sys.platform.startswith("linux"):
        memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemAvailable:", "MemTotal:"))}
    plan["memory"] = memory
    plan["preflight_pass"] = (bool(memory) and memory.get("MemAvailable", 0) >= plan["available_memory_bytes_required"]
                              and all(row["sufficient"] for row in plan["filesystems"]))
    (args.out / "resource-preflight.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "selected-cases.tsv").write_text(f"{args.case}\t{row['image']}\t{row['task']}\n")
    if not plan["preflight_pass"]:
        raise SystemExit("insufficient hosted Linux memory or disk capacity; see resource-preflight.json; no model work started")


if __name__ == "__main__":
    main()
