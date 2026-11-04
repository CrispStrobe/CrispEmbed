#!/usr/bin/env python3
"""Supplemental receipt barrier for the frozen Pixtral epsilon continuation."""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
import tempfile

FIRST_RUN = 41885702781
FIRST_SOURCE = "f27223fc276674c4621f32b709aa03ef0d035f09"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(first, run, first_run):
    """Validate an already authenticated receipt; performs no I/O."""
    require(type(first_run) is int and first_run == FIRST_RUN, "first run differs")
    require(type(run.get("id")) is int and run["id"] == FIRST_RUN, "producer run differs")
    require(first.get("source") == run.get("head_sha") == FIRST_SOURCE, "first source differs")
    require(first.get("phase") == "first", "first phase differs")
    for key in ("completed", "pass"):
        require(first.get(key) is True, "first flag differs: " + key)
    for key, expected in (("native_request_processes", 12), ("original_forward_calls", 0)):
        require(type(first.get(key)) is int and first[key] == expected, "first counter differs: " + key)
    rows = first.get("rows")
    require(isinstance(rows, list) and len(rows) == 4, "first row count differs")
    expected = {("form-grounding", precision, gate) for precision in ("f32", "f16") for gate in ("0", "1")}
    seen = set()
    for row in rows:
        require(isinstance(row, dict), "first row must be an object")
        identity = (row.get("case"), row.get("precision"), row.get("gate"))
        require(all(isinstance(value, str) for value in identity), "invalid first row identity")
        require(identity in expected and identity not in seen, "first row coverage differs")
        seen.add(identity)
        if identity[2] == "0":
            controls = row.get("gate0_byte_controls")
            require(isinstance(controls, dict) and set(controls) == {"forced", "autonomous", "plain"}, "gate0 arms differ")
            for arm, control in controls.items():
                require(isinstance(control, dict) and control.get("all_bytes_exact") is True,
                        "gate0 bytes differ")
                count = control.get("tensor_files")
                require(type(count) is int and (count == 0 if arm == "plain" else count > 0),
                        "gate0 tensor count invalid")
                tensors = control.get("tensors")
                require(isinstance(tensors, list) and len(tensors) == count, "gate0 inventory count differs")
                names = set()
                for tensor in tensors:
                    require(isinstance(tensor, dict), "invalid gate0 inventory record")
                    name = tensor.get("name")
                    require(isinstance(name, str) and name.endswith(".f32") and name not in names,
                            "invalid gate0 tensor name")
                    names.add(name)
                    size = tensor.get("bytes")
                    require(type(size) is int and size > 0 and size % 4 == 0, "invalid gate0 tensor bytes")
                    digest = tensor.get("sha256")
                    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                            "invalid gate0 tensor digest")
        else:
            require(row.get("first_norm_improved") is True, "first norm did not improve")
            comparison = row.get("comparison")
            require(isinstance(comparison, dict) and comparison.get("owner_bar_pass") is True, "owner bar failed")
            decoded = comparison.get("decoded")
            require(isinstance(decoded, dict) and decoded.get("ids_exact") is True and decoded.get("text_exact") is True,
                    "decoded output differs")
    require(seen == expected, "first row coverage incomplete")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("first", "full"), default="first")
    parser.add_argument("--first-run", type=int)
    parser.add_argument("--first-artifact", type=int)
    parser.add_argument("--first-sha")
    args = parser.parse_args()
    if args.phase == "first":
        return 0
    require(args.first_run == FIRST_RUN, "first run differs")
    path = Path(__file__).with_name("lighton3_pixtral_eps_ab.py")
    spec = importlib.util.spec_from_file_location("lighton3_eps_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    token = os.environ["GH_TOKEN"]
    protocol = worker.archive.read_json(worker.PROTOCOL.read_bytes())
    with tempfile.TemporaryDirectory(prefix="lighton3-eps-first-") as directory:
        first = worker.first_proof(args, protocol, Path(directory), token)
    run = worker.archive.fetch_metadata("/actions/runs/" + str(args.first_run), token)
    validate(first, run, args.first_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
