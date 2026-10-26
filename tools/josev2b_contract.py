#!/usr/bin/env python3
"""Source-only Josev contract checks. This module imports no model framework."""
from __future__ import annotations

import hashlib
import json
import math
import ast
import dataclasses
import sys
import types
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/josev2b-source-contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def require_cuda_capability(capability) -> None:
    minimum = tuple(protocol()["scope"]["min_compute_capability"])
    if tuple(capability) < minimum:
        raise ValueError("unsupported GPU compute capability; require >=" + ".".join(map(str, minimum)) + ", no retry")


def environment_versions() -> dict:
    cfg = protocol()
    versions = {name: importlib.metadata.version(name) for name in cfg["runtime"] if name != "python_minor"}
    if any(versions[name].split("+")[0] != version for name, version in cfg["runtime"].items() if name != "python_minor"):
        raise ValueError(f"package profile differs: {versions}")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != cfg["runtime"]["python_minor"]:
        raise ValueError("Python minor differs")
    original = Path(importlib.metadata.distribution("transformers").locate_file(
        "transformers/models/qwen3_5/modeling_qwen3_5.py"))
    if sha256(original) != cfg["transformers_source"]["sha256"]:
        raise ValueError("installed original Qwen source differs")
    # Full locked distribution and metadata/origin audit precedes model payloads.
    from josev2b_environment import verify_installed
    import os
    prefix = os.environ.get("CRISPEMBED_JOSEV_EXPECTED_VENV")
    if not prefix:
        raise ValueError("reviewed isolated venv prefix required")
    verify_installed(cfg, Path(prefix))
    return versions


def finite_json(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, str) and "<|" in value:
        raise ValueError("user special-token surface is outside text-v1")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for key, item in value.items():
            finite_json(key)
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("not JSON")


def validate_request(request: dict) -> None:
    """Explicit subset, without silently changing upstream input semantics."""
    finite_json(request)
    if len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 65536:
        raise ValueError("request exceeds 65536 UTF-8 bytes")
    if set(request) != {"model", "state", "questions"}:
        raise ValueError("text-v1 accepts exactly model/state/questions; media is excluded")
    if request["model"] != protocol()["repository"]:
        raise ValueError("unexpected model")
    questions = request["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 8:
        raise ValueError("expected 1..8 questions")
    options = 0
    for key, question in questions.items():
        if not key or not isinstance(question, dict):
            raise ValueError("nonempty field ID and object required")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("unknown question members")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise ValueError("noul criteria may override true/false only")
            options += 2
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria or any(not key for key in criteria):
                raise ValueError("choice requires nonempty string option IDs")
            options += len(criteria)
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("score requires an ordered nonempty list")
            options += len(criteria)
        else:
            raise ValueError("unsupported question type")
    if options > 24:
        raise ValueError("too many options")


def panel() -> dict:
    contract = protocol()
    path = ROOT / contract["panel"]["path"]
    if sha256(path) != contract["panel"]["sha256"]:
        raise ValueError("frozen input panel changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    cases = result["cases"]
    if len(cases) != contract["panel"]["cases"] or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case coverage changed")
    for case in cases:
        validate_request(case["request"])
    return result


def verify_snapshot(directory: Path, *, source_only: bool = False) -> dict:
    verified = {}
    for name, expected in protocol()["files"].items():
        if source_only and expected["verification"].startswith("published-LFS"):
            continue
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"snapshot bytes differ: {name}")
        verified[name] = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
    return verified


def inspect_serialization(source: Path) -> dict:
    """Execute only pinned serializer/answer declarations with a byte-ID stub.

    This checks real source semantics, not real Qwen tokenizer IDs or inference.
    Never imports the upstream module or torch and never changes its declarations.
    """
    if sha256(source) != protocol()["files"]["joint_schema_model.py"]["sha256"]:
        raise ValueError("source inspection requires exact pinned module bytes")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    declarations = {"EncodedQuestion", "EncodedRecord", "render", "question_options", "_tokens",
                    "_encode_media", "encode_record", "systemone_answer"}
    constants = {"SYSTEM_PROMPT", "IMAGE_PLACEHOLDER", "VIDEO_PLACEHOLDER", "MEDIA_BATCH_KEYS",
                 "MEDIA_TOKEN_KEYS", "QUESTION_TYPES"}
    selected = [node for node in tree.body if
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in declarations or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    module = types.ModuleType("_josev_pinned_serializer_inspection")
    module.__dict__.update(json=json, dataclass=dataclasses.dataclass, dataclass_field=dataclasses.field)
    sys.modules[module.__name__] = module
    header = ast.parse("from __future__ import annotations").body
    try:
        exec(compile(ast.Module(body=header + selected, type_ignores=[]), str(source), "exec"), module.__dict__)

        class ByteTokenizer:
            def __call__(self, text, add_special_tokens):
                if add_special_tokens:
                    raise ValueError("upstream changed special-token policy")
                return types.SimpleNamespace(input_ids=list(text.encode("utf-8")))

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        require(module.render({"z": "ä", "a": [None, False]}) == '{"a":[null,false],"z":"ä"}', "JSON rendering differs")
        require(module.render("  literal\n") == "  literal\n", "string whitespace changed")
        require(module.question_options({"type": "noul"})[0][0] == "true", "noul order differs")
        require(module.question_options({"type": "choice", "criteria": {"z": None, "a": "A"}}) ==
                [("a", "A"), ("z", None)], "choice sorting differs")
        require(module.question_options({"type": "score", "criteria": ["low", "high"]}) ==
                [("0", "low"), ("1", "high")], "ordinal order differs")
        request = {"state": "abcdef", "questions": {"Frage": {"type": "choice", "instructions": "",
                   "criteria": {"z": None, "a": "ä"}}}}
        full = module.encode_record(ByteTokenizer(), request, max_length=4096)
        clipped = module.encode_record(ByteTokenizer(), request, max_length=len(full.input_ids) - 3)
        question = full.questions[0]
        require(bytes(full.input_ids[slice(*question.question_span)]).decode() == "Frage", "instruction fallback differs")
        require(question.option_ids == ("a", "z"), "encoded option IDs differ")
        require(bytes(full.input_ids[slice(*question.option_spans[1])]).decode() == '{"option_id":"z"}',
                "null description must be omitted")
        shift = question.question_span[0] - clipped.questions[0].question_span[0]
        require(shift == 3 and len(clipped.input_ids) == len(full.input_ids) - 3, "state-only clipping differs")
        require(full.input_ids[question.question_span[0]:] ==
                clipped.input_ids[clipped.questions[0].question_span[0]:], "schema/suffix changed during clipping")
        try:
            module.encode_record(ByteTokenizer(), request, max_length=len(full.input_ids) - 7)
        except ValueError:
            pass
        else:
            raise ValueError("schema-only overflow did not reject")
        choice = module.systemone_answer(request["questions"]["Frage"], {"a": .5, "z": .5})
        require(choice["choice"] == "z" and list(choice["probabilities"]) == ["z", "a"], "original-order tie differs")
        score = module.systemone_answer({"type": "score", "criteria": [10, 100]}, {"0": .25, "1": .75})
        require(score["score"] == .75 and score["legend"] == {"0": 10, "1": 100}, "ordinal expectation differs")
        require(module.systemone_answer({"type": "noul"}, {"true": .23456})["noul"] == .2346, "noul rounding differs")
        return {"source_sha256": sha256(source), "serializer_controls": "PASS",
                "tokenizer": "UTF-8 byte-ID stub; no real tokenizer/model inference"}
    finally:
        sys.modules.pop(module.__name__, None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path)
    args = parser.parse_args()
    panel()
    if args.source_directory:
        verify_snapshot(args.source_directory, source_only=True)
        print(json.dumps(inspect_serialization(args.source_directory / "joint_schema_model.py")))
    print("Josev source/panel controls PASS; no model execution")
