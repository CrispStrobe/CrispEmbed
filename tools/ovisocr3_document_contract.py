#!/usr/bin/env python3
"""Audit the pinned OvisOCR3 Markdown task; no frameworks, weights or runtime permission."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/ovisocr3-document-source"
CONTRACT = ROOT / "docs/ovisocr3-document-protocol.json"
SOURCE_REPORT = ROOT / "tests/results/qwen35-ocr/ovisocr3-source-contract.json"
SOURCE_REPORT_SHA256 = "8ca73ca6a349b67a7533c36de6c14ac066e1152f2c8d517af849b2655087a331"
MODEL = "ATH-MaaS/OvisOCR3"
REVISION = "5f6ce6932efee9ec10c30746db55bd780214b7fd"
PROMPT_SHA256 = "dbf31d5899a86047ce0cd4d0d12cdc5616df1bea5d9a600b823098442c32997c"
STOPS = (248044, 248046)
MAX_NEW_TOKENS = 16384


def strict_json(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number: " + value)
        return number

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=finite_constant, parse_float=finite_float)


def exact(actual, expected):
    """Require literal types, so JSON 0/1 cannot stand in for false/true."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(left, right) for left, right in zip(actual, expected))
    return actual == expected


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(fixture=FIXTURE):
    require(sha(SOURCE_REPORT) == SOURCE_REPORT_SHA256, "prior exact-source audit changed")
    source = strict_json(SOURCE_REPORT.read_text())
    names = ("README.md", "LICENSE", "chat_template.jinja", "preprocessor_config.json",
             "generation_config.json", "prompts/document_markdown.txt")
    for name in names:
        expected = (source["prompt_files"] if name.startswith("prompts/") else source["source_files"])[name]
        data = (Path(fixture) / name).read_bytes()
        require(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"],
                "pinned Markdown source bytes changed: " + name)
    return source


def validate_contract(contract, fixture=FIXTURE):
    source = verify_sources(fixture)
    require(isinstance(contract, dict), "Markdown task contract must be an object")
    require(contract.get("schema") == "crispembed.ovisocr3.document-markdown.protocol.v1" and
            contract.get("model") == MODEL and contract.get("revision") == REVISION,
            "Markdown task source identity differs")
    require(exact(contract.get("checkpoint"), {"file": "model.safetensors", "bytes": 1706030528,
                "sha256": "aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600",
                "tensor_count": 473, "mtp_hidden_layers": 0}), "Markdown checkpoint identity differs")
    source_names = ("README.md", "LICENSE", "chat_template.jinja", "preprocessor_config.json",
                    "generation_config.json", "prompts/document_markdown.txt")
    require(exact(contract.get("source_files"), {name: (source["prompt_files"] if name.startswith("prompts/")
                else source["source_files"])[name] for name in source_names}), "Markdown source manifest differs")
    task = contract.get("task", {})
    require(exact(task, {"name": "document_markdown", "prompt_file": "prompts/document_markdown.txt",
                     "prompt_bytes": 512, "prompt_sha256": PROMPT_SHA256, "images_per_request": 1,
                     "message_order": ["image", "task_text"], "add_generation_prompt": True,
                     "enable_thinking": False, "rendered_prompt_sha256": None}),
            "separate Markdown task/prompt contract differs")
    require(exact(contract.get("generation"), {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
                "do_sample": False, "source_eos_ids": list(STOPS), "custom_stop_strings": [],
                "actual_markdown_stop_contract_captured": False,
                "budget_origin": "unchanged upstream README example; separate task, not an old-panel cap change"}),
            "frozen Markdown generation contract differs")
    require(exact(contract.get("processor"), {"rgb": True, "kwargs_path": "mm_processor_kwargs.images_kwargs",
                "min_pixels": 448 * 448, "max_pixels": 2880 * 2880,
                "saved_processor_class": "Qwen3VLProcessor", "saved_image_class": "Qwen2VLImageProcessorFast",
                "installed_markdown_processor_captured": False,
                "native_existing_patch_bound": 1024, "full_upstream_geometry_qualified": False}),
            "Markdown processor contract differs")
    require(exact(contract.get("postprocess"), {"source": "README.md",
                "ordered_steps": ["strip", "optional_image_tag_block_filter", "_clean_truncated_repeats"],
                "upstream_filter_imgtags_default": True, "retain_raw_and_both_filter_variants": True,
                "applies_regardless_of_finish_reason": True, "cleanup_can_establish_completion": False,
                "cleanup_defaults": {"min_text_len": 8000, "max_period": 200, "min_period": 1,
                                     "min_repeat_chars": 100, "min_repeat_times": 5}}),
            "Markdown cleanup contract differs")
    require(exact(contract.get("qualification"), {"original_metadata": False, "original_neural": False,
                "native_profile": False, "native_numerical": False, "source_quality": False,
                "benchmark": False, "alias": False, "publication": False}),
            "source audit cannot grant model qualification")
    require(exact(contract.get("evaluation"), {
        "frozen_input_gold_manifest": None, "outputs_observed": False,
        "required_before_outputs": [
            "case IDs and image/RGB SHA-256", "licensed or authored images and provenance",
            "independent Markdown/text/table/formula/image-reference annotations",
            "case split and output schema", "task prompt, token cap, processor settings and scorer version"],
        "minimum_case_families": [
            "multicolumn reading order", "headings and lists", "HTML table cells and spans",
            "LaTeX formulas", "visual region references", "multilingual document text"],
        "required_capture": [
            "raw token IDs, terminal EOS/finish reason and explicit forced-EOS evidence", "raw Markdown before any cleanup",
            "upstream default and image-preserving processed variants",
            "rendered prompt, input IDs, packed pixels, attention masks and image grids",
            "every generated-position full-vocabulary raw logit row on the original prefix",
            "original module stages/cache and cold/warm timings"],
        "completion_rule": "raw terminal EOS plus non-length finish and no forced EOS or post-EOS tokens; cleanup cannot repair completion",
        "quality_rule": "evaluate content, reading order, tables, formulas and image references separately; no invented text-region boxes",
        "benchmark_score_claim": False}),
        "source-only protocol cannot alter evaluation gates or substitute for frozen input/gold")
    require(exact(contract.get("upstream_runtime"), {"implementation": "vllm", "version": "0.22.1",
                "source_commit": source["original_example"]["vllm_tag_commit"], "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8, "gdn_prefill_backend": "triton", "limit_mm_per_prompt_images": 2}),
            "upstream runtime recipe differs")
    prior = contract.get("preserved_plain_ocr", {})
    expected_files = {
        "tests/results/qwen35-ocr/ovisocr3-quality-panel.json": "cfdce3132f21a0188a034c015b3b0ce46d62e0fde6ad84697972bab517b86897",
        "tests/results/qwen35-ocr/ovisocr3-original-quality.json": "10a11d6aa9b8389bcb2c69417290109e2fa3c8f16a236fa7325a076d6da17c7e",
        "tests/results/qwen35-ocr/ovisocr3-f16-quality.json": "fc110bc5fca6aa6eb324527897a8a2db032cbca9c524e6e8c9b05dd61a87839b",
    }
    require(prior.get("files_sha256") == expected_files and all(sha(ROOT / name) == digest for name, digest in expected_files.items()),
            "frozen plain OCR evidence changed")
    require(prior.get("task") == "plain_ocr" and prior.get("two_column_cap") == 1024 and
            prior.get("prompt_sha256") == source["prompt_files"]["prompts/plain_ocr.txt"]["sha256"] and
            prior.get("source_quality_qualified") is False and prior.get("superseded") is False,
            "separate Markdown task must preserve old plain OCR failure")
    panel = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-quality-panel.json").read_text())
    failed = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-original-quality.json").read_text())["producer"]
    case = next(row for row in panel["cases"] if row["id"] == "two-column-page")
    scores = failed["scores"]["cases"]["two-column-page"]
    require(case["cap"] == 1024 and scores["selected_token_count"] == 1024 and scores["cap_hit"] is True and
            scores["complete"] is False and scores["stop_id"] is None,
            "old capped two-column failure was not retained")
    return contract


def reference_cleanup(fixture=FIXTURE):
    """Compile only the unmodified, hash-verified upstream pure string helper."""
    verify_sources(fixture)
    readme = (Path(fixture) / "README.md").read_text()
    functions = [node for block in re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
                 for node in ast.parse(block).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_clean_truncated_repeats"]
    require(len(functions) == 1, "upstream Markdown cleanup function is missing or duplicated")
    namespace = {"__builtins__": {"len": len, "min": min, "range": range, "str": str}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<pinned-ovisocr3-readme-helper>", "exec"), namespace)
    return namespace["_clean_truncated_repeats"]


def observe_output(raw_markdown, token_ids, finish_reason, *, forced_eos, fixture=FIXTURE):
    """Retain upstream presentation variants; completion uses raw generation evidence."""
    require(isinstance(raw_markdown, str) and isinstance(token_ids, list) and 0 < len(token_ids) <= MAX_NEW_TOKENS and
            all(type(token) is int and 0 <= token < 248320 for token in token_ids), "invalid Markdown raw output/IDs")
    require(finish_reason in ("stop", "length"), "unsupported original vLLM finish reason")
    require(type(forced_eos) is bool, "forced EOS evidence must be an explicit boolean")
    require(not any(token in STOPS for token in token_ids[:-1]), "generation contains tokens after EOS")
    clean = reference_cleanup(fixture)
    stripped = raw_markdown.strip()
    filtered = "\n\n".join(block for block in stripped.split("\n\n")
                            if not block.strip().startswith('<img src="images/bbox_'))
    complete = finish_reason == "stop" and token_ids[-1] in STOPS and not forced_eos
    return {"raw_markdown": raw_markdown, "token_ids": token_ids.copy(), "finish_reason": finish_reason,
            "terminal_eos": token_ids[-1] if token_ids[-1] in STOPS else None,
            "forced_eos": forced_eos, "token_budget_reached": len(token_ids) == MAX_NEW_TOKENS,
            "generation_complete": complete, "upstream_default_markdown": clean(filtered),
            "image_preserving_markdown": clean(stripped), "task_qualified": False,
            "cleanup_changes_completion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = validate_contract(strict_json(args.contract.read_text()))
    print(json.dumps({"source_contract_pass": True, "task": contract["task"]["name"],
                      "protocol_sha256": sha(args.contract), "qualification": contract["qualification"],
                      "next_gate": "freeze licensed document input/gold manifest, then original task metadata and outputs"}, indent=2))


if __name__ == "__main__":
    main()
