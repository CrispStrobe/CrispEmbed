#!/usr/bin/env python3
"""Compare official 2.0.7 document helpers with the existing native document route.

Both arms deliberately use the SAME native neural inference, isolating helper
semantics. Original Python neural parity is measured separately by
mineru2605_validate.py / mineru2605_qualify.py. This script does not bypass or
modify CrispMinerU.two_step_extract's frozen 2604 helper-source guard.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
from pathlib import Path
import sys
import time

import mineru2605_validate as protocol

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threads", type=protocol.positive_int, default=4)
    parser.add_argument("--max-tokens", type=protocol.token_limit, default=2048)
    parser.add_argument("--seed", type=int, default=2605)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol.verify_blueprint(args.blueprint)
    conversion = json.loads(Path(args.gguf + ".json").read_text())
    gguf_sha256 = protocol.sha256(args.gguf)
    if (conversion.get("model") != protocol.MODEL_ID or
            conversion.get("source_revision") != protocol.MODEL_REVISION or
            conversion.get("gguf_sha256") != gguf_sha256):
        raise ValueError("document probe requires the exact converted 2605 GGUF and its conversion receipt")
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispMinerU
    from PIL import Image
    from mineru_vl_utils.mineru_client import MinerUClientHelper, DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS
    from mineru_vl_utils.post_process import json2md

    args.out.mkdir(parents=True, exist_ok=True)
    page = Image.open(args.image).convert("RGB")
    helper = MinerUClientHelper(backend="transformers", prompts=deepcopy(DEFAULT_PROMPTS),
        sampling_params=deepcopy(DEFAULT_SAMPLING_PARAMS), layout_image_size=(1036, 1036),
        min_image_edge=28, max_image_edge_ratio=50, simple_post_process=False,
        handle_equation_block=True, abandon_list=False, abandon_paratext=False,
        image_analysis=True, debug=False, enable_table_formula_eq_wrap=False,
        enable_cross_page_table_merge=False)
    model = CrispMinerU(args.gguf, n_threads=args.threads, lib_path=args.lib,
                       image_analysis=True, allow_truncated_content=True)
    try:
        random.seed(args.seed)
        start = time.perf_counter()
        raw_layout = model.predict(helper.prepare_for_layout(page), helper.prompts["[layout]"], args.max_tokens)
        blocks = helper.parse_layout_output(raw_layout["text"])
        images, prompts, params, indices = helper.prepare_for_extract(page, blocks, None, True)
        if not (len(images) == len(prompts) == len(params) == len(indices)):
            raise RuntimeError("official 2.0.7 helper returned inconsistent crop requests")
        for image, prompt, sampling, index in zip(images, prompts, params, indices):
            if (sampling.temperature != 0.0 or sampling.top_k != 1 or sampling.repetition_penalty != 1.0 or
                    sampling.no_repeat_ngram_size != 100 or sampling.max_new_tokens is not None):
                raise RuntimeError("2.0.7 crop generation policy differs from the native MinerU profile")
            blocks[index].content = model.predict(image, prompt, args.max_tokens)["text"]
        official = helper.post_process(blocks)
        official_time = time.perf_counter() - start
        trace = list(model.last_trace)
        official_json = json.dumps(list(official), ensure_ascii=False, separators=(",", ":"))
        official_md = json2md(official)
        start = time.perf_counter()
        model.native_two_step_extract(page, max_new_tokens=args.max_tokens, table_token_seed=args.seed)
        native_time = time.perf_counter() - start
        native = model.last_native_document
        native_json, native_md = model.native_render("mineru-json"), model.native_render("mineru-markdown")
        (args.out / "official207.json").write_text(official_json)
        (args.out / "official207.md").write_text(official_md)
        (args.out / "native.json").write_text(native_json)
        (args.out / "native.md").write_text(native_md)
        protocol.write_json(args.out / "official207-requests.json", trace)
        protocol.write_json(args.out / "native-document.json", native)
        complete = all(t["result"].get("eos") and not t["result"].get("capped") for t in trace)
        report = {
            "schema": "crispembed.mineru2605.document-helpers.v1", "model": protocol.MODEL_ID,
            "revision": protocol.MODEL_REVISION, "official_helper": protocol.BLUEPRINT_REVISION,
            "native_helper_blueprint": "c9797cde8f54cc48306196fc6aebe73ea31dd4a3",
            "scope": "single page; both helper arms use the same native neural inference",
            "json_exact": native_json == official_json, "markdown_exact": native_md == official_md,
            "text_metrics": protocol.text_metrics(official_md, native_md),
            "official_blocks": len(official), "native_blocks": len(native.get("blocks", [])),
            "official_requests": len(trace), "reference_reached_eos": complete,
            "helper_parity_pass": complete and native_json == official_json and native_md == official_md,
            "original_python_neural_parity": "separate per-request qualification required",
            "cross_page_table_merge_qualified": False, "benchmark_qualified": False,
            "image_sha256": protocol.sha256(args.image), "gguf_sha256": gguf_sha256,
            "threads": args.threads, "table_token_seed": args.seed,
            "timing_s": {"official_helper_first": official_time, "native_helper_second": native_time,
                         "comparable_warm_speed": False},
        }
        protocol.write_json(args.out / "document.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["helper_parity_pass"] else 1
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
