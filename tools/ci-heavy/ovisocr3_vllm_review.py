"""Review retained small OvisOCR3 GPU reports without loading model packages."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "tools/kaggle/ovisocr3-vllm-reference/ovisocr3_vllm_reference.py"
WORKER = ROOT / "tools/kaggle/ovisocr3-vllm-reference/worker.py"
METADATA = ROOT / "tests/results/qwen35-ocr/ovisocr3-metadata.json"
RECIPE_COMMIT = "41be637befd1d0aaed049ef415854afe520c9526"
WORKER_COMMIT = "bd1f1fecc3d17d180286a9fb87393a18b830d9ca"
RECIPE_SHA256 = "af06e8a367b2539a1e198dec71acf94ed01e1fe3cf93a26d5ed92edbe9af62d4"
WORKER_SHA256 = "cc199f9ee10a49594d9c28d1c095992cc9a622e28b1fab73d9e4e3a509e61879"
FIRST_REPORT_SHA256 = "92e23a9dbeffc02d6816cbe69e774eea6388302c80b2544432a0d82cf15191b3"
FIRST_NOTEBOOK_LOG_SHA256 = "87a64eeb748d9ba0b3b1d4ea741e727fa9e6052f4658c1f3fd205763969aaabb"
SECOND_REPORT_SHA256 = "24fae4c0f6e348e2bfafceaf19d375e841f7230e9bf5f6c889e611a92692424c"
SECOND_NOTEBOOK_LOG_SHA256 = "35045471a75989bcec606abc770585866023401ee510b4e7c59e7cc9578e9ce4"
WORKER_REPORT_SHA256 = "af5cc204044b83d9542a096c01320134eae98115592e82a07c3841e694611aa9"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, digest):
    raw = Path(path).read_bytes()
    require(sha(raw) == digest, "Retained report SHA differs: " + Path(path).name)
    return json.loads(raw)


def review(first, second, metadata, logs):
    """Return a public receipt after checking exact retained producer records."""
    require(sha(RECIPE.read_bytes()) == RECIPE_SHA256 and sha(WORKER.read_bytes()) == WORKER_SHA256,
            "GPU recipe/worker source bytes differ")
    require(first.get("status") == "inconclusive" and first.get("phase") == "hardware"
            and first.get("error_type") == "ValueError"
            and first.get("error_message") == "Expected exactly one visible T4; no device redraw"
            and first.get("commands") == [] and "worker" not in first,
            "Initial hardware preflight record differs")
    require(second.get("status") == "completed bounded original OvisOCR3 vLLM T4 reference"
            and second.get("phase") == "complete"
            and second.get("script_sha256") == RECIPE_SHA256
            and second.get("worker_sha256") == WORKER_SHA256,
            "Completed runner source/status differs")
    worker = second.get("worker")
    require(isinstance(worker, dict)
            and worker.get("status") == "completed bounded original OvisOCR3 vLLM reference"
            and worker.get("phase") == "complete"
            and worker.get("gpu_control", {}).get("passed") is True
            and worker.get("resolved_vllm_dtype") == "torch.float16"
            and worker.get("resolved_decoder_attention_backend") == "TRITON_ATTN"
            and worker.get("attention_backend_override", {}).get("readme_auto_backend_equivalence_qualified") is False
            and worker.get("runtime_stop_probe", {}).get("all_stop_ids") == [248044, 248046]
            and worker.get("runtime_stop_probe", {}).get("primary_eos") == 248046,
            "Completed GPU execution/precision/stop proof differs")
    gpu = second.get("hardware", {})
    require(gpu.get("selected_physical_index") == 0 and gpu.get("visible_gpu_count", 0) >= 1
            and gpu.get("name") == "Tesla T4" and gpu.get("capability") == "7.5"
            and worker.get("gpu_control", {}).get("device") == "Tesla T4"
            and worker.get("gpu_control", {}).get("capability") == [7, 5],
            "Selected real T4 hardware proof differs")
    upstream = metadata.get("producer", {})
    require(metadata.get("independent_review", {}).get("metadata_positive") is True
            and metadata.get("independent_review", {}).get("producer_canonical_sha256") == canonical(upstream)
            and upstream.get("source_revision") == "5f6ce6932efee9ec10c30746db55bd780214b7fd"
            and second.get("metadata_proof", {}).get("producer_sha256") == canonical(upstream)
            and second.get("metadata_proof", {}).get("artifact_id") == metadata.get("artifact_id")
            and second.get("metadata_proof", {}).get("artifact_zip_sha256") == metadata.get("artifact_zip_sha256")
            and worker.get("checkpoint_sha256") == "aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600"
            and worker.get("source_files_sha256") == upstream.get("source_files_sha256")
            and worker.get("prompt", {}).get("rendered_sha256") == upstream.get("rendered_prompt_sha256")
            and worker.get("prompt", {}).get("rendered_utf8_bytes") == 531,
            "Reviewed metadata/source/checkpoint identity differs")
    expected = {row["case"]: row for row in upstream["inputs"]}
    cases = worker.get("processor_cases", [])
    require([row.get("case") for row in cases] == ["square", "odd", "small-control"],
            "Processor case coverage differs")
    for row in cases:
        original = expected[row["case"]]
        require(row.get("rgb_sha256") == original["rgb_sha256"]
                and row.get("actual_fields") == original["fields"]
                and row.get("expected_fields") == original["fields"]
                and row.get("exact_original_fields") is True
                and re.fullmatch(r"[0-9a-f]{64}", row.get("replay_sha256", "")),
                "Processor metadata fields differ: " + row["case"])
    readable = worker.get("owned_readable", {})
    require(readable.get("case") == "readable" and readable.get("width") == 448
            and readable.get("height") == 448 and readable.get("gold_smoke_only") == "Hello OCR\n12345"
            and all(re.fullmatch(r"[0-9a-f]{64}", readable.get(k, ""))
                    for k in ("rgb_sha256", "png_sha256")),
            "Owned readable image identity differs")
    generations = worker.get("generations", [])
    require([row.get("case") for row in generations] == ["square", "readable"],
            "Generated case coverage differs")
    summaries = []
    for row in generations:
        case = row["case"]
        expected_ids = expected["square"]["ids"][0] if case == "square" else readable["hf_prompt_ids"]
        ids = row.get("token_ids")
        scores = row.get("generated_scores")
        require(row.get("prompt_token_ids") == expected_ids
                and row.get("expected_hf_prompt_token_ids") == expected_ids
                and row.get("prompt_ids_exact") is True
                and row.get("temperature") == 0.0 and row.get("bounded_max_tokens") == 256
                and row.get("gdn_prefill_backend") == "triton"
                and row.get("decoder_attention_backend") == "TRITON_ATTN"
                and isinstance(ids, list) and 0 < len(ids) < 256 and ids[-1] == 248046
                and row.get("finish_reason") == "stop" and row.get("reached_eos") is True
                and row.get("cap_hit") is False and row.get("stop_reason") in (None, 248046)
                and isinstance(scores, list) and len(scores) == len(ids),
                "Generated prompt/IDs/EOS/cap differ: " + case)
        for key in ("raw_text", "stripped_text"):
            require(isinstance(row.get(key), str), "Decoded text missing: " + case)
        require(sha(row["raw_text"].encode()) == row.get("raw_text_sha256"),
                "Raw decoded text hash differs: " + case)
        for step, (token, entry) in enumerate(zip(ids, scores)):
            candidates = entry.get("scores", [])
            require(entry.get("step") == step and entry.get("selected_token_id") == token
                    and entry.get("top_requested") == 20
                    and isinstance(entry.get("selected_logprob"), (float, int))
                    and math.isfinite(entry["selected_logprob"])
                    and isinstance(candidates, list) and candidates
                    and any(candidate.get("token_id") == token for candidate in candidates)
                    and all(isinstance(candidate.get("logprob"), (float, int))
                            and math.isfinite(candidate["logprob"]) for candidate in candidates),
                    "Generated selected/top scores differ: " + case + "/" + str(step))
        summaries.append({"case": case, "token_ids": ids, "raw_text": row["raw_text"],
                          "generated_scores_complete": True, "reached_eos": True, "cap_hit": False,
                          "prompt_ids_exact": True})
    require(summaries[1]["raw_text"] == readable["gold_smoke_only"],
            "Readable OCR smoke text differs")
    commands = second.get("commands", [])
    require([row.get("name") for row in commands] == ["bootstrap-isolated-pip", "install-pinned-stack",
                                                       "package-freeze", "original-vllm-worker"],
            "Hosted command coverage differs")
    for row in commands:
        name = row["log"]
        require(Path(name).name == name and row.get("exit_code") == 0
                and row.get("timed_out") is False
                and sha(logs[name]) == row.get("log_sha256"),
                "Hosted command log/exit differs: " + name)
    engine = logs["original-vllm-worker.log"].decode(errors="replace")
    for marker in ("Using AttentionBackendEnum.TRITON_ATTN backend.",
                   "Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention",
                   "Using Triton/FLA GDN prefill kernel (requested=triton"):
        require(marker in engine, "Actual backend log selection missing")
    for value in (first, second):
        require(not re.search(r"(?:kaggle\.com/code/|/mnt/|/home/|/tmp/)",
                              json.dumps(value, ensure_ascii=False), re.I),
                "Public producer contains private run locator or host path")
    return {
        "schema": "crispembed-ovisocr3-vllm-gpu-reference-review-v1",
        "service": "Kaggle genuine GPU session",
        "recipe_source": RECIPE_COMMIT,
        "worker_source": WORKER_COMMIT,
        "recipe_path": "tools/kaggle/ovisocr3-vllm-reference/ovisocr3_vllm_reference.py",
        "worker_path": "tools/kaggle/ovisocr3-vllm-reference/worker.py",
        "producer_report_sha256": SECOND_REPORT_SHA256,
        "worker_report_sha256": WORKER_REPORT_SHA256,
        "initial_attempt": {
            "producer_report_sha256": FIRST_REPORT_SHA256,
            "notebook_log_sha256": FIRST_NOTEBOOK_LOG_SHA256,
            "producer_canonical_sha256": canonical(first),
            "producer": first,
            "independent_review": {"hardware_preflight_failed_before_packages_or_weights": True,
                                   "genuine_gpu_forward_completed": False},
        },
        "producer": second,
        "independent_review": {
            "genuine_gpu_forward_completed": True,
            "producer_canonical_sha256": canonical(second),
            "worker_canonical_sha256": canonical(worker),
            "notebook_log_sha256": SECOND_NOTEBOOK_LOG_SHA256,
            "metadata_producer_canonical_sha256": canonical(upstream),
            "processor_fields_matched_reviewed_metadata": 3,
            "selected_device": "Tesla T4, physical index 0",
            "actual_dtype": "torch.float16",
            "decoder_attention_backend": "TRITON_ATTN",
            "effective_stop_ids": [248044, 248046],
            "generated_scores_complete": True,
            "generations": summaries,
            "source_files_and_logs_verified": True,
            "limits": [
                "Controlled synthetic GPU reference only; no 11-case OCR quality or CPU native parity claim.",
                "Processor archive payloads remain remote; this small-report review compares their reported field hashes to reviewed metadata.",
                "README auto decoder attention and BF16 behavior are unqualified by this T4 FP16 run.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = read_pinned(args.first_dir / "ovisocr3-vllm-reference.json", FIRST_REPORT_SHA256)
    second = read_pinned(args.second_dir / "ovisocr3-vllm-reference.json", SECOND_REPORT_SHA256)
    worker = read_pinned(args.second_dir / "ovisocr3-vllm-worker.json", WORKER_REPORT_SHA256)
    require(sha((args.first_dir / "crispembed-ovisocr3-pinned-vllm-reference.log").read_bytes())
            == FIRST_NOTEBOOK_LOG_SHA256, "Initial notebook log SHA differs")
    require(sha((args.second_dir / "crispembed-ovisocr3-pinned-vllm-reference.log").read_bytes())
            == SECOND_NOTEBOOK_LOG_SHA256, "Completed notebook log SHA differs")
    require(second.get("worker") == worker, "Separate retained worker JSON differs")
    metadata = json.loads(METADATA.read_text())
    logs = {row["log"]: (args.second_dir / "ovisocr3-vllm-logs" / row["log"]).read_bytes()
            for row in second.get("commands", [])}
    receipt = review(first, second, metadata, logs)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
