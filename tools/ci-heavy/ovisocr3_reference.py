"""Hosted original OvisOCR3 CPU/F32 reference; requires reviewed metadata first."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import qwen35_ovis3_profile as policy

spec = importlib.util.spec_from_file_location("dump_ovisocr3_reference", ROOT / "tools/dump_ovisocr3_reference.py")
dumper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dumper)


def sha(path):
    return policy._file_sha256(path)


def canonical(value):
    return policy._canonical_sha256(value)


def failure_detail(error):
    text = str(error)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]+", "<credential>", text)
    text = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s\]\)]+)", "<path>", text)
    return text[:600]


def metadata_service(call=subprocess.check_output):
    producer = dumper.require_reviewed_metadata()
    run = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{dumper.METADATA_RUN}"]))
    artifact = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{dumper.METADATA_ARTIFACT_ID}"]))
    if (run.get("status") != "completed" or run.get("conclusion") != "success"
            or run.get("head_sha") != dumper.METADATA_HEAD
            or run.get("path") != ".github/workflows/ovisocr2-metadata.yml"
            or artifact.get("workflow_run", {}).get("id") != dumper.METADATA_RUN
            or artifact.get("name") != f"ovisocr3-metadata-{dumper.METADATA_RUN}"
            or artifact.get("digest") != "sha256:" + dumper.METADATA_ARTIFACT_SHA256
            or artifact.get("expired") is not False):
        raise ValueError("OvisOCR3 metadata service run/head/artifact differs")
    return producer, {"run": dumper.METADATA_RUN, "head": dumper.METADATA_HEAD,
                      "artifact_id": dumper.METADATA_ARTIFACT_ID,
                      "artifact_zip_sha256": dumper.METADATA_ARTIFACT_SHA256}


def replay_metadata_anchors(directory, producer, out, on_case=None):
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    if (transformers.__version__ != "5.16.1" or torch.__version__.split("+")[0] != "2.11.0"):
        raise ValueError("OvisOCR3 original reference CPU stack differs")
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    actual_processor = {"class_name": type(processor).__module__ + "." + type(processor).__name__,
                        "image_class": type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
                        "image_backend": getattr(processor.image_processor, "backend", None),
                        "image_config": json.loads(json.dumps(processor.image_processor.to_dict(), allow_nan=False))}
    if actual_processor != producer["processor"] or processor.tokenizer.eos_token_id != producer["tokenizer_eos"]:
        raise ValueError("OvisOCR3 original processor source/config differs")
    prompt = (Path(directory) / "prompts/plain_ocr.txt").read_text()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
    if hashlib.sha256(rendered.encode()).hexdigest() != producer["rendered_prompt_sha256"]:
        raise ValueError("OvisOCR3 original rendered prompt differs")
    expected = {entry["case"]: entry for entry in producer["inputs"]}
    if set(expected) != {"square", "odd", "small-control"}:
        raise ValueError("OvisOCR3 original metadata anchor set differs")
    results = []
    for label, width, height in (("square", 448, 448), ("odd", 449, 417), ("small-control", 96, 64)):
        y, x = np.indices((height, width))
        pixels = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                           (x + y * 7) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")
        rgb_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        inputs = processor(text=[rendered], images=[image], return_tensors="pt",
                           images_kwargs={"min_pixels": 448 * 448, "max_pixels": 2880 * 2880})
        arrays = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        archive = out / f"{label}-replay.npz"
        np.savez_compressed(archive, **arrays)
        fields = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()}
        entry = {"case": label, "rgb_sha256": rgb_sha, "fields": fields,
                 "grid": arrays["image_grid_thw"].tolist(), "ids": arrays["input_ids"].tolist(),
                 "attention_mask": arrays["attention_mask"].tolist(), "archive_sha256": sha(archive)}
        original = expected[label]
        entry["checks"] = {"rgb": rgb_sha == original["rgb_sha256"],
                           "fields": fields == original["fields"],
                           "grid": entry["grid"] == original["grid"],
                           "ids": entry["ids"] == original["ids"],
                           "attention_mask": entry["attention_mask"] == original["attention_mask"]}
        results.append(entry)
        if on_case is not None:
            on_case(entry)
        if not all(entry["checks"].values()):
            raise ValueError("OvisOCR3 Torch2.11 processor anchor differs: " + label)
    return results


def main(preflight_only=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-reference"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = out / "ovisocr3-original-reference.json"
    report = {"status": "started", "phase": "reviewed-metadata-preflight", "source_commit": os.environ.get("GITHUB_SHA"),
              "scope": "original OvisOCR3 HF5.16.1 CPU/F32 per-module reference; no native/public qualification",
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "models/qwen35_ovis3_profile.py", "tools/dump_ovisocr3_reference.py",
                  "tools/ci-heavy/ovisocr3_reference.py", "models/convert-qwen35-ocr-to-gguf.py")}}
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        producer, service = metadata_service()
        report["metadata_service"] = service
        report["metadata_producer_canonical_sha256"] = canonical(producer)
        report["phase"] = "metadata-preflight-complete"; save()
        if preflight_only:
            report["status"] = "preflight passed; no framework/weight loaded"; save()
            return report
        import numpy as np
        import torch
        import transformers
        import tokenizers
        import gguf
        from PIL import Image, ImageDraw
        from huggingface_hub import snapshot_download, hf_hub_download
        if (torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "5.16.1"
                or tokenizers.__version__ != "0.23.3"):
            raise ValueError("Pinned OvisOCR3 reference environment differs")
        torch.set_num_threads(2)
        report["environment"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "tokenizers": tokenizers.__version__, "numpy": np.__version__,
                                 "gguf": getattr(gguf, "__version__", "unreported")}
        report["phase"] = "pinned-metadata-snapshot"; save()
        directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                           allow_patterns=list(policy.metadata_files())))
        profile = policy.load_profile(directory, policy.MODEL, policy.REVISION)
        policy.validate_config(json.loads((directory / "config.json").read_text()), profile)
        if {name: sha(directory / name) for name in policy.metadata_files()} != producer["source_files_sha256"]:
            raise ValueError("OvisOCR3 source metadata differs from reviewed receipt")
        report["phase"] = "original-processor-anchor-replay"; save()
        report["anchor_replay"] = []
        replay_metadata_anchors(directory, producer, out,
                                on_case=lambda entry: (report["anchor_replay"].append(entry), save()))
        save()
        image = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(image).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
        image_path = scratch / "owned-readable.png"
        image.save(image_path)
        report["owned_readable"] = {"file_sha256": sha(image_path),
                                    "rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                                    "size": [448, 448], "gold": "Hello OCR\n12345",
                                    "scope": "owned smoke input only; no quality benchmark"}
        save()
        report["phase"] = "checkpoint-download-after-anchors"; save()
        checkpoint = Path(hf_hub_download(policy.MODEL, "model.safetensors", revision=policy.REVISION))
        report["checkpoint_sha256"] = policy.verify_checkpoint(checkpoint, profile)
        if checkpoint.parent.resolve() != directory.resolve():
            raise ValueError("OvisOCR3 metadata/checkpoint snapshots differ")
        report["phase"] = "original-hf-f32-capture"; save()
        reference = scratch / "ovisocr3-original.npz"
        command = [sys.executable, str(ROOT / "tools/dump_ovisocr3_reference.py"),
                   "--model-dir", str(directory), "--image", str(image_path),
                   "--output", str(reference), "--max-new-tokens", "12", "--threads", "2"]
        with (out / "capture.log").open("wb") as log:
            finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False)
        report["capture_exit_code"] = finished.returncode
        report["capture_log_sha256"] = sha(out / "capture.log")
        report["reference_artifacts"] = {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                         for name, path in (("npz", reference),
                                                            ("json", reference.with_suffix(".json")),
                                                            ("gguf", scratch / "official-delta.gguf")) if path.exists()}
        save()
        if finished.returncode != 0 or set(report["reference_artifacts"]) != {"npz", "json", "gguf"}:
            raise ValueError("OvisOCR3 original HF capture failed; retained partial reports")
        actual = json.loads(reference.with_suffix(".json").read_text())
        if (actual.get("missing_stages") or actual.get("cached_vs_full_prefix_same_argmax") is not True
                or not all(actual.get("observational_control", {}).values())
                or actual.get("checkpoint_sha256") != policy.CHECKPOINT_SHA256):
            raise ValueError("OvisOCR3 original per-module capture controls differ")
        report["reference_summary"] = {key: actual.get(key) for key in (
            "generated_ids", "generated_stop_id", "decoded", "raw_decoded", "generation_result",
            "cached_vs_full_prefix_max_abs", "cached_vs_full_prefix_same_argmax")}
        report["stage_count"] = len(actual["stages"])
        report["status"] = "completed original OvisOCR3 HF CPU/F32 reference; native parity pending"
        report["phase"] = "complete"; save()
        return report
    except BaseException as error:
        report["status"] = "failed original OvisOCR3 reference"
        report["error_type"] = type(error).__name__
        report["error_message"] = failure_detail(error)
        save()
        raise


if __name__ == "__main__":
    main("--preflight" in sys.argv)
