"""Hosted OvisOCR3 original plain-OCR eleven-case capture; F16 follows reviewed 14-arm replay."""

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import qwen35_ovis3_profile as policy


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


contract = module("ovisocr3_quality_contract", ROOT / "tools/ci-heavy/ovisocr3_quality_contract.py")
reference = module("ovisocr3_reference_gate", ROOT / "tools/ci-heavy/ovisocr3_reference.py")
dumper = reference.dumper

# Bind only after the original F32 per-module archive is independently reviewed.
ORIGINAL_REFERENCE_RUN = 40177416485
ORIGINAL_REFERENCE_HEAD = "2ce861682089e3390ba17b08ccba1e46a7eb598d"
ORIGINAL_REFERENCE_ARTIFACT_ID = 12592918499
ORIGINAL_REFERENCE_ARTIFACT_SHA256 = "d4efc96a779d3054cc1565bd3cb23c5cb2ecaa08126632b5eff580961787eb33"
ORIGINAL_REFERENCE_PRODUCER_SHA256 = "fbd71d4a40df55db3314ce4221ac036512c7c58711a5afe74f20dd5f7d1fccc3"
ORIGINAL_REFERENCE_RECEIPT = ROOT / "tests/results/qwen35-ocr/ovisocr3-original-reference.json"


def sha(path):
    return policy._file_sha256(path)


def require_original_reference():
    if (ORIGINAL_REFERENCE_RUN <= 0 or ORIGINAL_REFERENCE_ARTIFACT_ID <= 0
            or len(ORIGINAL_REFERENCE_HEAD) != 40 or len(ORIGINAL_REFERENCE_ARTIFACT_SHA256) != 64
            or len(ORIGINAL_REFERENCE_PRODUCER_SHA256) != 64):
        raise ValueError("Reviewed original OvisOCR3 F32 module reference is absent")
    receipt = json.loads(ORIGINAL_REFERENCE_RECEIPT.read_text())
    producer, review = receipt.get("producer"), receipt.get("independent_review")
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError("Reviewed original OvisOCR3 reference producer/review absent")
    canonical = policy._canonical_sha256(producer)
    if (receipt.get("run") != ORIGINAL_REFERENCE_RUN or receipt.get("head_sha") != ORIGINAL_REFERENCE_HEAD
            or receipt.get("artifact_id") != ORIGINAL_REFERENCE_ARTIFACT_ID
            or receipt.get("artifact_zip_sha256") != ORIGINAL_REFERENCE_ARTIFACT_SHA256
            or canonical != ORIGINAL_REFERENCE_PRODUCER_SHA256
            or review.get("producer_canonical_sha256") != canonical
            or review.get("original_reference_positive") is not True
            or producer.get("status") != "completed original OvisOCR3 HF CPU/F32 reference; native parity pending"
            or producer.get("checkpoint_sha256") != policy.CHECKPOINT_SHA256
            or producer.get("metadata_producer_canonical_sha256") != dumper.METADATA_PRODUCER_SHA256
            or producer.get("stage_count", 0) < 888):
        raise ValueError("Reviewed original OvisOCR3 F32 reference identity/coverage differs")
    return producer


def verify_prepared_archive(archive, preservation, prior):
    """Exactly four immutable reviewed members, no archive path extraction."""
    if Path(archive).stat().st_size != preservation["bytes"] or sha(archive) != preservation["sha256"]:
        raise ValueError("Prepared document ZIP size/SHA differs")
    cases = preservation["source"]["case_inputs"]
    expected = {"prepared.json": prior["artifact_report_sha256"]}
    expected.update({row["file"]: row["png_sha256"] for row in cases})
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if (len(infos) != 4 or {item.filename for item in infos} != set(expected)
                or any(item.filename != Path(item.filename).name or item.is_dir() or item.flag_bits & 1
                       or stat.S_ISLNK(item.external_attr >> 16) or item.file_size > preservation["bytes"]
                       for item in infos)):
            raise ValueError("Prepared document ZIP member/path differs")
        members = {item.filename: zipped.read(item) for item in infos}
    if any(hashlib.sha256(value).hexdigest() != expected[name] for name, value in members.items()):
        raise ValueError("Prepared document member SHA differs")
    prepared = json.loads(members["prepared.json"])
    if prepared.get("cases") != cases or prepared.get("source_commit") != prior["source_commit"]:
        raise ValueError("Prepared document provenance differs")
    return prepared, members


def prepare_inputs(destination):
    import importlib.metadata
    from PIL import Image, ImageDraw
    panel = contract.load_panel()
    if importlib.metadata.version("Pillow") != "12.1.0":
        raise ValueError("Pinned synthetic renderer differs")
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in panel["cases"][:8]:
        image = Image.new("RGB", tuple(panel["renderer"]["size"]), panel["renderer"]["background"])
        render = spec["renderer"]
        ImageDraw.Draw(image).text((render["x"], render["y"]), spec["gold"],
                                   fill=panel["renderer"]["fill"], font_size=render["font_size"])
        path = destination / (spec["id"] + ".png")
        image.save(path)
        cases.append({"id": spec["id"], "split": spec["split"], "license": spec["license"],
                      "path": path, "cap": spec["cap"], "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                      "png_sha256": sha(path)})
    preservation = json.loads(contract.PRESERVATION.read_text())
    prior = json.loads((ROOT / "tests/results/ocr-quality-prepared.json").read_text())
    archive = destination / "prepared.zip"
    with urllib.request.urlopen(preservation["url"], timeout=90) as response, archive.open("xb") as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            output.write(block)
            if output.tell() > preservation["bytes"]:
                raise ValueError("Prepared document ZIP exceeds frozen size")
    prepared, members = verify_prepared_archive(archive, preservation, prior)
    for spec, frozen in zip(panel["cases"][8:], prepared["cases"]):
        path = destination / frozen["file"]
        path.write_bytes(members[frozen["file"]])
        image = Image.open(path).convert("RGB")
        rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
        if (rgb_sha != spec["rgb_sha256"] or sha(path) != spec["png_sha256"]
                or frozen["gold_sha256"] != spec["gold_sha256"]):
            raise ValueError("Reviewed licensed document pixels/gold differ: " + spec["id"])
        cases.append({"id": spec["id"], "split": spec["split"], "license": spec["license"],
                      "path": path, "cap": spec["cap"], "rgb_sha256": rgb_sha,
                      "png_sha256": sha(path)})
    archive.unlink()  # Only the verified task-owned transport ZIP; preserved at immutable mirror.
    return cases, {"prepared_zip_sha256": preservation["sha256"], "prepared_zip_bytes": preservation["bytes"],
                   "prepared_json_sha256": hashlib.sha256(members["prepared.json"]).hexdigest(),
                   "synthetic_renderer": panel["renderer"], "case_count": len(cases)}


def original_cases(directory, cases, producer, report, save):
    import gc
    import math
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from transformers import AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration
    if (torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "5.16.1"
            or tokenizers.__version__ != "0.23.3"):
        raise ValueError("Pinned OvisOCR3 original quality stack differs")
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    profile = policy.load_profile(directory, policy.MODEL, policy.REVISION)
    profile["processor"] = producer["processor"]
    profile["primary_eos"] = producer["tokenizer_eos"]
    profile["generation"] = producer["resolved_generation"]
    actual = {name: sha(inspect.getfile(cls)) for name, cls in (
        ("generation", GenerationConfig), ("model", Qwen3_5ForConditionalGeneration),
        ("processor", type(processor)), ("image_processor", type(processor.image_processor)),
        ("tokenizer", type(processor.tokenizer)))}
    if actual != producer["loaded_source_sha256"]:
        raise ValueError("OvisOCR3 original quality framework source differs")
    report["processor_contract"] = dumper.processor_contract(processor, profile)
    messages = [{"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": profile["task_prompt"]}]}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
    if hashlib.sha256(rendered.encode()).hexdigest() != producer["rendered_prompt_sha256"]:
        raise ValueError("OvisOCR3 quality rendered prompt differs")
    report["rendered_prompt_sha256"] = producer["rendered_prompt_sha256"]
    save()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").eval()
    report["generation_contract"] = dumper.generation_contract(model, profile, directory, GenerationConfig)
    save()
    outputs = {}
    for case in cases:
        image = Image.open(case["path"]).convert("RGB")
        if hashlib.sha256(image.tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("Frozen quality RGB differs: " + case["id"])
        inputs = processor(text=[rendered], images=[image], return_tensors="pt",
                           return_mm_token_type_ids=True,
                           images_kwargs={"min_pixels": 448 * 448, "max_pixels": 2880 * 2880})
        if "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
            raise ValueError("Original quality multimodal token types differ")
        start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=case["cap"], do_sample=False,
                                       use_cache=True, eos_token_id=[248044, 248046],
                                       return_dict_in_generate=True, output_scores=True)
        ids = generated.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        if not ids or len(ids) > case["cap"] or len(generated.scores) != len(ids):
            raise ValueError("Original quality generated ID/score coverage differs: " + case["id"])
        confidences = []
        for token, scores in zip(ids, generated.scores):
            if list(scores.shape) != [1, 248320]:
                raise ValueError("Original quality selected vocabulary differs")
            confidences.append(float(torch.softmax(scores[0].float(), dim=-1)[token]))
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
            raise ValueError("Original quality selected confidence differs")
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        stop = ids[-1] if ids[-1] in contract.STOPS else None
        row = {"rgb_sha256": case["rgb_sha256"], "png_sha256": case["png_sha256"],
               "input_ids_sha256": hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest(),
               "token_ids": ids, "raw_text": raw, "confidences": confidences,
               "mean_confidence": sum(confidences) / len(confidences),
               "stop_id": stop, "stop_kind": "eos" if stop is not None else "length",
               "cap": case["cap"], "wall_seconds": time.monotonic() - start,
               "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "confidence_source": "actual original generate output_scores softmax at selected IDs"}
        outputs[case["id"]] = row
        report["original_cases"] = outputs
        report["completed_case_count"] = len(outputs)
        save()
        del generated, inputs
        gc.collect()
    del model
    gc.collect()
    return outputs


def main(preflight=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-quality"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"phase": "preflight", "status": "started", "source_commit": os.environ.get("GITHUB_SHA"),
              "scope": "original OvisOCR3 HF CPU/F32 plain OCR eleven-case quality; no native/public claim",
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "tools/ci-heavy/ovisocr3_quality_run.py", "tools/ci-heavy/ovisocr3_quality_contract.py",
                  "tools/dump_ovisocr3_reference.py", "models/qwen35_ovis3_profile.py")}}
    report_path = out / "ovisocr3-original-quality.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        panel = contract.load_panel()
        original = require_original_reference()
        metadata, service = reference.metadata_service()
        report["metadata_service"] = service
        report["metadata_producer_canonical_sha256"] = policy._canonical_sha256(metadata)
        report["original_reference_producer_canonical_sha256"] = policy._canonical_sha256(original)
        report["panel_canonical_sha256"] = contract.PANEL_CANONICAL_SHA256
        save()
        if preflight:
            report["phase"] = "preflight-complete"
            report["status"] = "preflight passed; no model/framework loaded"
            save()
            return report
        from huggingface_hub import snapshot_download, hf_hub_download
        report["phase"] = "pinned-metadata-and-input-freeze"; save()
        directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                           allow_patterns=list(policy.metadata_files())))
        profile = policy.load_profile(directory, policy.MODEL, policy.REVISION)
        if {name: sha(directory / name) for name in policy.metadata_files()} != metadata["source_files_sha256"]:
            raise ValueError("OvisOCR3 quality metadata source differs")
        cases, inputs = prepare_inputs(scratch / "inputs")
        report["inputs"] = [{key: value for key, value in case.items() if key != "path"} for case in cases]
        report["input_preparation"] = inputs
        save()
        report["phase"] = "checkpoint-after-input-freeze"; save()
        checkpoint = Path(hf_hub_download(policy.MODEL, "model.safetensors", revision=policy.REVISION))
        report["checkpoint_sha256"] = policy.verify_checkpoint(checkpoint, profile)
        if checkpoint.parent.resolve() != directory.resolve():
            raise ValueError("OvisOCR3 quality checkpoint/metadata snapshots differ")
        report["phase"] = "original-eleven-case-forward"; save()
        outputs = original_cases(directory, cases, metadata, report, save)
        report["scores"] = contract.score_panel(cases, outputs)
        report["phase"] = "complete"
        report["status"] = "completed original OvisOCR3 HF CPU/F32 eleven-case plain OCR quality"
        save()
        return report
    except BaseException as error:
        report["status"] = "failed original OvisOCR3 plain OCR quality"
        report["error_type"] = type(error).__name__
        report["error_message"] = reference.failure_detail(error)
        save()
        raise


if __name__ == "__main__":
    main("--preflight" in sys.argv)
