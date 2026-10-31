#!/usr/bin/env python3
"""Convert the pinned source, then execute a GGUF-fed Transformers GPU control.

The staging script embeds the exact converter, profile and v3 evidence bytes.
This is a conversion/round-trip control on the GPU, not a native parity test.
"""
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zlib

# Replaced only in the staged Kaggle script; not an executable source default.
BUNDLE = "__CRISPEMBED_SOURCE_BUNDLE__"
WORK = Path("/kaggle/working/xiaomi-ocr-0-export")
ROOT = Path("/kaggle/temp/xiaomi-ocr-0-export/source")
REPORT = {"scope": "F32 conversion and GGUF-fed original GPU control; no native parity",
          "publication": "none", "native_alias_qualified": False, "status": "running"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(phase):
    REPORT["phase"] = phase
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")
    print("===", phase, flush=True)


def unpack():
    if BUNDLE == "__CRISPEMBED_SOURCE_BUNDLE__":
        raise ValueError("Unstaged source bundle")
    package = json.loads(zlib.decompress(base64.b64decode(BUNDLE)))
    files = package["files"]
    allowed = {"models/convert-qwen35-ocr-to-gguf.py", "models/qwen35_xiaomi_profile.py",
               "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"}
    if set(files) != allowed:
        raise ValueError("Source closure differs")
    for name, record in files.items():
        content = base64.b64decode(record["content"], validate=True)
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Embedded source hash differs: " + name)
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    REPORT["source_closure"] = {"base_commit": package["base_commit"],
                                "runner_template_sha256": package["runner_template_sha256"],
                                "staged_runner_sha256": digest(__file__),
                                "files_sha256": {name: row["sha256"] for name, row in files.items()}}
    (WORK / "source-closure.json").write_text(json.dumps(REPORT["source_closure"], indent=2) + "\n")


def find_frozen(name, expected_sha256):
    found = list(Path("/kaggle/input").rglob(name))
    matches = [path for path in found if digest(path) == expected_sha256]
    if len(matches) != 1:
        raise ValueError(f"Frozen input {name}: {len(matches)} digest matches")
    return matches[0]


def validate_frozen_fields(case, fields):
    if set(fields.files) != set(case["fields"]):
        raise ValueError("Frozen processor field names differ: " + case["name"])
    for name, expected in case["fields"].items():
        array = fields[name]
        if (len(expected) != 3 or list(array.shape) != expected[0] or str(array.dtype) != expected[1]
                or hashlib.sha256(array.tobytes()).hexdigest() != expected[2]):
            raise ValueError("Frozen processor field differs: " + case["name"] + "/" + name)


def main():
    save("resource-admission")
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("This GPU control must run on Kaggle")
    free = shutil.disk_usage("/kaggle/working").free
    REPORT["initial_free_bytes"] = free
    if free < 14 * 1024 ** 3:
        raise RuntimeError("Insufficient hosted disk for checkpoint and F32 conversion")
    unpack()
    save("install")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "transformers==5.8.1", "tokenizers==0.22.2", "gguf==0.19.0",
                    "safetensors==0.8.0"], check=True)
    import gguf
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("GPU control requires CUDA")
    REPORT["versions"] = {"transformers": transformers.__version__, "numpy": np.__version__,
                          "torch": torch.__version__,
                          **{name: importlib.metadata.version(name) for name in
                             ("tokenizers", "gguf", "safetensors", "huggingface-hub")}}
    REPORT["device"] = torch.cuda.get_device_name(0)
    evidence = json.loads((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json").read_text())
    frozen = {case["name"]: find_frozen(case["name"] + "-processor.npz", case["processor_sha256"])
              for case in evidence["cases"]}
    for case in evidence["cases"]:
        with np.load(frozen[case["name"]], allow_pickle=False) as fields:
            validate_frozen_fields(case, fields)
    REPORT["input_cases"] = list(frozen)
    spec = importlib.util.spec_from_file_location("xiaomi_export", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load embedded converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    policy = converter._xiaomi_profile()
    save("download-metadata")
    local = Path(snapshot_download(policy.MODEL, revision=policy.REVISION, allow_patterns=list(policy.metadata_files())))
    policy.load_profile(local, policy.MODEL, policy.REVISION)
    save("download-checkpoint")
    local = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                   allow_patterns=list(policy.metadata_files()) + ["*.safetensors"]))
    export = WORK / "xiaomi-ocr-0-f32-experimental.gguf"
    save("convert-f32")
    manifest = converter.convert(local, export, policy.MODEL, policy.REVISION, dtype="f32", profile="xiaomi")
    REPORT["export"] = {"bytes": export.stat().st_size, "sha256": digest(export),
                        "records": len(manifest["records"]), "dtype": manifest["dtype"],
                        "manifest_sha256": digest(export.with_suffix(".gguf.manifest.json"))}
    save("gguf-roundtrip")
    reader = gguf.GGUFReader(str(export))
    records = {r["name"]: r for r in manifest["records"]}
    tensors = {t.name: t for t in reader.tensors}
    if len(records) != 473 or set(records) != set(tensors):
        raise ValueError("GGUF tensor set differs from audited export")
    for name, row in records.items():
        data = tensors[name].data
        if (list(data.shape) != row["shape"] or str(data.dtype) != "float32"
                or hashlib.sha256(memoryview(data).cast("B")).hexdigest() != row["sha256"]):
            raise ValueError("GGUF roundtrip bytes or shape differ: " + name)
    REPORT["roundtrip_tensors"] = len(records)
    save("load-original-and-replace-from-gguf")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        local, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).eval()
    state = model.state_dict()
    copied = 0
    with torch.no_grad():
        for name, row in records.items():
            target = state[row["source"]]
            data = tensors[name].data
            if list(target.shape) != row["source_shape"] or not target.is_contiguous():
                raise ValueError("GGUF original-model shape/layout differs: " + name)
            source_flat, target_flat = data.reshape(-1), target.view(-1)
            # Compare widened original BF16 values directly in F32 before any
            # rounding back to BF16; bounded chunks avoid duplicating embeddings.
            for start in range(0, target.numel(), 4 * 1024 * 1024):
                end = min(start + 4 * 1024 * 1024, target.numel())
                value = torch.from_numpy(np.array(source_flat[start:end], copy=True))
                if not torch.equal(target_flat[start:end].float(), value):
                    raise ValueError("GGUF F32 values differ from loaded checkpoint: " + name)
                target_flat[start:end].copy_(value.to(dtype=target.dtype))
            copied += 1
    if copied != 473 or model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise ValueError("GGUF model tensor or tied-head count differs")
    REPORT["checkpoint_equal_and_reloaded_tensors"] = copied
    del state, reader, tensors
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(local, trust_remote_code=False, local_files_only=True)
    original = {item["name"]: item for item in evidence["original_gpu"]}
    REPORT["gpu_controls"] = []
    save("gpu-greedy-control")
    for case in evidence["cases"]:
        name = case["name"]
        with np.load(frozen[name], allow_pickle=False) as fields:
            inputs = {key: torch.from_numpy(fields[key]).to("cuda") for key in fields.files}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        expected = original[name]
        result = {"name": name, "ids_equal": ids == expected["generated_ids"],
                  "text_sha256": text_sha, "text_equal": text_sha == expected["text_sha256"],
                  "generated_count": len(ids), "capped": len(ids) >= 512, "seconds": seconds}
        REPORT["gpu_controls"].append(result)
        (WORK / (name + "-gguf-gpu.txt")).write_text(text)
        save("gpu-greedy-control")
        if not result["ids_equal"] or not result["text_equal"] or result["capped"] != expected["capped"]:
            raise ValueError("GGUF-fed original GPU generation differs: " + name)
    REPORT["status"] = "completed GGUF-fed original GPU control; no native parity claimed"
    save("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT["status"] = "failed GGUF-fed original GPU control"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1200]
        REPORT["traceback"] = traceback.format_exc()[-4000:]
        save(REPORT.get("phase", "startup"))
        raise
