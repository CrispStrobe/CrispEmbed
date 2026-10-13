#!/usr/bin/env python3
"""Hosted LightOnOCR-3-0.8B export/reference checkpoint; no native parity claim."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    import gguf
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from PIL import Image, ImageDraw
    from safetensors.numpy import save_file

    if transformers.__version__ != "5.5.4":
        raise ValueError("Reference implementation must be pinned to transformers==5.5.4")
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"])
    converter = module(repo / "models/convert-qwen35-ocr-to-gguf.py", "qwen35_export")
    reference = module(repo / "tools/dump_qwen35_ocr_reference.py", "qwen35_reference")
    report = {"native_parity": "not implemented", "publication": "none",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    try:
        api = HfApi()
        # Resolve metadata and verify permissive terms before downloading weights.
        sources = [(converter.LIGHTON_MODEL, converter.LIGHTON_REVISION),
                   ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")]
        provenance = []
        for model, revision in sources:
            info = api.model_info(model, revision=revision)
            card = info.card_data.to_dict()
            if info.sha != revision or card.get("license") != "apache-2.0":
                raise ValueError(f"Unexpected source revision/license for {model}")
            readme = Path(hf_hub_download(model, "README.md", revision=revision))
            entry = {"model": model, "revision": revision, "license": card["license"],
                     "readme_sha256": hashlib.sha256(readme.read_bytes()).hexdigest()}
            (out / (model.replace("/", "--") + "-README.md")).write_bytes(readme.read_bytes())
            if model.startswith("Qwen/"):
                license_file = Path(hf_hub_download(model, "LICENSE", revision=revision))
                text = license_file.read_text()
                if "Apache License" not in text or "Version 2.0" not in text:
                    raise ValueError("Base model license text is not Apache-2.0")
                (out / "base-LICENSE.txt").write_text(text)
                entry["license_sha256"] = hashlib.sha256(license_file.read_bytes()).hexdigest()
            provenance.append(entry)
        report["provenance"] = provenance
        local = Path(snapshot_download(converter.LIGHTON_MODEL, revision=converter.LIGHTON_REVISION,
                                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE*"]))
        # Freeze actual configuration and schema provenance with the report.
        for filename in ["config.json", "generation_config.json", "processor_config.json",
                         "preprocessor_config.json", "tokenizer_config.json", "chat_template.jinja"]:
            (out / filename).write_bytes((local / filename).read_bytes())
        export = scratch / "lightonocr-3-0.8b-f16-experimental.gguf"
        manifest = converter.convert(local, export, converter.LIGHTON_MODEL, converter.LIGHTON_REVISION)
        (out / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        reader = gguf.GGUFReader(str(export))
        expected = {record["name"]: record for record in manifest["records"]}
        if set(expected) != {tensor.name for tensor in reader.tensors}:
            raise ValueError("GGUF roundtrip missing/extra tensors")
        for tensor in reader.tensors:
            entry = expected[tensor.name]
            if list(tensor.data.shape) != entry["shape"]:
                raise ValueError(f"GGUF roundtrip shape mismatch: {tensor.name}")
            if hashlib.sha256(tensor.data.tobytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"GGUF roundtrip byte mismatch: {tensor.name}")
        report["roundtrip_tensors"] = len(expected)
        report["export_bytes"] = manifest["bytes"]
        del reader

        # Malformed checkpoint controls must fail before creating a usable output.
        negative = []
        for name, array in [("model.language_model.embed_tokens.weight", np.zeros((1, 1), np.float32)),
                            ("unexpected.adapter.weight", np.zeros((2, 2), np.float32))]:
            with tempfile.TemporaryDirectory(dir=scratch) as folder:
                bad = Path(folder)
                for metadata in local.glob("*.json"):
                    (bad / metadata.name).symlink_to(metadata)
                save_file({name: array}, bad / "model.safetensors")
                target = bad / "must-not-exist.gguf"
                try:
                    converter.convert(bad, target, converter.LIGHTON_MODEL, converter.LIGHTON_REVISION)
                except ValueError as error:
                    if "Tensor schema mismatch" not in str(error) or target.exists():
                        raise
                    negative.append(name)
                else:
                    raise AssertionError("Malformed checkpoint was accepted")
        report["rejected_checkpoint_controls"] = negative
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
        image_path = out / "reference-input.png"
        image.save(image_path)
        ref = reference.dump_reference(local, image_path, out / "official-reference.npz")
        report["reference"] = {key: ref[key] for key in ["vision_layers", "decoder_layers", "decoded",
                                                       "cached_vs_full_prefix_max_abs", "cached_vs_full_prefix_same_argmax"]}
        report["result"] = "export and official reference checks passed; native support pending"
    except BaseException as error:
        report["result"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "summary.md").write_text("# Qwen3.5 OCR bootstrap\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
