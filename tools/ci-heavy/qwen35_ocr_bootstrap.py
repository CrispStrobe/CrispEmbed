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
    import gc
    import gguf
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from PIL import Image, ImageDraw
    from safetensors.numpy import save_file

    if transformers.__version__ != "5.5.4" or tokenizers.__version__ != "0.22.2":
        raise ValueError("Reference requires transformers==5.5.4 and tokenizers==0.22.2")
    precision_audit = sys.argv[1:] == ["--vision-precision"]
    math_audit = sys.argv[1:] == ["--vision-math"]
    frontend_keys = sys.argv[1:] == ["--frontend-key-f32"]
    frontend_vision = sys.argv[1:] == ["--frontend-vision-f32"]
    frontend_rms = sys.argv[1:] == ["--frontend-precise-rms"]
    if sys.argv[1:] and not (precision_audit or math_audit or frontend_keys or frontend_vision or frontend_rms):
        raise ValueError("Unexpected arguments")
    # Set explicitly for all native child processes; never inherit an accidental
    # operator setting into the baseline. The original Python reference is unchanged.
    os.environ["CRISPEMBED_QWEN35_PRECISE_RMS"] = "1" if frontend_rms else "0"
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
        keep_f32 = [f"l.layers.{i}.self_attn.k_proj.weight" for i in range(3,24,4)] if frontend_keys else []
        if frontend_vision:
            keep_f32 = converter.precision_recipe("vision-f32")
        manifest = converter.convert(local, export, converter.LIGHTON_MODEL, converter.LIGHTON_REVISION, keep_f32=keep_f32)
        report["weight_recipe"] = "vision-f32" if frontend_vision else "full-attention-k-f32" if frontend_keys else "original-f16"
        report["keep_f32"] = keep_f32
        report["rms_reduction"] = "f64-sum-f32-squares" if frontend_rms else "original-sequential-f32"
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
        gc.collect()
        build = scratch / "qwen35-native-build"
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF",
                        "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-delta",
                        "test-qwen35-linear", "test-qwen35-attention", "test-qwen35-decoder", "test-qwen35-vision", "-j", "4"], check=True)
        native = subprocess.run([str(build / "test-qwen35-delta"), str(out / "official-delta.gguf")],
                                capture_output=True, text=True)
        (out / "native-delta.log").write_text(native.stdout + native.stderr)
        print(native.stdout + native.stderr)
        native.check_returncode()
        report["native_delta"] = "passed official recurrence output/state comparisons and controls"
        weighted = subprocess.run([str(build / "test-qwen35-linear"), str(export), str(out / "official-delta.gguf")],
                                  capture_output=True, text=True)
        (out / "native-linear.log").write_text(weighted.stdout + weighted.stderr)
        print(weighted.stdout + weighted.stderr)
        weighted.check_returncode()
        report["native_linear"] = "passed all 18 checkpoint linear-attention blocks: projections/core/output and native carried states"
        attention = subprocess.run([str(build / "test-qwen35-attention"), str(export), str(out / "official-delta.gguf")],
                                   capture_output=True, text=True)
        (out / "native-attention.log").write_text(attention.stdout + attention.stderr)
        print(attention.stdout + attention.stderr)
        attention.check_returncode()
        report["native_attention"] = "passed six checkpoint full-attention blocks, native KV carry, fresh full-prefix and controls"
        def probe(label, weights, target, flags=()):
            result = subprocess.run([str(build / target), str(weights), str(out / "official-delta.gguf"), *flags],
                                    capture_output=True, text=True)
            (out / (label + ".log")).write_text(result.stdout + result.stderr)
            print(result.stdout + result.stderr)
            return result.returncode == 0
        report["native_vision"] = probe("native-vision", export, "test-qwen35-vision")
        report["native_decoder"] = probe("native-decoder", export, "test-qwen35-decoder")
        report["native_connected"] = probe("native-connected", export, "test-qwen35-decoder", ["--native-vision"])
        report["native_positions"] = probe("native-positions", export, "test-qwen35-decoder", ["--native-positions"])
        report["native_frontend"] = probe("native-frontend", export, "test-qwen35-decoder", ["--native-frontend"])
        if precision_audit:
            arms = []
            recipes = [
                ("f32", "f32", []),
                ("critical-mlp", "f16", [f"v.blocks.{i}.mlp.linear_fc{fc}.weight" for i in [5, 11] for fc in [1, 2]]),
                ("vision-f32", "f16", [record["name"] for record in manifest["records"]
                                        if record["name"].startswith("v.") and record["dtype"] == "float16"]),
            ]
            for label, dtype, retained in recipes:
                candidate = scratch / (label + "-experimental.gguf")
                candidate_manifest = converter.convert(local, candidate, converter.LIGHTON_MODEL,
                                                       converter.LIGHTON_REVISION, dtype, retained)
                (out / (label + "-manifest.json")).write_text(json.dumps(candidate_manifest, indent=2) + "\n")
                reader = gguf.GGUFReader(str(candidate))
                records = {record["name"]: record for record in candidate_manifest["records"]}
                if len(reader.tensors) != len(records):
                    raise ValueError("Precision arm roundtrip tensor count differs")
                for tensor in reader.tensors:
                    record = records[tensor.name]
                    if list(tensor.data.shape) != record["shape"] or hashlib.sha256(tensor.data.tobytes()).hexdigest() != record["sha256"]:
                        raise ValueError("Precision arm roundtrip mismatch: " + tensor.name)
                    if tensor.name in retained and tensor.data.dtype != np.float32:
                        raise ValueError("Retained tensor is not F32: " + tensor.name)
                del reader
                arms.append({"name": label, "dtype": dtype, "keep_f32": retained,
                             "bytes": candidate_manifest["bytes"], "roundtrip_tensors": len(records),
                             "vision": probe("native-vision-" + label, candidate, "test-qwen35-vision"),
                             "connected": probe("native-connected-" + label, candidate, "test-qwen35-decoder", ["--native-vision"])})
                candidate.unlink()
                candidate.with_suffix(".gguf.manifest.json").unlink()
            report["precision_arms"] = arms
            # Diagnostic run remains red when any strict numerical arm fails;
            # all other arm reports are retained to locate the first discrepancy.
            if not all(arm["vision"] and arm["connected"] for arm in arms):
                raise ValueError("At least one audited precision recipe fails strict native parity; inspect retained stage logs")
        elif math_audit:
            arms = []
            for label, norm, attention in [("norm", "1", "0"), ("attention", "0", "1"), ("both", "1", "1")]:
                os.environ["CRISPEMBED_QWEN35_VISION_PRECISE_NORM"] = norm
                os.environ["CRISPEMBED_QWEN35_VISION_PRECISE_ATTN"] = attention
                arms.append({"name": label, "norm": norm, "attention": attention,
                             "vision": probe("native-vision-math-" + label, export, "test-qwen35-vision"),
                             "connected": probe("native-connected-math-" + label, export, "test-qwen35-decoder", ["--native-vision"])})
            candidate = scratch / "f32-math-experimental.gguf"
            candidate_manifest = converter.convert(local, candidate, converter.LIGHTON_MODEL,
                                                   converter.LIGHTON_REVISION, "f32")
            arms.append({"name": "both-f32", "norm": "1", "attention": "1",
                         "bytes": candidate_manifest["bytes"],
                         "vision": probe("native-vision-math-both-f32", candidate, "test-qwen35-vision"),
                         "connected": probe("native-connected-math-both-f32", candidate, "test-qwen35-decoder", ["--native-vision"])})
            report["math_arms"] = arms
            if not all(arm["vision"] and arm["connected"] for arm in arms if arm["name"] in ["both", "both-f32"]):
                raise ValueError("Precise vision reductions fail the unchanged stage/connected parity gates")
        elif not all(report[key] for key in ["native_vision", "native_decoder", "native_connected", "native_positions", "native_frontend"]):
            raise ValueError("Native vision/decoder/connected parity failed; inspect retained stage logs")
        if not precision_audit and not math_audit:
            resized_out = out / "resized-frontend"
            resized_out.mkdir(exist_ok=True)
            resized_image = Image.new("RGB", (211, 301), "white")
            ImageDraw.Draw(resized_image).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
            resized_path = resized_out / "input.png"
            resized_image.save(resized_path)
            resized_reference = reference.dump_reference(local, resized_path, resized_out / "official-reference.npz")
            gc.collect()
            attention_result = subprocess.run([str(build / "test-qwen35-attention"), str(export),
                                               str(resized_out / "official-delta.gguf")], capture_output=True, text=True)
            (out / "native-resized-attention.log").write_text(attention_result.stdout + attention_result.stderr)
            print(attention_result.stdout + attention_result.stderr)
            report["native_resized_attention"] = attention_result.returncode == 0
            result = subprocess.run([str(build / "test-qwen35-decoder"), str(export),
                                     str(resized_out / "official-delta.gguf"), "--native-frontend"], capture_output=True, text=True)
            (out / "native-frontend-resized.log").write_text(result.stdout + result.stderr)
            print(result.stdout + result.stderr)
            report["native_frontend_resized"] = result.returncode == 0
            report["resized_reference"] = {key: resized_reference[key] for key in ["decoded", "generated_ids", "image_size", "resolved_image_processor"]}
            result.check_returncode()
            attention_result.check_returncode()
            report["native_parity"] = "native raw RGB/BPE/vision/positions/decoder/text checked on two owned fixtures; production integration/corpus validation pending"
        report["result"] = "export/reference/native model checks passed; production OCR support pending"
    except BaseException as error:
        report["result"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "summary.md").write_text("# Qwen3.5 OCR bootstrap\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
