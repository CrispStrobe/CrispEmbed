"""Hosted fourteen-arm OvisOCR3 HF/F32 versus native F32/F16 replay."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


profile = module("ovisocr3_hf_model_profile", ROOT / "models/qwen35_ovis3_hf_profile.py")
source = profile.source
reference = module("ovisocr3_native_reference_gate", ROOT / "tools/ci-heavy/ovisocr3_reference.py")
TARGETS = ("test-qwen35-decoder", "test-qwen35-vision", "test-qwen35-delta", "test-ovisocr3-context")
NATIVE_ENV = {name: "1" for name in ("CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR3_FRONTEND",
                                      "CRISPEMBED_OVISOCR3_INFERENCE", "CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS",
                                      "CRISPEMBED_QWEN35_VISION_PRECISE_NORM",
                                      "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN")}
NATIVE_ENV.update({name: "0" for name in (
    "CRISPEMBED_OVISOCR2_FRONTEND", "CRISPEMBED_OVISOCR2_INFERENCE",
    "CRISPEMBED_OVISOCR2_LARGE_IMAGES",
    "CRISPEMBED_QWEN35_PRECISE_RMS", "CRISPEMBED_QWEN35_VISION_CASCADE_NORM",
    "CRISPEMBED_QWEN35_IMATRIX", "CRISPEMBED_QWEN35_PARALLEL_F16",
    "CRISPEMBED_QWEN35_PROFILE", "CRISPEMBED_QWEN35_QUANT",
    "CRISPEMBED_QWEN35_TOKENIZER_MARKS")})
SELECTED_VISION_PRESET = {
    "name": "both",
    "enabled_gates": ["CRISPEMBED_QWEN35_VISION_PRECISE_NORM",
                      "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN"],
    "all_other_native_gates_unchanged": True,
    "no_production_default_promoted": True,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_hf_stop_probe(out):
    """Exercise the installed pinned HF criterion on both EOS IDs before weights."""
    import inspect
    import torch
    import transformers
    from transformers.generation.stopping_criteria import EosTokenCriteria

    criterion_source = Path(inspect.getfile(EosTokenCriteria))
    proof = {
        "transformers_version": transformers.__version__,
        "class": EosTokenCriteria.__module__ + "." + EosTokenCriteria.__name__,
        "source_sha256": sha(criterion_source),
        "eos_ids": source.SOURCE_EOS_IDS,
        "input_last_ids": [248044, 248046, 42],
        "expected_stops": [True, True, False],
    }
    ids = torch.tensor([[7, value] for value in proof["input_last_ids"]], dtype=torch.long)
    actual = EosTokenCriteria(eos_token_id=source.SOURCE_EOS_IDS)(ids, scores=None)
    proof["actual_shape"] = list(actual.shape)
    proof["actual_dtype"] = str(actual.dtype)
    proof["actual_stops"] = actual.tolist()
    path = out / "ovisocr3-hf-stop-runtime.json"
    path.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    if (proof["transformers_version"] != "5.16.1"
            or proof["source_sha256"] != profile.STOPPING_CRITERIA_SOURCE_SHA256
            or proof["actual_shape"] != [3] or proof["actual_dtype"] != "torch.bool"
            or proof["actual_stops"] != proof["expected_stops"]):
        raise ValueError("Installed HF dual EOS criterion runtime differs")
    return path, proof


def service_and_review(call=subprocess.check_output):
    metadata, receipt = profile._reviewed_receipts()
    source_run = profile.REFERENCE_RUN
    result = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{source_run}"]))
    artifact = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{profile.REFERENCE_ARTIFACT_ID}"]))
    small = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{receipt['artifact_id']}"]))
    if (result.get("status") != "completed" or result.get("conclusion") != "success"
            or result.get("head_sha") != profile.REFERENCE_HEAD
            or result.get("path") not in (".github/workflows/ovisocr3-reference.yml",
                                          ".github/workflows/ovisocr2-metadata.yml")
            or artifact.get("workflow_run", {}).get("id") != source_run
            or artifact.get("name") != f"ovisocr3-reference-{source_run}"
            or artifact.get("digest") != "sha256:" + profile.REFERENCE_ARTIFACT_SHA256
            or artifact.get("size_in_bytes") != profile.REFERENCE_ARTIFACT_BYTES
            or artifact.get("expired") is not False
            or small.get("workflow_run", {}).get("id") != source_run
            or small.get("id") != receipt["artifact_id"]
            or small.get("name") != f"ovisocr3-reference-reports-{source_run}"
            or small.get("digest") != "sha256:" + receipt["artifact_zip_sha256"]
            or small.get("size_in_bytes") != receipt["artifact_bytes"]
            or small.get("expired") is not False):
        raise ValueError("Reviewed OvisOCR3 original reference service run/artifact differs")
    meta_producer, meta_service = reference.metadata_service(call)
    if source._canonical_sha256(meta_producer) != profile.METADATA_PRODUCER_SHA256:
        raise ValueError("OvisOCR3 source metadata/replay proof differs")
    return metadata, receipt, {"reference_run": source_run, "reference_head": profile.REFERENCE_HEAD,
                               "reference_artifact_id": profile.REFERENCE_ARTIFACT_ID,
                               "reference_artifact_zip_sha256": profile.REFERENCE_ARTIFACT_SHA256,
                               "reference_artifact_bytes": profile.REFERENCE_ARTIFACT_BYTES,
                               "report_artifact_id": receipt["artifact_id"],
                               "report_artifact_zip_sha256": receipt["artifact_zip_sha256"],
                               "metadata_service": meta_service}


def checked_reference_archive(archive, receipt, folder):
    if sha(archive) != profile.REFERENCE_ARTIFACT_SHA256:
        raise ValueError("OvisOCR3 original reference archive SHA differs")
    expected = {"ovisocr3-original.npz": receipt["producer"]["reference_artifacts"]["npz"]["sha256"],
                "ovisocr3-original.json": receipt["producer"]["reference_artifacts"]["json"]["sha256"],
                "official-delta.gguf": receipt["producer"]["reference_artifacts"]["gguf"]["sha256"],
                "owned-readable.png": receipt["producer"]["owned_readable"]["file_sha256"]}
    folder.mkdir(parents=True, exist_ok=True)
    output = {}
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = {Path(item.filename).name for item in members}
        if (len(members) != 4 or names != set(expected)
                or any(item.is_dir() or item.flag_bits & 1 or ".." in Path(item.filename).parts
                       or stat.S_ISLNK(item.external_attr >> 16)
                       or item.file_size > 2_500_000_000 for item in members)):
            raise ValueError("OvisOCR3 reference artifact member/path contract differs")
        for item in members:
            name = Path(item.filename).name
            digest = hashlib.sha256()
            target = folder / name if name != "ovisocr3-original.npz" else None
            with zipped.open(item) as stream:
                if target is None:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                else:
                    with target.open("xb") as output_stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                            output_stream.write(block)
            if digest.hexdigest() != expected[name]:
                raise ValueError("OvisOCR3 reference member SHA differs: " + name)
            if target is not None:
                output[name] = target
    original = json.loads(output["ovisocr3-original.json"].read_text())
    controls = original.get("observational_control")
    if (len(original.get("stages", {})) != receipt["producer"]["stage_count"]
            or original.get("checkpoint_sha256") != source.CHECKPOINT_SHA256
            or original.get("image_sha256") != receipt["producer"]["owned_readable"]["file_sha256"]
            or original.get("missing_stages")
            or controls != {"prefill_logits_exact": True, "greedy_ids_exact": True}
            or original.get("cached_vs_full_prefix_same_argmax") is not True
            or original.get("generated_ids") != receipt["producer"]["reference_summary"]["generated_ids"]):
        raise ValueError("OvisOCR3 original reference stage/output proof differs")
    return original, output


def checked_readable(image, receipt):
    """Bind the actual decoded RGB bytes to the original owned PNG declaration."""
    from PIL import Image
    expected = receipt["producer"]["owned_readable"]
    with Image.open(image) as encoded:
        if encoded.format != "PNG" or encoded.mode != "RGB" or encoded.size != (448, 448):
            raise ValueError("OvisOCR3 reviewed readable image format/size differs")
        rgb_sha = hashlib.sha256(encoded.tobytes()).hexdigest()
    png_sha = sha(image)
    if (png_sha != expected["file_sha256"] or rgb_sha != expected["rgb_sha256"]
            or expected["size"] != [448, 448] or expected["gold"] != "Hello OCR\n12345"):
        raise ValueError("OvisOCR3 reviewed readable image RGB/source differs")
    return {"png_sha256": png_sha, "rgb_sha256": rgb_sha, "size": [448, 448],
            "gold": expected["gold"], "scope": expected["scope"]}


def validate_manifest(manifest, precision, schema):
    records = manifest.get("records", [])
    expected = set(schema) - {"lm_head.weight"}
    if (manifest.get("profile", {}).get("name") != "ovisocr3"
            or manifest.get("profile", {}).get("qualification_kind") != "hf5.16.1-original-reference"
            or manifest.get("source_model") != source.MODEL or manifest.get("source_revision") != source.REVISION
            or manifest.get("dtype") != precision or manifest.get("keep_f32") != []
            or manifest.get("checkpoint_tensor_count") != 473 or manifest.get("export_tensor_count") != 473
            or len(records) != 473 or {row.get("source") for row in records} != expected
            or len({row.get("name") for row in records}) != 473):
        raise ValueError("OvisOCR3 F32/F16 manifest identity/tensor coverage differs")
    for record in records:
        source_name = record["source"]
        name = ("l." + source_name.removeprefix("model.language_model.")
                if source_name.startswith("model.language_model.") else
                "v." + source_name.removeprefix("model.visual.")
                if source_name.startswith("model.visual.") else None)
        if name is None:
            raise ValueError("OvisOCR3 unknown source tensor prefix")
        shape = list(schema[source_name])
        if name == "v.patch_embed.proj.weight":
            shape = [shape[0], math.prod(shape[1:])]
        wanted = "float16" if precision == "f16" and len(shape) == 2 and "norm" not in name else "float32"
        if (record["name"] != name or record["source_shape"] != schema[source_name]
                or record["shape"] != shape or record["dtype"] != wanted
                or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise ValueError("OvisOCR3 original schema/precision record differs: " + source_name)


def roundtrip(model, manifest, checkpoint, precision, schema):
    validate_manifest(manifest, precision, schema)
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType
    from safetensors import safe_open
    reader = GGUFReader(model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    records = manifest["records"]
    if len(reader.tensors) != 473 or len(tensors) != 473 or set(tensors) != {row["name"] for row in records}:
        raise ValueError("OvisOCR3 GGUF tensor coverage/duplicates differ")
    fields = reader.fields
    if (fields["qwen35_ocr.format_version"].contents() != 3
            or fields["qwen35_ocr.profile"].contents() != "ovisocr3"
            or fields["qwen35_ocr.missing_source_files"].contents() != ["processor_config.json"]
            or fields["qwen35_ocr.metadata_qualification"].contents() != profile.METADATA_PRODUCER_SHA256
            or fields["qwen35_ocr.original_reference_qualification"].contents() != profile.REFERENCE_PRODUCER_SHA256
            or fields["qwen35_ocr.source_eos_token_ids"].contents() != [248044, 248046]
            or fields["qwen35_ocr.stop_token_ids"].contents() != [248044, 248046]
            or fields["qwen35_ocr.hf_stop_policy_qualified"].contents() != True
            or fields["qwen35_ocr.hf_stop_runtime_proof_sha256"].contents() != manifest["profile"]["hf_stop_runtime_proof_sha256"]
            or fields["qwen35_ocr.effective_stops_qualified"].contents() is not False
            or fields["qwen35_ocr.native_alias_qualified"].contents() is not False):
        raise ValueError("OvisOCR3 HF-only GGUF metadata differs")
    checked = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {row["source"] for row in records}:
            raise ValueError("OvisOCR3 checkpoint source tensor set differs")
        for row in records:
            tensor = tensors[row["name"]]
            original = handle.get_tensor(row["source"]).float().numpy()
            if list(original.shape) != row["source_shape"]:
                raise ValueError("OvisOCR3 source tensor shape differs")
            expected = np.ascontiguousarray(original.reshape(row["shape"]), dtype=np.dtype(row["dtype"]))
            digest = hashlib.sha256(expected.tobytes()).hexdigest()
            actual = hashlib.sha256(tensor.data.tobytes()).hexdigest()
            kind = GGMLQuantizationType.F32 if row["dtype"] == "float32" else GGMLQuantizationType.F16
            if (digest != row["sha256"] or actual != digest or tensor.tensor_type != kind
                    or list(reversed(tensor.shape.tolist())) != row["shape"]):
                raise ValueError("OvisOCR3 exact original-cast GGUF tensor differs: " + row["source"])
            checked.append({"name": row["name"], "sha256": actual})
    return {"tensor_count": len(checked), "all_original_cast_payloads_exact": True,
            "records": checked, "model_sha256": sha(model), "bytes": Path(model).stat().st_size}


def arms(precision, build, model, image, reference_gguf):
    result = {
        precision + "-delta": [build / "test-qwen35-delta", reference_gguf],
        precision + "-vision": [build / "test-qwen35-vision", model, reference_gguf],
        precision + "-decoder-prepared": [build / "test-qwen35-decoder", model, reference_gguf],
        precision + "-decoder-native-vision": [build / "test-qwen35-decoder", model, reference_gguf, "--native-vision"],
        precision + "-decoder-native-positions": [build / "test-qwen35-decoder", model, reference_gguf, "--native-positions"],
        precision + "-decoder-native-frontend": [build / "test-qwen35-decoder", model, reference_gguf, "--native-frontend"],
        precision + "-context": [build / "test-ovisocr3-context", model, image, reference_gguf],
    }
    if precision == "f32":
        result[precision + "-context"].append("--one-context")
    return result


def command(label, args, out, report, save, *, env=None, timeout=3600):
    path = out / (label + ".log")
    with path.open("w") as stream:
        try:
            completed = subprocess.run([str(value) for value in args], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write("Driver command error: " + type(error).__name__ + "\n")
            code = -1
    lines = path.read_text(errors="replace").splitlines()
    metrics = [line for line in lines if re.search(
        r"\b(?:PASS|FAIL|TRACE|BUDGET|DECODED|IDS|UTF8_HEX|cos_min|cache|argmax)\b"
        r"|^Native generated IDs:|^Native decoded UTF8 hex:|^Native/reference mean token confidence", line)]
    failures = [line for line in lines if re.search(r"\b(?:FAIL|ERROR|Traceback|Error:)\b", line)]
    entry = {"exit_code": code, "log_sha256": sha(path), "log_bytes": path.stat().st_size,
             "metric_lines": metrics, "first_failure": failures[0] if failures else None,
             "passed": code == 0 and path.stat().st_size > 0 and not failures}
    report.setdefault("commands", {})[label] = entry
    save()
    return entry


def raw_diff_manifest(folder, stages, metric_lines, label, generated_count):
    """Require an exact raw vector for every actual comparison, including TRACE."""
    log = "\n".join(metric_lines)
    compared = re.findall(r"^(?:PASS|FAIL) ([A-Za-z0-9_.]+): cos_min=", log, re.M)
    traced = re.findall(r"^TRACE ([A-Za-z0-9_.]+) max_abs=", log, re.M)
    if not compared or len(traced) != len(set(traced)):
        raise ValueError("OvisOCR3 actual comparison/TRACE names absent or repeated")
    if len(compared) != len(set(compared)) and not label.endswith("-delta"):
        raise ValueError("OvisOCR3 actual comparison stage repeated")
    names = set(compared) | set(traced)
    if label.endswith("-vision"):
        marker = "PASS VISION_RAW_DUMP_INTEGRITY stages=2"
        if len(names) != 37 or metric_lines.count(marker) != 1:
            raise ValueError("OvisOCR3 vision stages/raw-integrity marker differ")
    elif label.endswith("-delta"):
        marker = "PASS DELTA_RAW_DUMP_INTEGRITY stages=4 variants=linear-core,reset"
        if metric_lines.count(marker) != 1 or not {"prefill.delta.output", "cached.delta.output"} <= names:
            raise ValueError("OvisOCR3 delta stages/raw-integrity marker differ")
    else:
        marker = None
        required_layers = {f"{phase}.decoder.layer.{layer}" for phase in ("prefill", "cached", "full_prefix")
                           for layer in range(24)}
        required_cache = {name for name in stages
                          if re.fullmatch(r"(prefill|cached)\.cache\.\d+\.(keys|values|conv_states|recurrent_states)", name)}
        required_scores = {f"generated.scores.{step}" for step in range(generated_count)}
        if (not required_layers <= names or not required_cache <= names or not required_scores <= names
                or (label.endswith("-decoder-native-frontend") and not traced)):
            raise ValueError("OvisOCR3 all-layer/cache/all-token-score/TRACE coverage differs")
    files = {path.name.removesuffix(".f32"): path for path in folder.glob("*.f32")}
    if set(files) != names:
        raise ValueError("OvisOCR3 raw primary file set differs from actual comparisons")
    variants = {}
    if label.endswith("-delta"):
        expected_variants = {"linear-core": {"prefill.delta.final_state", "cached.delta.final_state"},
                             "reset": {"prefill.delta.output", "prefill.delta.final_state"}}
        for variant, expected in expected_variants.items():
            found = {path.name.removesuffix(".f32"): path for path in (folder / variant).glob("*.f32")}
            if set(found) != expected:
                raise ValueError("OvisOCR3 delta raw positive-variant file set differs")
            variants[variant] = found
        exact = {path.name for path in files.values()} | {
            variant + "/" + stage + ".f32" for variant, expected in expected_variants.items() for stage in expected}
    else:
        exact = {path.name for path in files.values()}
    if {path.relative_to(folder).as_posix() for path in folder.rglob("*.f32")} != exact:
        raise ValueError("OvisOCR3 raw nested file set differs")
    records = []
    for variant, stage_files in [("primary", files), *sorted(variants.items())]:
        for stage, path in sorted(stage_files.items()):
            shape = stages.get(stage)
            if (not isinstance(shape, list) or not shape
                    or any(type(value) is not int or value < 1 for value in shape)):
                raise ValueError("OvisOCR3 original source shape absent: " + stage)
            expected_bytes = math.prod(shape) * 4
            if path.stat().st_size != expected_bytes:
                raise ValueError("OvisOCR3 raw F32 byte count differs: " + stage)
            records.append({"stage": stage, "variant": variant, "original_stage": stage,
                            "file": "native-diff/" + label + "/" +
                                    (variant + "/" if variant != "primary" else "") + path.name,
                            "bytes": expected_bytes, "sha256": sha(path), "source_shape": shape})
    return {"arm": label, "metric_stage_count": len(names), "trace_stage_count": len(traced),
            "raw_vector_count": len(records), "variant_stage_count": len(records) - len(names),
            "dump_integrity_marker": marker, "coverage_complete": True, "files": records}


def main(preflight=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-native14"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "phase": "reviewed-reference-preflight", "source_commit": os.environ.get("GITHUB_SHA"),
              "scope": "same OvisOCR3 checkpoint original HF F32 modules versus experimental native F32/F16; no public alias",
              "failures": [], "commands": {}, "exports": {}, "native_environment": NATIVE_ENV,
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "tools/ci-heavy/ovisocr3_native14.py", "models/convert-qwen35-ocr-to-gguf.py",
                  "models/qwen35_ovis3_profile.py", "models/qwen35_ovis3_hf_profile.py",
                  "tools/dump_ovisocr3_reference.py")},
              "native_source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                  [*sorted((ROOT / "src").glob("qwen35*")),
                   *(ROOT / name for name in ("src/crispembed_diff.h", "tests/qwen35_parity.h",
                                              "tests/test_qwen35_decoder.cpp", "tests/test_qwen35_vision.cpp",
                                              "tests/test_qwen35_delta.cpp", "tests/test_ovisocr3_context.cpp"))]
                  if path.is_file() and path.suffix in (".cpp", ".h")}}
    report_path = out / "ovisocr3-native14.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        metadata, receipt, service = service_and_review()
        report["reference_service"] = service
        report["metadata_producer_canonical_sha256"] = profile.METADATA_PRODUCER_SHA256
        report["original_reference_producer_canonical_sha256"] = profile.REFERENCE_PRODUCER_SHA256
        report["phase"] = "preflight-complete"; save()
        if preflight:
            report["status"] = "preflight passed; no framework/model loaded"; save()
            return report
        import importlib.metadata
        from huggingface_hub import snapshot_download, hf_hub_download
        import torch
        import transformers
        import tokenizers
        if (torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "5.16.1"
                or tokenizers.__version__ != "0.23.3" or importlib.metadata.version("gguf") != "0.19.0"):
            raise ValueError("OvisOCR3 native14 pinned framework stack differs")
        torch.set_num_threads(2)
        report["phase"] = "installed-HF-dual-EOS-runtime-probe"; save()
        stop_proof_path, stop_proof = installed_hf_stop_probe(out)
        os.environ["CRISPEMBED_OVISOCR3_HF_STOP_PROOF"] = str(stop_proof_path)
        report["hf_stop_runtime"] = {"proof_sha256": sha(stop_proof_path), **stop_proof,
                                     "scope": "installed HF5.16.1 EosTokenCriteria only; vLLM unobserved"}
        save()
        report["phase"] = "frozen-original-reference-artifact"; save()
        archive = scratch / "original-reference.zip"
        with archive.open("wb") as stream, (out / "reference-download.log").open("wb") as log:
            result = subprocess.run(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{profile.REFERENCE_ARTIFACT_ID}/zip"],
                                    stdout=stream, stderr=log, timeout=1800, check=False)
        report["reference_download"] = {"exit_code": result.returncode, "bytes": archive.stat().st_size,
                                        "zip_sha256": sha(archive), "log_sha256": sha(out / "reference-download.log")}
        save()
        if result.returncode or report["reference_download"]["zip_sha256"] != profile.REFERENCE_ARTIFACT_SHA256:
            raise ValueError("Reviewed OvisOCR3 original reference download differs")
        original, files = checked_reference_archive(archive, receipt, scratch / "reference")
        archive.unlink()
        report["owned_readable"] = checked_readable(files["owned-readable.png"], receipt)
        report["selected_vision_preset"] = SELECTED_VISION_PRESET
        (out / "ovisocr3-original.json").write_bytes(files["ovisocr3-original.json"].read_bytes())
        report["reference_artifacts"] = {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                         for name, path in files.items()}
        report["reference_original_json_sha256"] = sha(files["ovisocr3-original.json"])
        report["original_capture"] = {key: original.get(key) for key in
                                      ("generated_ids", "decoded", "raw_decoded", "generation_result",
                                       "cached_vs_full_prefix_max_abs", "cached_vs_full_prefix_same_argmax",
                                       "observational_control")}
        save()
        report["phase"] = "source-and-checkpoint"; save()
        directory = Path(snapshot_download(source.MODEL, revision=source.REVISION,
                                           allow_patterns=list(source.metadata_files())))
        qualified = profile.load_profile(directory, source.MODEL, source.REVISION)
        if {name: sha(directory / name) for name in source.metadata_files()} != metadata["producer"]["source_files_sha256"]:
            raise ValueError("OvisOCR3 native14 source metadata differs")
        checkpoint = Path(hf_hub_download(source.MODEL, "model.safetensors", revision=source.REVISION))
        report["checkpoint_sha256"] = profile.verify_checkpoint(checkpoint, qualified)
        if checkpoint.parent.resolve() != directory.resolve():
            raise ValueError("OvisOCR3 native14 checkpoint/source snapshot differs")
        report["phase"] = "portable-native-build"; save()
        build = scratch / "build"
        for label, args in (("cmake-configure", ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                                               "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF",
                                               "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"]),
                            ("cpu-baseline-check", [sys.executable, ROOT / "scripts/check-cpu-baseline.py", build]),
                            ("native-target-build", ["cmake", "--build", build, "--target", *TARGETS, "-j2"])):
            if not command(label, args, out, report, save, timeout=1800)["passed"]:
                raise ValueError("OvisOCR3 native14 portable build gate failed: " + label)
        report["native_binary_sha256"] = {name: sha(build / name) for name in TARGETS}
        save()
        reference_gguf = files["official-delta.gguf"]
        image = files["owned-readable.png"]
        schema = metadata["producer"]["schema"]["official_state_dict"]
        for precision in ("f32", "f16"):
            report["phase"] = precision + "-qualified-export-and-seven-arms"; save()
            model = scratch / ("ovisocr3-" + precision + ".gguf")
            conversion = command(precision + "-convert", [sys.executable, ROOT / "models/convert-qwen35-ocr-to-gguf.py",
                                                          "--model", source.MODEL, "--revision", source.REVISION,
                                                          "--profile", "ovisocr3-hf-reference", "--dtype", precision,
                                                          "--output", model], out, report, save, timeout=3600)
            if conversion["passed"] and model.is_file():
                manifest_path = model.with_suffix(".gguf.manifest.json")
                try:
                    manifest = json.loads(manifest_path.read_text())
                    result = roundtrip(model, manifest, checkpoint, precision, schema)
                    report["exports"][precision] = {"manifest_sha256": sha(manifest_path), **result}
                    (out / manifest_path.name).write_bytes(manifest_path.read_bytes())
                except (OSError, ValueError, KeyError) as error:
                    report["failures"].append(precision + "-export: " + str(error)[:300])
            else:
                report["failures"].append(precision + "-conversion")
            save()
            if precision not in report["exports"]:
                continue
            for label, args in arms(precision, build, model, image, reference_gguf).items():
                env = dict(os.environ, **NATIVE_ENV)
                env.pop("CRISPEMBED_DIFF_COS_THRESHOLD", None)
                dump = None
                if not label.endswith("-context"):
                    dump = scratch / "native-diff" / label
                    dump.mkdir(parents=True, exist_ok=True)
                    env["CRISPEMBED_DIFF_DUMP_DIR"] = str(dump)
                else:
                    env.pop("CRISPEMBED_DIFF_DUMP_DIR", None)
                result = command(label, args, out, report, save, env=env, timeout=2400)
                metrics = "\n".join(result["metric_lines"])
                if dump is not None:
                    try:
                        manifest = raw_diff_manifest(dump, original["stages"], result["metric_lines"], label,
                                                     len(original["generated_ids"][0]))
                        path = out / (label + "-raw-manifest.json")
                        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                        result["raw_diff_manifest_sha256"] = sha(path)
                        result["raw_diff_coverage_complete"] = True
                    except (OSError, ValueError, KeyError) as error:
                        result["raw_diff_coverage_complete"] = False
                        result["integrity_error"] = type(error).__name__ + ": " + str(error)
                        result["passed"] = False
                if "-decoder-" in label:
                    layers = re.findall(r"^(?:PASS|FAIL) (?:prefill|cached|full_prefix)\.decoder\.layer\.\d+:", metrics, re.M)
                    result["comparison_coverage"] = len(layers) == 72 and result.get("raw_diff_coverage_complete") is True
                    match = re.search(r"^Native generated IDs:((?: \d+)+)$", metrics, re.M)
                    result["generated_ids"] = [int(value) for value in match[1].split()] if match else None
                    if result["generated_ids"] != original["generated_ids"][0]:
                        result["comparison_coverage"] = False
                    if label.endswith("-decoder-native-frontend"):
                        decoded = re.search(r"^Native decoded UTF8 hex:([0-9a-f]*)$", metrics, re.M)
                        result["decoded_utf8_hex"] = decoded[1] if decoded else None
                        expected_decoded = original["decoded"][0].encode().hex()
                        if result["decoded_utf8_hex"] != expected_decoded:
                            result["comparison_coverage"] = False
                elif label.endswith("-context"):
                    required = {"raw", "repeat", "gate-recovery"}
                    if precision == "f16":
                        required |= {"independent-context", "survivor-after-free"}
                    result["comparison_coverage"] = all(
                        re.search(r"^PASS OVIS3_CONTEXT " + re.escape(part) + r"\b", metrics, re.M)
                        for part in required)
                elif label.endswith("-delta"):
                    result["comparison_coverage"] = result.get("raw_diff_coverage_complete") is True
                else:
                    result["comparison_coverage"] = result.get("raw_diff_coverage_complete") is True
                result["passed"] &= result["comparison_coverage"]
                if not result["passed"]:
                    report["failures"].append(label)
                save()
            if precision == "f32" and len([label for label in report["commands"] if label.startswith("f32-") and label != "f32-convert"]) == 7:
                model.unlink()
                report["f32_cleanup"] = {"verified_model_sha256": report["exports"]["f32"]["model_sha256"],
                                         "removed_task_owned_after_seven_arms": True,
                                         "source_checkpoint_sha256": source.CHECKPOINT_SHA256}
                save()
        report["arm_count"] = len([label for label in report["commands"] if re.fullmatch(r"f(?:32|16)-(?:delta|vision|decoder-.*|context)", label)])
        report["phase"] = "complete"
        report["status"] = "completed diagnostic OvisOCR3 fourteen-arm native replay"
        report["native14_positive"] = report["arm_count"] == 14 and not report["failures"] and set(report["exports"]) == {"f32", "f16"}
        save()
        if not report["native14_positive"]:
            raise ValueError("OvisOCR3 fourteen-arm native parity incomplete; diagnostic artifacts retained")
        return report
    except BaseException as error:
        report["status"] = "failed OvisOCR3 native14 execution"
        report["error_type"] = type(error).__name__
        report["error_message"] = reference.failure_detail(error)
        save()
        raise


if __name__ == "__main__":
    main("--preflight" in sys.argv)
