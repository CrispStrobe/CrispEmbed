"""Hosted Xiaomi-OCR-0 metadata or bounded original/native F32/F16 experiment.

No execution at import. Metadata runs first and must succeed before any weights.
The frozen two-image protocol is an integration experiment, never a benchmark.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "tests/fixtures/xiaomi0-bounded/protocol.json"
PROTOCOL_SHA256 = "4fa314d26c4c2372f929fc7ada8b8aa0188bbc9148ec82a50b509d9eabb56c30"
CPU_REQUIREMENTS = ("torch==2.14.0", "torchvision==0.29.0")
PYTHON_REQUIREMENTS = ("transformers==5.17.0", "tokenizers==0.23.3", "numpy==2.2.6",
                       "pillow==12.1.0", "huggingface-hub==1.33.0", "gguf==0.19.0",
                       "safetensors==0.8.0", "jinja2==3.1.6")
WORKER_SECONDS = 7200
STAGES = (["vision.embedded"] + [f"vision.layer.{i}" for i in range(12)]
          + ["vision.merger", "decoder.input"] + [f"decoder.layer.{i}" for i in range(24)]
          + ["decoder.norm"])
ACTIVE_CHILDREN = []


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


policy = module("xiaomi0_hosted_policy", ROOT / "models/qwen35_xiaomi0_profile.py")
converter = module("xiaomi0_hosted_export", ROOT / "models/convert-qwen35-ocr-to-gguf.py")


def preflight():
    if policy.file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Frozen Xiaomi integration protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["dtype_arms"] != ["f32", "f16"] or len(protocol["cases"]) != 2:
        raise ValueError("Xiaomi initial execution budget differs")
    for case in protocol["cases"]:
        if policy.file_sha256(ROOT / case["image"]) != case["image_sha256"] or case["max_new_tokens"] != 16:
            raise ValueError("Xiaomi frozen image/cap differs")
    policy.evidence()
    return protocol


def native_env():
    # Do not inherit old diagnostic/precision gates into a new source experiment.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_QWEN35_", "CRISPEMBED_OVISOCR", "CRISPEMBED_XIAOMI_"))}
    env.update({"CRISPEMBED_QWEN35_OCR": "1", "CRISPEMBED_XIAOMI_OCR0_FRONTEND": "1",
                "CRISPEMBED_XIAOMI_OCR0_INFERENCE": "1", "CRISPEMBED_BUILD_MODE": "portable"})
    return env


def run(command, path, *, env=None, timeout=1200):
    """Bound subprocess groups and retain their actual output on failure."""
    command = list(map(str, command))
    with path.open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ACTIVE_CHILDREN.append(child)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise ValueError("Bounded Xiaomi command timed out: " + path.name)
        finally:
            ACTIVE_CHILDREN.remove(child)
    return subprocess.CompletedProcess(command, code, stdout=path.read_text(errors="replace"))


def directories(output=None, scratch=None):
    if output is None:
        if not os.environ.get("HEAVY_OUT"):
            raise ValueError("Provide --output or the heavy-cpu HEAVY_OUT directory")
        output = Path(os.environ["HEAVY_OUT"]) / "xiaomi0"
    if scratch is None:
        if not os.environ.get("HEAVY_SCRATCH"):
            raise ValueError("Provide --scratch or the heavy-cpu HEAVY_SCRATCH directory")
        scratch = Path(os.environ["HEAVY_SCRATCH"]) / "xiaomi0"
    output, scratch = Path(output).resolve(), Path(scratch).resolve()
    if output == scratch or output in scratch.parents or scratch in output.parents:
        raise ValueError("Uploaded output and model/build scratch must be disjoint")
    return output, scratch


def resource_snapshot(out, scratch):
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key] = int(value.split()[0]) * 1024
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "cpus": os.cpu_count(),
            "load": list(os.getloadavg()), "memory_bytes": memory,
            "scratch_free_bytes": shutil.disk_usage(scratch).free,
            "output_free_bytes": shutil.disk_usage(out).free,
            "same_filesystem": out.stat().st_dev == scratch.stat().st_dev}


def require_resources(snapshot, inference):
    required_memory = (9 if inference else 3) * 2**30
    required_disk = (12 if inference else 8) * 2**30
    required_output = 2 * 2**30
    if snapshot.get("same_filesystem") is not False:
        # Scratch and retained tensors often share the runner's root volume.
        # Independent checks would count the same free bytes twice.
        required_disk += required_output
        required_output = required_disk
    if (snapshot["memory_bytes"].get("MemAvailable", 0) < required_memory
            or snapshot["scratch_free_bytes"] < required_disk
            or snapshot["output_free_bytes"] < required_output):
        raise ValueError("Insufficient hosted Xiaomi RAM/scratch/output capacity")


def job_env(scratch):
    env = native_env()
    for key in ("PYTHONPATH", "PYTHONHOME", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
        env.pop(key, None)
    env.update({"HF_HOME": str(scratch / "hf"), "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    return env


def interrupted(signum, _frame):
    for child in tuple(ACTIVE_CHILDREN):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    raise ValueError("Hosted Xiaomi interrupted by signal " + str(signum))


def build_receipt(build, targets, revision):
    cache = (build / "CMakeCache.txt").read_text()
    for name in ("GGML_NATIVE", "CRISPEMBED_NATIVE"):
        if re.findall(r"^" + name + r":(?:BOOL|UNINITIALIZED)=(.*)$", cache, re.MULTILINE) != ["OFF"]:
            raise ValueError("Actual native build cache is not portable: " + name)
    return {"schema": "xiaomi0-native-build/v1", "runtime_revision": revision,
            "build_mode": "portable", "ggml_native": False, "crispembed_native": False,
            "binaries": {name: {"sha256": policy.file_sha256(build / name),
                                "bytes": (build / name).stat().st_size} for name in targets}}


def installed_versions():
    from importlib.metadata import version
    result = {}
    for requirement in CPU_REQUIREMENTS + PYTHON_REQUIREMENTS:
        name, expected = requirement.split("==")
        actual = version(name)
        if actual.split("+")[0] != expected:
            raise ValueError("Xiaomi installed dependency differs: " + name)
        result[name] = actual
    return result


def bootstrap(args, out, scratch):
    """A direct heavy-cpu dispatch installs pinned direct dependencies first."""
    if out.exists() or scratch.exists():
        raise ValueError("Fresh task-owned Xiaomi output and scratch directories required")
    out.mkdir(parents=True)
    scratch.mkdir(parents=True)
    state = {"schema": "xiaomi0-hosted-bootstrap/v1", "phase": "resources",
             "direct_dependency_pins": list(CPU_REQUIREMENTS + PYTHON_REQUIREMENTS),
             "inference_requested": args.inference, "worker_seconds": WORKER_SECONDS}

    def save():
        (out / "bootstrap.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    try:
        state["resources"] = resource_snapshot(out, scratch)
        save()
        require_resources(state["resources"], args.inference)
        state["phase"] = "isolated-dependencies"
        save()
        environment = scratch / "venv"
        venv.EnvBuilder(with_pip=True, clear=False).create(environment)
        python = environment / "bin/python"
        env = job_env(scratch)
        commands = [
            ([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
              "--index-url", "https://download.pytorch.org/whl/cpu", *CPU_REQUIREMENTS], "install-torch.log"),
            ([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
              *PYTHON_REQUIREMENTS], "install-python.log"),
            ([python, "-m", "pip", "check"], "pip-check.log"),
            ([python, "-m", "pip", "list", "--format=json"], "installed-packages.json"),
        ]
        for command, name in commands:
            if run(command, out / name, env=env, timeout=900 if name.startswith("install-") else 60).returncode:
                raise ValueError("Xiaomi dependency installation failed: " + name)
        state["phase"] = "worker"
        save()
        command = [python, Path(__file__).resolve(), "--prepared-environment",
                   "--output", out, "--scratch", scratch]
        if args.model_dir:
            command += ["--model-dir", args.model_dir.resolve()]
        if args.inference:
            command.append("--inference")
        result = run(command, out / "worker.log", env=env, timeout=WORKER_SECONDS)
        state["worker_returncode"] = result.returncode
        state["phase"] = "complete" if result.returncode == 0 else "failed"
        save()
        if result.returncode:
            raise ValueError("Xiaomi hosted worker failed; retained worker.log and result JSON")
    except Exception as error:
        state["failure"] = str(error)
        save()
        raise


def metadata(directory, out, scratch):
    import gguf
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    pinned = policy.load_profile(directory, policy.MODEL, policy.REVISION)
    runtime = policy.validate_runtime_metadata(directory)
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    del model
    schema_pin = policy.validate_main_schema(expected)
    (out / "original-meta-schema.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    writer = gguf.GGUFWriter(str(scratch / "frontend.gguf"), "qwen35_ocr")
    writer.add_string("general.source.huggingface.repository", policy.MODEL)
    writer.add_string("general.source.huggingface.revision", policy.REVISION)
    writer.add_uint32("qwen35_ocr.format_version", 4)
    converter.write_tokenizer_metadata(writer, converter.read_tokenizer_metadata(
        Path(directory), 248320, pinned["generation"]))
    converter.write_xiaomi0_metadata(writer, pinned)

    frontend_arrays = {}

    def save(name, values):
        value = np.asarray(values)
        if not value.size or not np.isfinite(value).all():
            raise ValueError("Empty/nonfinite frontend reference: " + name)
        frontend_arrays[name] = np.ascontiguousarray(value, dtype=np.float32)
        writer.add_tensor(name, frontend_arrays[name])

    def text(name, value):
        save(name, list(value.encode("utf-8")))

    height, width = 257, 359
    pixels = ((np.arange(height * width * 3, dtype=np.uint32) * 73 + 29) % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels)
    resized = []
    original = processor.image_processor.resize

    def capture_resize(*args, **kwargs):
        result = original(*args, **kwargs)
        resized.append(result.detach().cpu().numpy().copy())
        return result

    processor.image_processor.resize = capture_resize
    tasks = list(policy.evidence()["prompts"])
    writer.add_uint32("test.xiaomi_cases", len(tasks))
    try:
        for index, task in enumerate(tasks):
            prompt = pinned["prompts"][task]
            rendered = policy.rendered_prompt(task)
            inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
            label = f"case.{index}"
            writer.add_uint32(label + ".image_tokens", int((inputs["input_ids"] == 248056).sum()))
            text(label + ".prompt", prompt)
            text(label + ".rendered", rendered)
            save(label + ".ids", inputs["input_ids"].numpy())
            save(label + ".modalities", inputs["mm_token_type_ids"].numpy())
            if index == 0:
                if len(resized) != 1 or resized[0].dtype != np.uint8 or resized[0].shape[0] != 1:
                    raise ValueError("Xiaomi uint8 torchvision resize capture differs")
                save("image.rgb", pixels)
                save("image.resized", resized[0][0].transpose(1, 2, 0))
                save("image.patches", inputs["pixel_values"].numpy())
                save("image.grid", inputs["image_grid_thw"].numpy())
    finally:
        processor.image_processor.resize = original
    texts = ["Hello OCR 42", "e\u0301lan", "किरण ١٢३ 中文", "\u0301\u0308abc", "x^2+y^2=25", "<think>ok</think><|im_end|>"]
    writer.add_uint32("test.xiaomi_texts", len(texts))
    for index, value in enumerate(texts):
        ids = processor.tokenizer.encode(value, add_special_tokens=False)
        text(f"text.{index}.input", value)
        save(f"text.{index}.ids", ids)
        text(f"text.{index}.decoded", processor.tokenizer.decode(ids, skip_special_tokens=True))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    np.savez_compressed(out / "frontend-reference.npz", **frontend_arrays)
    return {"runtime": runtime, "meta_tensor_count": len(expected),
            "meta_schema_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
            "checkpoint_main_schema_sha256": schema_pin,
            "frontend_gguf_sha256": policy.file_sha256(scratch / "frontend.gguf"),
            "source_payload_verified": False, "model_forward_executed": False}


def original_reference(directory, case, output, model, processor):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(ROOT / case["image"]).convert("RGB")
    if hashlib.sha256(image.tobytes()).hexdigest() != case["rgb_sha256"]:
        raise ValueError("Frozen Xiaomi RGB changed")
    prompt = policy.evidence()["prompts"][case["task"]]
    rendered = policy.rendered_prompt(case["task"])
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    arrays, handles = {}, []

    def capture(name):
        def observer(_module, _args, result):
            if name in arrays:
                return  # Retain the first, actual prefill only.
            tensor = result[0] if isinstance(result, tuple) else result
            if not torch.is_tensor(tensor):
                raise ValueError("Unexpected original stage: " + name)
            arrays[name] = tensor.detach().float().cpu().numpy().copy()
        return observer

    def before(name):
        def observer(_module, args, kwargs):
            if name not in arrays:
                tensor = kwargs.get("hidden_states", kwargs.get("inputs_embeds", args[0] if args else None))
                if not torch.is_tensor(tensor):
                    raise ValueError("Missing original stage input: " + name)
                arrays[name] = tensor.detach().float().cpu().numpy().copy()
        return observer

    visual, language = model.model.visual, model.model.language_model
    handles.append(visual.blocks[0].register_forward_pre_hook(before("vision.embedded"), with_kwargs=True))
    for index, layer in enumerate(visual.blocks):
        handles.append(layer.register_forward_hook(capture(f"vision.layer.{index}")))
    handles.append(visual.merger.register_forward_hook(capture("vision.merger")))
    handles.append(language.register_forward_pre_hook(before("decoder.input"), with_kwargs=True))
    for index, layer in enumerate(language.layers):
        handles.append(layer.register_forward_hook(capture(f"decoder.layer.{index}")))
    handles.append(language.norm.register_forward_hook(capture("decoder.norm")))
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=case["max_new_tokens"], do_sample=False,
                                       return_dict_in_generate=True, output_scores=True, output_logits=True)
    finally:
        for handle in handles:
            handle.remove()
    ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    logits = np.stack([tensor[0].detach().float().cpu().numpy() for tensor in generated.logits])
    if logits.shape != (len(ids), 248320) or set(arrays) != set(STAGES) or not np.isfinite(logits).all():
        raise ValueError("Incomplete Xiaomi actual logits or prefill stages")
    arrays["generated.logits"] = logits
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite original stage: " + name)
    np.savez(output / (case["id"] + "-original.npz"), **arrays)
    record = {"ids": ids, "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
              "stop": "eos" if ids and ids[-1] == 248046 else "max_tokens",
              "cap": case["max_new_tokens"], "image_sha256": case["image_sha256"],
              "input_ids": inputs["input_ids"][0].tolist(), "grid": inputs["image_grid_thw"].tolist(),
              "stage_shapes": {name: list(value.shape) for name, value in arrays.items()}}
    if record["stop"] == "max_tokens" and len(ids) != record["cap"]:
        raise ValueError("Unexplained original generation termination")
    return record


def metrics(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite or shape-mismatched comparison")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    norms_a = np.linalg.norm(rows_a, axis=1)
    norms_b = np.linalg.norm(rows_b, axis=1)
    den = norms_a * norms_b
    cos = np.divide((rows_a * rows_b).sum(1), den, out=np.zeros_like(den), where=den > 0)
    cos[(norms_a == 0) & (norms_b == 0)] = 1
    maximum = float(np.max(np.abs(a - b)))
    minimum = float(cos.min())
    return {"cosine_min": minimum, "max_abs": maximum,
            "rms_actual": float(np.sqrt((a * a).mean())), "rms_original": float(np.sqrt((b * b).mean())),
            "strict_pass": minimum >= 0.99999 and maximum <= 0.002}


def compare_native(log, logits_path, original_path, reference, dtype):
    import numpy as np
    rows = [json.loads(line.removeprefix("XIAOMI0_RESULT ")) for line in log.splitlines() if line.startswith("XIAOMI0_RESULT ")]
    if len(rows) != 1:
        raise ValueError("Missing unique actual Xiaomi generation result")
    row = rows[0]
    stage_rows = [json.loads(line.removeprefix("XIAOMI0_STAGE ")) for line in log.splitlines() if line.startswith("XIAOMI0_STAGE ")]
    if len(stage_rows) != len(STAGES) or {x["name"] for x in stage_rows} != set(STAGES):
        raise ValueError("Incomplete native stage capture")
    with np.load(original_path) as arrays:
        if set(arrays) != set(STAGES) | {"generated.logits"}:
            raise ValueError("Original stage/logit coverage differs")
        raw = np.fromfile(str(logits_path) + ".stages", dtype=np.float32)
        stages = {}
        offset = 0
        for stage in stage_rows:
            if stage["byte_offset"] != offset * 4 or stage["count"] <= 0 or stage["name"] not in arrays:
                raise ValueError("Invalid native stage range/name")
            expected = arrays[stage["name"]]
            if expected.size != stage["count"]:
                raise ValueError("Native stage shape mismatch")
            actual = raw[offset:offset + stage["count"]].reshape(expected.shape)
            stages[stage["name"]] = metrics(actual, expected)
            offset += stage["count"]
        if offset != raw.size:
            raise ValueError("Native stage tail or truncation")
        native_logits = np.fromfile(logits_path, dtype=np.float32).reshape(-1, 248320)
        if native_logits.shape[0] != len(row["token_ids"]):
            raise ValueError("Native logits/IDs length differs")
        complete = native_logits.shape == arrays["generated.logits"].shape
        final = metrics(native_logits, arrays["generated.logits"]) if complete else None
    native_text = bytes.fromhex(row["raw_text_utf8_hex"]).decode("utf-8")
    ids_equal = row["token_ids"] == reference["ids"]
    stop_equal = row["stop_kind"] == (1 if reference["stop"] == "eos" else 2)
    worst = min(stages, key=lambda name: stages[name]["cosine_min"])
    return {"native": row, "ids_equal": ids_equal, "text_equal": native_text == reference["text"],
            "stop_equal": stop_equal, "prefixes_equal": ids_equal, "final_logits": final,
            "comparison_kind": "actual-free-generation; divergent prefixes are not teacher-forced",
            "logit_rows": native_logits.shape[0], "vocabulary": 248320,
            "owner_bar": 0.99 if dtype == "f32" else 0.98,
            "owner_final_logit_bar_pass": final is not None and final["cosine_min"] > (0.99 if dtype == "f32" else 0.98),
            "stage_metrics": stages, "worst_intermediate_stage": worst,
            "precision_qualified": False, "full_document_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Defaults to HEAVY_OUT/xiaomi0")
    parser.add_argument("--scratch", type=Path, help="Defaults to HEAVY_SCRATCH/xiaomi0; never uploaded")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inference", action="store_true", help="Run exactly two frozen original cases and four native arms on hosted CI")
    parser.add_argument("--prepared-environment", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = preflight()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("This recipe builds/captures only on hosted CI; lightweight source tests run locally")
    if os.environ.get("CRISPEMBED_BUILD_MODE", "portable") != "portable":
        parser.error("Xiaomi precision evidence requires the portable CPU build")
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        parser.error("The initial Xiaomi recipe requires hosted Linux x86_64 and Python 3.12")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out, scratch = directories(args.output, args.scratch)
    if not args.prepared_environment:
        bootstrap(args, out, scratch)
        return
    if Path(sys.prefix).resolve() != (scratch / "venv").resolve() or not (out / "bootstrap.json").is_file():
        parser.error("Prepared worker must use its task-owned isolated environment")
    started = time.monotonic()
    report = {"schema": "xiaomi0-hosted-result/v1", "phase": "metadata", "protocol_sha256": PROTOCOL_SHA256,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_audit_sha256": policy.AUDIT_SHA256, "build_mode": "portable", "original": {}, "native": {},
              "full_document_qualified": False, "publication_qualified": False}

    def save():
        (out / "xiaomi0-result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        report["installed_direct_dependencies"] = installed_versions()
        report["resources_before_metadata"] = resource_snapshot(out, scratch)
        require_resources(report["resources_before_metadata"], args.inference)
        save()
        from huggingface_hub import snapshot_download
        import torch
        import torchvision
        if torch.__version__.split("+")[0] != "2.14.0" or torchvision.__version__.split("+")[0] != "0.29.0":
            raise ValueError("Xiaomi original verified torch/torchvision versions differ")
        torch.set_num_threads(protocol["threads"])
        torch.manual_seed(0)
        directory = args.model_dir or Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                                             allow_patterns=list(policy.metadata_files())))
        report["metadata"] = metadata(directory, out, scratch)
        build = scratch / "build"
        env = job_env(scratch)
        result = run(["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                      "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"],
                     out / "configure.log", env=env)
        if result.returncode:
            raise ValueError("Native configure failed")
        targets = ["test-xiaomi0-frontend"]
        if args.inference:
            targets += ["test-xiaomi0-run", "test-xiaomi0-public"]
        result = run(["cmake", "--build", build, "--target", *targets, "-j", "2"],
                     out / "build.log", env=env)
        if result.returncode:
            raise ValueError("Native build failed")
        receipt = build_receipt(build, targets, report["source_commit"])
        (out / "native-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result = run([build / "test-xiaomi0-frontend", scratch / "frontend.gguf"], out / "frontend.log", env=env)
        report["frontend_pass"] = result.returncode == 0
        save()
        if result.returncode:
            raise ValueError("Original/native Xiaomi frontend mismatch")
        if args.inference:
            report["resources_before_checkpoint"] = resource_snapshot(out, scratch)
            require_resources(report["resources_before_checkpoint"], True)
            save()
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            if args.model_dir is None:
                directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                                    allow_patterns=list(policy.metadata_files()) + ["model.safetensors"]))
            policy.verify_checkpoint(directory / "model.safetensors", policy.expected_profile())
            report["phase"] = "original"; save()
            processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(directory, dtype=torch.float32,
                        attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
            if model.generation_config.eos_token_id != 248046:
                raise ValueError("Loaded Xiaomi model EOS differs from metadata")
            for case in protocol["cases"]:
                report["original"][case["id"]] = original_reference(directory, case, out, model, processor)
                save()
            del model, processor
            gc.collect()
            report["phase"] = "native"; save()
            for dtype in protocol["dtype_arms"]:
                output = scratch / ("xiaomi0-" + dtype + ".gguf")
                converter.convert(directory, output, policy.MODEL, policy.REVISION, dtype=dtype, profile="xiaomi-ocr-0")
                manifest = output.with_suffix(output.suffix + ".manifest.json")
                shutil.copyfile(manifest, out / manifest.name)
                report.setdefault("exports", {})[dtype] = {
                    "sha256": policy.file_sha256(output), "bytes": output.stat().st_size,
                    "weights_uploaded": False, "manifest": manifest.name}
                if dtype == "f16":
                    public = run([build / "test-xiaomi0-public", output], out / "public-admission.log", env=env)
                    report["public_admission_pass"] = public.returncode == 0
                    if public.returncode:
                        raise ValueError("Xiaomi generic public admission/state check failed")
                for case in protocol["cases"]:
                    key = dtype + "-" + case["id"]
                    prompt = out / (case["id"] + ".prompt.txt")
                    prompt.write_text(policy.evidence()["prompts"][case["task"]])
                    logits = out / (key + ".logits.f32")
                    result = run([build / "test-xiaomi0-run", output, ROOT / case["image"], case["max_new_tokens"],
                                  prompt, logits, protocol["threads"]], out / (key + ".log"), env=env)
                    if result.returncode:
                        raise ValueError("Native Xiaomi execution failed: " + key)
                    report["native"][key] = compare_native(result.stdout, logits,
                        out / (case["id"] + "-original.npz"), report["original"][case["id"]], dtype)
                    save()
                output.unlink()
            report["bounded_final_logit_bars_pass"] = all(x["owner_final_logit_bar_pass"] for x in report["native"].values())
        report["phase"] = "complete"
        report["scope"] = protocol["scope"]
        report["elapsed_seconds"] = time.monotonic() - started
        report["artifact_policy"] = "reports and actual reference/native tensors only; all GGUF/build/cache files in scratch"
        save()
        if args.inference and not report["bounded_final_logit_bars_pass"]:
            raise ValueError("Bounded final-logit precision bars failed; actual artifacts retained")
    except Exception as error:
        report["failure"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
