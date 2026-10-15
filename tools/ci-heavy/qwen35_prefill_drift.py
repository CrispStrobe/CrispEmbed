"""Hosted fixed-input weight-rounding/backend isolation; no new quality claim."""
import gc
import hashlib
import json
import os
import re
import subprocess
from collections import Counter

from qwen35_quant_calibration import file_sha


def run(repo, build, baseline, local, out, scratch, module, converter, anchor_image, anchor_npz):
    import gguf
    from PIL import Image

    capture = module(repo / "tools/dump_qwen35_ocr_page_reference.py", "qwen35_drift_reference")
    root = out / "prefill-drift"
    root.mkdir(exist_ok=True)
    frozen = json.loads((repo / "tests/results/qwen35-ocr/page-parity.json").read_text())
    report = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  model=converter.LIGHTON_MODEL, revision=converter.LIGHTON_REVISION,
                  scope="same three larger document inputs, prefill only; no full-output, quality or speed promotion",
                  prior_workflow=frozen["workflow"], key_absolute_budget=0.002,
                  reference="original Transformers5.5.4 F32 eager; backend-off and per-patch-convolution controls",
                  original_numeric_gates="unchanged", cases=[], arms=[])
    try:
        report["anchor"] = capture.dump_reference(local, anchor_image, root / "anchor", max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        for old in frozen["cases"]:
            if old["name"] not in ("form", "receipt", "german-header"):
                continue
            source = repo / old["source"]["local_file"]
            if file_sha(source) != old["source"]["sha256"]:
                raise ValueError("Changed original document")
            folder = root / old["name"]
            folder.mkdir(exist_ok=True)
            image = Image.open(source).convert("RGB")
            if old["crop"]:
                image = image.crop(old["crop"])
            if old["explicit_resize"]:
                image = image.resize(old["explicit_resize"], resample=Image.Resampling.BICUBIC)
            path = folder / "input.png"
            image.save(path)
            if file_sha(path) != old["input_sha256"]:
                raise ValueError("Changed evaluation derivative bytes")
            case = dict(name=old["name"], source=old["source"], crop=old["crop"], explicit_resize=old["explicit_resize"],
                        input_sha256=file_sha(path))
            report["cases"].append(case)
            case["reference"] = capture.dump_reference(local, path, folder, max_new_tokens=1, prefill_controls=True)
            if any(len(rows) != 76 for rows in case["reference"]["cpu_controls"].values()):
                raise ValueError("Incomplete original backend prefill controls")
            gc.collect()
        unrounded = scratch / "lightonocr-all-f32.gguf"
        manifest = converter.convert(local, unrounded, converter.LIGHTON_MODEL, converter.LIGHTON_REVISION, dtype="f32")
        report["unrounded_conversion"] = dict(bytes=manifest["bytes"], tensors=len(manifest["records"]), dtype="f32")
        reader = gguf.GGUFReader(str(baseline))
        schema = {t.name: dict(shape=t.shape.tolist(), type=t.tensor_type.name,
                              sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
        del reader
        for name, model, precise in [("f16-text-vision-f32", baseline, False),
                                     ("all-f32", unrounded, False), ("all-f32-precise-rms", unrounded, True)]:
            reader = gguf.GGUFReader(str(model))
            if set(schema) != {t.name for t in reader.tensors}:
                raise ValueError("Changed tensor names/count")
            audit = []
            for tensor in reader.tensors:
                before = schema[tensor.name]
                protected = tensor.name.startswith("v.") or len(tensor.shape) != 2
                if tensor.shape.tolist() != before["shape"]:
                    raise ValueError("Changed tensor shape")
                if protected and (tensor.tensor_type.name != before["type"] or
                                  hashlib.sha256(tensor.data.tobytes()).hexdigest() != before["sha256"]):
                    raise ValueError("Changed vision/vector/conv bytes")
                if model == unrounded and tensor.tensor_type.name != "F32":
                    raise ValueError("Unrounded control contains non-F32 weights")
                audit.append(dict(name=tensor.name, shape=tensor.shape.tolist(), type=tensor.tensor_type.name,
                                  bytes=tensor.data.nbytes))
            del reader
            arm = dict(name=name, bytes=model.stat().st_size, gguf_sha256=file_sha(model),
                       precise_rms=precise, types=dict(Counter(t["type"] for t in audit)), tensor_audit=audit, cases=[])
            report["arms"].append(arm)
            for case in report["cases"]:
                folder = root / case["name"]
                env = dict(os.environ, CRISPEMBED_QWEN35_PRECISE_RMS="1" if precise else "0")
                result = subprocess.run([str(build / "test-qwen35-quant"), str(model), str(folder / "input.png"),
                                         str(folder / "official-delta.gguf"), "", "1", "--prefill-only"],
                                        capture_output=True, text=True, env=env)
                log = result.stdout + result.stderr
                (folder / (name + ".log")).write_text(log)
                result.check_returncode()
                metrics = []
                pattern = r'^METRIC (\S+) n=(\d+) cos_min=(\S+) cos_global=(\S+) max_abs=(\S+) mean_abs=(\S+) rms=(\S+) norms=(\S+)/(\S+)$'
                for x in re.findall(pattern, log, re.M):
                    metrics.append(dict(stage=x[0], elements=int(x[1]), **dict(zip(
                        ("cos_min", "cos_global", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"), map(float, x[2:])))))
                status = re.search(r'^PREFILL_ONLY finite=1 metrics=77 selected=(\d+) reference=(\d+) strict_failures=(\d+)$', log, re.M)
                if len(metrics) != 77 or not status:
                    raise ValueError("Missing finite prefill measurements")
                keys = [m for m in metrics if m["stage"].endswith(".keys")]
                cached = [m for m in metrics if m["stage"].startswith("prefill.cache.")]
                entry = dict(name=case["name"], metrics=metrics, selected=int(status[1]), reference=int(status[2]),
                             strict_failed_stages=re.findall(r'^FAIL (\S+): cos_min=', log, re.M),
                             strict_budget_records=[dict(stage=x[0], strict_absolute_pass=bool(int(x[1])),
                                                         calibrated=bool(int(x[2])), self_max_abs=float(x[3]), max_ratio=float(x[4]))
                                 for x in re.findall(r'^BUDGET (\S+): strict_absolute_pass=(\d+) calibrated=(\d+) self_max_abs=(\S+) max_ratio=(\S+)$', log, re.M)],
                             key_max_abs=max(m["max_abs"] for m in keys),
                             strict_cache_exceedances=[m for m in cached if m["max_abs"] > 0.002])
                if len(entry["strict_failed_stages"]) != int(status[3]) or len(entry["strict_budget_records"]) != 77:
                    raise ValueError("Incomplete unchanged strict-budget diagnostics")
                arm["cases"].append(entry)
                print(name, case["name"], "max key", entry["key_max_abs"],
                      "cache budget exceedances", len(entry["strict_cache_exceedances"]), flush=True)
        report["result"] = "completed weight/backend controls; preserve all failures, no default or tolerance change"
    finally:
        (out / "prefill-drift.json").write_text(json.dumps(report, indent=2) + "\n")
