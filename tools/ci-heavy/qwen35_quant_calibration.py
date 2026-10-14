"""Hosted native quantization sweep; original F16 gates remain unchanged."""
import gc
import fnmatch
import hashlib
import json
import os
import re
import subprocess
from collections import Counter


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i]
        for j, right in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return previous[-1]


def errors(reference, actual):
    return {"cer": distance(reference, actual) / max(1, len(reference)),
            "wer": distance(reference.split(), actual.split()) / max(1, len(reference.split()))}


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_native(repo, build, baseline, quant_out, cases):
    import gguf
    import numpy as np
    from PIL import Image, ImageDraw

    root = quant_out / "native-calibration"
    root.mkdir(exist_ok=True)
    texts = ["Alpha beta\n67890", "Page 7\nReference", "Receipt\nPaid $8.75",
             "Cafe menu\nTea 2.40", "Section B\nDraft copy", "Sample text\nCode ZX9"]
    examples = []
    paths = []
    evaluation_hashes = {file_sha(case[1]) for case in cases}
    for i, text in enumerate(texts):
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).text((22, 64), text, fill="black", font_size=20 + 2 * (i % 3))
        path = root / (str(i) + ".png")
        image.save(path)
        fingerprint = file_sha(path)
        if fingerprint in evaluation_hashes or text in {case[4] for case in cases}:
            raise ValueError("Calibration/evaluation image or transcript overlap")
        paths.append(path)
        examples.append({"index": i, "gold": text, "image_sha256": fingerprint, "font_size": 20 + 2 * (i % 3)})
    stats_path = root / "native-ffn-stats.tsv"
    result = subprocess.run([str(build / "test-qwen35-imatrix"), str(baseline), str(stats_path), *map(str, paths)],
                            capture_output=True, text=True)
    (root / "native-collector.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    observations = re.findall(r'^CALIB index=(\d+) stop=(\d+) ids:([^\n]*)\nCALIB utf8_hex:([^\n]*)$', result.stdout, re.M)
    if len(observations) != len(texts):
        raise ValueError("Missing native calibration outputs")
    for entry, observation in zip(examples, observations):
        assert entry["index"] == int(observation[0])
        entry["stop"] = int(observation[1])
        entry["native_ids"] = list(map(int, observation[2].split()))
        entry["native_text"] = bytes.fromhex(observation[3]).decode("utf-8")
        entry["native_vs_gold"] = errors(entry["gold"], entry["native_text"])
    target = root / "native-ffn-imatrix.gguf"
    writer = gguf.GGUFWriter(str(target), "crispembed-imatrix")
    writer.add_uint32("imatrix.version", 1)
    expected = {f"l.layers.{i}.mlp.{kind}_proj.weight": 3584 if kind == "down" else 1024
                for i in range(24) for kind in ("gate", "up", "down")}
    collected = []
    for line in stats_path.read_text().splitlines():
        name, count, width, *values = line.split()
        count, width = int(count), int(width)
        if name not in expected or width != expected.pop(name) or count <= 0 or len(values) != width:
            raise ValueError("Native importance schema/count mismatch")
        sums = np.asarray(values, dtype=np.float64).astype(np.float32)
        if not np.isfinite(sums).all() or np.min(sums) < 0 or not np.max(sums) > 0:
            raise ValueError("Invalid native activation statistics")
        writer.add_uint64("count." + name, count)
        writer.add_tensor(name, sums)
        collected.append({"name": name, "columns": width, "rows": count,
                          "sum_min": float(np.min(sums)), "sum_max": float(np.max(sums)),
                          "sum_sha256": hashlib.sha256(sums.tobytes()).hexdigest()})
    if expected or len(collected) != 72 or len({row["rows"] for row in collected}) != 1:
        raise ValueError("Incomplete native FFN importance coverage")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return target, {"producer": "independent native F16 owning-context image/prompt/greedy inference; no reference arrays or teacher forcing",
                    "collector_gate": "CRISPEMBED_QWEN35_IMATRIX", "literal0_registration_rejected": True,
                    "source_recipe": "vision-f32", "data": "six disjoint owned synthetic images; same renderer family as evaluation, not a corpus",
                    "imatrix_sha256": file_sha(target), "imatrix_bytes": target.stat().st_size,
                    "matrices": collected, "examples": examples}


def run(repo, build, baseline, image_path, resized_path, local, reference, out, scratch, ffn=False):
    import gguf
    import numpy as np
    from PIL import Image, ImageDraw

    quant_out = out / "quantization"
    quant_out.mkdir(exist_ok=True)
    cases = [("original", image_path, out, "", "Hello OCR\n12345"),
             ("resized", resized_path, resized_path.parent, "", "Hello OCR\n12345")]
    prompted = quant_out / "prompted"
    prompted.mkdir(exist_ok=True)
    reference.dump_reference(local, image_path, prompted / "official-reference.npz", prompt="Transcribe the text.")
    gc.collect()
    cases.append(("custom-prompt", image_path, prompted, "Transcribe the text.", "Hello OCR\n12345"))
    invoice = quant_out / "invoice"
    invoice.mkdir(exist_ok=True)
    invoice_image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(invoice_image).text((20, 60), "Invoice 042\nTotal $12.50", fill="black", font_size=22)
    invoice_path = invoice / "input.png"
    invoice_image.save(invoice_path)
    reference.dump_reference(local, invoice_path, invoice / "official-reference.npz", max_new_tokens=32)
    gc.collect()
    cases.append(("invoice", invoice_path, invoice, "", "Invoice 042\nTotal $12.50"))

    # Retain strict F16 evidence alongside quantization diagnostics. A new
    # fixture failure must remain visible and blocks a production parity claim.
    strict = []
    for name, image, ref_root, prompt, gold in cases[2:]:
        result = subprocess.run([str(build / "test-qwen35-decoder"), str(baseline),
                                 str(ref_root / "official-delta.gguf"), "--native-frontend"],
                                capture_output=True, text=True)
        (quant_out / (name + "-f16-strict.log")).write_text(result.stdout + result.stderr)
        failures = [line for line in result.stdout.splitlines() if line.startswith("FAIL")]
        strict.append({"case": name, "passed": result.returncode == 0, "failures": failures,
                       "log": "quantization/" + name + "-f16-strict.log"})
        if name == "custom-prompt":
            result.check_returncode()

    head = ["l.embed_tokens.weight=f16"]
    controls = ["l.layers.*.linear_attn.in_proj_a.weight=f16", "l.layers.*.linear_attn.in_proj_b.weight=f16"]
    attention = ["l.layers.*.self_attn.*.weight=q8_0", "l.layers.*.linear_attn.in_proj_qkv.weight=q8_0",
                 "l.layers.*.linear_attn.in_proj_z.weight=q8_0", "l.layers.*.linear_attn.out_proj.weight=q8_0"]
    # Each arm is recorded, including failures to preserve greedy output.
    recipes = [("q8-vision-f32", "q8_0", []),
               ("q4-vision-f32", "q4_k", []),
               ("q8-head-f16", "q8_0", head),
               ("q4-head-f16", "q4_k", head),
               ("q4-head-controls-f16", "q4_k", head + controls),
               ("q4-attention-q8", "q4_k", head + attention + controls),
               ("q4-attention-f16", "q4_k", head + [x.replace("=q8_0", "=f16") for x in attention] + controls)]
    imatrix, importance = None, None
    if ffn:
        imatrix, importance = collect_native(repo, build, baseline, quant_out, cases)
        compact_attention = attention + controls
        recipes = [("q8-vision-f32", "q8_0", []), ("q4-vision-f32", "q4_k", []),
                   ("q4-ffn-down-q8", "q4_k", ["l.layers.*.mlp.down_proj.weight=q8_0"]),
                   ("q4-ffn-q6", "q4_k", ["l.layers.*.mlp.*_proj.weight=q6_k"]),
                   ("q4-ffn-q8", "q4_k", ["l.layers.*.mlp.*_proj.weight=q8_0"]),
                   ("q4-attention-q8-compact", "q4_k", compact_attention),
                   ("q4-attention-q8-ffn-q6", "q4_k", compact_attention + ["l.layers.*.mlp.*_proj.weight=q6_k"]),
                   ("q4-ffn-imatrix", "q4_k", []),
                   ("q4-attention-q8-imatrix", "q4_k", compact_attention)]
    source = gguf.GGUFReader(str(baseline))
    schema = {t.name: {"shape": t.shape.tolist(), "type": t.tensor_type.name,
                       "sha256": hashlib.sha256(t.data.tobytes()).hexdigest()} for t in source.tensors}
    del source
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "model": "lightonai/LightOnOCR-3-0.8B", "revision": "36d8636925327c671bd8e479c517a004f7566bd9",
              "baseline_bytes": baseline.stat().st_size, "vision_policy": "preserve all v.* at calibrated F32 recipe precision",
              "quant_gate": "CRISPEMBED_QWEN35_QUANT", "original_f16_gates": "unchanged",
              "scope": "four owned synthetic cases, ARM CPU F32 arithmetic; storage/quality calibration, not a corpus or latency benchmark",
              "reference": "original Transformers5.5.4/tokenizers0.22.2; official arrays test-only, teacher-forced scores labeled separately from independent native greedy context",
              "new_fixture_strict_f16": strict, "publication": "none", "arms": []}
    if ffn:
        report["native_importance"] = importance
    try:
        for name, kind, overrides in [("f16-baseline", None, [])] + recipes:
            model = baseline
            arm = {"name": name, "target": kind or "source-f16", "overrides": ["v.*=f32"] + overrides if kind else [], "cases": []}
            use_imatrix = name.endswith("-imatrix")
            arm["native_imatrix"] = use_imatrix
            report["arms"].append(arm)
            if kind:
                model = scratch / (name + ".gguf")
                command = [str(build / "crispembed-quantize"), str(baseline), str(model), kind]
                for rule in arm["overrides"]:
                    command += ["--tensor-type", rule]
                if use_imatrix:
                    command += ["--imatrix", str(imatrix)]
                result = subprocess.run(command, capture_output=True, text=True)
                (quant_out / (name + "-quantizer.log")).write_text(result.stdout + result.stderr)
                result.check_returncode()
                if use_imatrix and ", 72 with imatrix" not in result.stdout:
                    raise ValueError("Quantizer did not apply all72 native FFN importance vectors")
            reader = gguf.GGUFReader(str(model))
            stored = []
            if set(schema) != {t.name for t in reader.tensors}:
                raise ValueError("Quantizer changed tensor names/count")
            for tensor in reader.tensors:
                original = schema[tensor.name]
                if tensor.shape.tolist() != original["shape"]:
                    raise ValueError("Quantizer changed logical tensor shape: " + tensor.name)
                protected = tensor.name.startswith("v.") or len(tensor.shape) != 2
                if protected and (tensor.tensor_type.name != original["type"] or
                                  hashlib.sha256(tensor.data.tobytes()).hexdigest() != original["sha256"]):
                    raise ValueError("Critical vision/vector/conv changed: " + tensor.name)
                if kind and not protected:
                    expected_type = "Q8_0" if "embed" in tensor.name else kind.upper()
                    for rule in arm["overrides"]:
                        pattern, precision = rule.rsplit("=", 1)
                        if fnmatch.fnmatchcase(tensor.name, pattern):
                            expected_type = precision.upper()
                    if tensor.tensor_type.name != expected_type:
                        raise ValueError("Effective tensor precision differs from recipe: " + tensor.name)
                stored.append({"name": tensor.name, "type": tensor.tensor_type.name,
                               "shape": tensor.shape.tolist(), "bytes": tensor.data.nbytes})
            arm["bytes"] = model.stat().st_size
            arm["gguf_sha256"] = file_sha(model)
            arm["types"] = dict(Counter(t["type"] for t in stored))
            arm["tensor_audit"] = stored
            del reader
            for case, image, ref_root, prompt, gold in cases:
                result = subprocess.run([str(build / "test-qwen35-quant"), str(model), str(image),
                                         str(ref_root / "official-delta.gguf"), prompt], capture_output=True, text=True)
                log = result.stdout + result.stderr
                (quant_out / (name + "-" + case + ".log")).write_text(log)
                result.check_returncode()
                metrics = []
                for match in re.findall(r'^METRIC (\S+) n=(\d+) cos_min=(\S+) cos_global=(\S+) max_abs=(\S+) mean_abs=(\S+) rms=(\S+) norms=(\S+)/(\S+)$', log, re.M):
                    metrics.append(dict(stage=match[0],elements=int(match[1]),**dict(zip(
                        ("cos_min", "cos_global", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"),map(float,match[2:])))))
                teacher = [dict(step=int(x[0]),selected=int(x[1]),reference=int(x[2]),same=bool(int(x[3])))
                           for x in re.findall(r'^TEACHER step=(\d+) selected=(\d+) reference=(\d+) same=(\d+)$',log,re.M)]
                ids = list(map(int,re.search(r'^FREE ids:(.*)$',log,re.M)[1].split()))
                actual = bytes.fromhex(re.search(r'^FREE utf8_hex:(.*)$',log,re.M)[1]).decode('utf-8')
                match = re.search(r'^FREE ids_exact=(\d+) text_exact=(\d+) stop=(\d+) mean_confidence=(\S+) bytes=(\d+)$',log,re.M)
                with np.load(ref_root / "official-reference.npz") as original:
                    original_ids = original['generated.ids'].reshape(-1).tolist()
                    expected_text = bytes(original['input.decoded_utf8'].astype(np.uint8).reshape(-1)).decode('utf-8')
                entry = dict(name=case,prompt=prompt,gold=gold,reference_text=expected_text,native_text=actual,
                             reference_ids=original_ids,native_ids=ids,ids_exact=bool(int(match[1])),text_exact=bool(int(match[2])),
                             stop=int(match[3]),native_mean_confidence=float(match[4]),native_decoded_bytes=int(match[5]),
                             paired=errors(expected_text,actual),reference_vs_gold=errors(gold,expected_text),native_vs_gold=errors(gold,actual),
                             teacher_forced=teacher,metrics=metrics)
                assert len(teacher)==len(original_ids) and len(metrics)==77+len(original_ids)
                arm["cases"].append(entry)
                print(f'{name}/{case}: bytes={arm["bytes"]} ids_exact={entry["ids_exact"]} text_exact={entry["text_exact"]} stop={entry["stop"]}',flush=True)
            arm["all_ids_eos_text_exact"] = all(c["ids_exact"] and c["text_exact"] and c["stop"]==1 for c in arm["cases"])
            if name == "f16-baseline" and not arm["all_ids_eos_text_exact"]:
                raise ValueError("F16 baseline does not preserve original decoded output on new fixtures")
        passing = [a for a in report["arms"] if a["all_ids_eos_text_exact"]]
        report["smallest_exact_synthetic_recipe"] = min(passing,key=lambda a:a['bytes'])['name']
        report["result"] = "completed diagnostic sweep; exact synthetic outputs do not establish corpus quality or publication readiness"
    finally:
        (out / ("ffn-calibration.json" if ffn else "quant-calibration.json")).write_text(json.dumps(report,indent=2)+'\n')
