"""Hosted fixed real-document panel; preserves quality failures as measurements."""
import fnmatch
import gc
import hashlib
import json
import re
import subprocess

from qwen35_quant_calibration import errors, file_sha


def parse(log, reference):
    metrics = []
    pattern = r'^METRIC (\S+) n=(\d+) cos_min=(\S+) cos_global=(\S+) max_abs=(\S+) mean_abs=(\S+) rms=(\S+) norms=(\S+)/(\S+)$'
    for x in re.findall(pattern, log, re.M):
        metrics.append(dict(stage=x[0], elements=int(x[1]), **dict(zip(
            ("cos_min", "cos_global", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"), map(float, x[2:])))))
    teacher = [dict(step=int(x[0]), selected=int(x[1]), reference=int(x[2]), same=bool(int(x[3])))
               for x in re.findall(r'^TEACHER step=(\d+) selected=(\d+) reference=(\d+) same=(\d+)$', log, re.M)]
    ids = list(map(int, re.search(r'^FREE ids:(.*)$', log, re.M)[1].split()))
    text = bytes.fromhex(re.search(r'^FREE utf8_hex:(.*)$', log, re.M)[1]).decode('utf-8')
    match = re.search(r'^FREE ids_exact=(\d+) text_exact=(\d+) stop=(\d+) mean_confidence=(\S+) bytes=(\d+)$', log, re.M)
    assert len(teacher) == len(reference["generated_ids"]) and len(metrics) == 77 + len(teacher)
    scores = [x for x in metrics if x["stage"].startswith("generated.scores.")]
    return dict(native_ids=ids, native_text=text, ids_exact=bool(int(match[1])), text_exact=bool(int(match[2])),
                stop=int(match[3]), native_mean_confidence=float(match[4]), native_decoded_bytes=int(match[5]),
                native_recognize_ms=float(re.search(r'^FREE time_ms=(\S+)$', log, re.M)[1]),
                native_timing_scope="owning recognize_file after context load; image decode/preprocess/vision/prefill/free greedy, no teacher diagnostics",
                native_lines=len(text.splitlines()), reference_lines=len(reference["decoded"].splitlines()),
                paired=errors(reference["decoded"], text), teacher_forced=teacher, metrics=metrics,
                score_summary=dict(vectors=len(scores), min_cosine=min(x["cos_min"] for x in scores),
                                   max_absolute_error=max(x["max_abs"] for x in scores)))


def run(repo, build, baseline, local, out, scratch, module, anchor_image, anchor_npz):
    import gguf
    from PIL import Image

    capture = module(repo / "tools/dump_qwen35_ocr_page_reference.py", "qwen35_page_reference")
    panel = out / "real-pages"
    panel.mkdir(exist_ok=True)
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "model": "lightonai/LightOnOCR-3-0.8B", "revision": "36d8636925327c671bd8e479c517a004f7566bd9",
              "scope": "four fixed repository CC0/public-domain documents; no corpus or upstream benchmark score",
              "reference": "original Transformers5.5.4/tokenizers0.22.2 F32 eager generate; compact capture verified against full anchor",
              "native": "independent image/prompt/tokenizer/free greedy; teacher-forced score diagnostics separate",
              "production_changes": "none", "publication": "none", "cases": [], "arms": []}
    try:
        anchor = capture.dump_reference(local, anchor_image, panel / "anchor", max_new_tokens=12, anchor=anchor_npz)
        report["compact_anchor"] = anchor
        gc.collect()
        root = repo / "tests/regression/images/cc0"
        manifest = json.loads((root / "MANIFEST.json").read_text())
        catalog = {x["file"]: x for x in manifest}
        gold = {x["file"]: x for x in json.loads((root / "ground_truth.json").read_text())["records"]}
        specs = [("arabic-line", "arabic_printed_line.png", None, None, 64, None),
                 ("form", "simple_form.png", None, None, 192, "simple_form.png"),
                 ("receipt", "receipt_example.png", None, (448, 576), 384, "commons_example_receipt.png"),
                 ("german-header", "german_official_print.jpg", (0, 0, 1920, 650), (768, 256), 128, None)]
        for name, filename, box, size, cap, gold_key in specs:
            source = root / filename
            provenance = catalog[filename]
            if provenance["license"] not in ("CC0-1.0", "Public-Domain") or file_sha(source) != provenance["sha256"]:
                raise ValueError("Fixture license/checksum mismatch: " + filename)
            folder = panel / name
            folder.mkdir(exist_ok=True)
            image = Image.open(source).convert("RGB")
            original_size = list(image.size)
            if box is not None:
                image = image.crop(box)
            if size is not None:
                image = image.resize(size, resample=Image.Resampling.BICUBIC)
            path = folder / "input.png"
            image.save(path)
            item = dict(name=name, source=provenance, original_size=original_size, crop=box,
                        explicit_resize=size, resampler="Pillow uint8 RGB bicubic" if size else None,
                        input_sha256=file_sha(path), max_new_tokens=cap,
                        gold=({k: v for k, v in gold[gold_key].items() if k != "notes"} if gold_key else None),
                        annotation_policy="existing annotations unchanged; no guessed crop/Arabic gold; form medium confidence/occluded checkbox ambiguity")
            report["cases"].append(item)
            item["reference"] = capture.dump_reference(local, path, folder, max_new_tokens=cap)
            if item["gold"]:
                item["reference_vs_gold"] = errors(item["gold"]["text"], item["reference"]["decoded"])
            gc.collect()
        original = gguf.GGUFReader(str(baseline))
        schema = {t.name: dict(shape=t.shape.tolist(), type=t.tensor_type.name,
                              sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in original.tensors}
        del original
        attention = ["l.layers.*.self_attn.*.weight=q8_0", "l.layers.*.linear_attn.in_proj_qkv.weight=q8_0",
                     "l.layers.*.linear_attn.in_proj_z.weight=q8_0", "l.layers.*.linear_attn.out_proj.weight=q8_0",
                     "l.layers.*.linear_attn.in_proj_a.weight=f16", "l.layers.*.linear_attn.in_proj_b.weight=f16",
                     "l.layers.*.mlp.*_proj.weight=q6_k"]
        recipes = [("f16-baseline", None, []), ("q8-vision-f32", "q8_0", []),
                   ("attention-q8-ffn-q6", "q4_k", attention)]
        for name, kind, overrides in recipes:
            model = baseline
            arm = dict(name=name, overrides=["v.*=f32"] + overrides if kind else [], cases=[])
            report["arms"].append(arm)
            if kind:
                model = scratch / (name + ".gguf")
                command = [str(build / "crispembed-quantize"), str(baseline), str(model), kind]
                for rule in arm["overrides"]:
                    command += ["--tensor-type", rule]
                result = subprocess.run(command, capture_output=True, text=True)
                (panel / (name + "-quantizer.log")).write_text(result.stdout + result.stderr)
                result.check_returncode()
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
                    raise ValueError("Changed protected vision/vector/conv bytes")
                if kind and not protected:
                    expected = "Q8_0" if "embed" in tensor.name else kind.upper()
                    for rule in arm["overrides"]:
                        pattern, precision = rule.rsplit("=", 1)
                        if fnmatch.fnmatchcase(tensor.name, pattern):
                            expected = precision.upper()
                    if tensor.tensor_type.name != expected:
                        raise ValueError("Wrong effective tensor precision")
                audit.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                                  shape=tensor.shape.tolist(), bytes=tensor.data.nbytes))
            del reader
            arm.update(bytes=model.stat().st_size, gguf_sha256=file_sha(model), tensor_audit=audit)
            for case in report["cases"]:
                folder = panel / case["name"]
                result = subprocess.run([str(build / "test-qwen35-quant"), str(model), str(folder / "input.png"),
                                         str(folder / "official-delta.gguf"), "", str(case["max_new_tokens"])],
                                        capture_output=True, text=True)
                log = result.stdout + result.stderr
                (folder / (name + "-native.log")).write_text(log)
                result.check_returncode()
                measured = parse(log, case["reference"])
                measured["name"] = case["name"]
                measured["same_stop"] = measured["stop"] == (1 if case["reference"]["reached_eos"] else 2)
                if case["gold"]:
                    measured["native_vs_gold"] = errors(case["gold"]["text"], measured["native_text"])
                arm["cases"].append(measured)
                print(name, case["name"], "exact IDs/text/stop", measured["ids_exact"], measured["text_exact"],
                      measured["same_stop"], "paired", measured["paired"], flush=True)
            arm["all_ids_text_stop_exact"] = all(c["ids_exact"] and c["text_exact"] and c["same_stop"] for c in arm["cases"])
        report["result"] = "completed fixed-panel measurements; quality differences retained, no tolerance/default changes"
    finally:
        (out / "page-parity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
