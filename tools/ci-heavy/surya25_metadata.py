"""Hosted weight-free source observation for datalab-to/surya-ocr-2.5.

Records exactly what the checkpoint declares before any converter or native
runtime work: resolved revision, file inventory, config/processor/generation/
tokenizer metadata, chat template, safetensors header shapes (bounded HTTP
ranges, no weights), the official meta-model schema comparison, the package's
training-time prompts rendered through the original chat template, and the
documented ``<|im_end|>`` spelling behaviour of the character-level vocabulary.

Every observation is captured independently: a failing step is recorded with a
redacted reason and the remaining steps still run, so one run yields the full
picture. Nothing here claims runtime support, quality or a qualified profile.
"""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback
import urllib.request
import zipfile

MODEL = "datalab-to/surya-ocr-2.5"
# Package source of the prompts / image scaling / eos-spelling contract.
SURYA_WHEEL = ("https://files.pythonhosted.org/packages/0e/d0/72916e91dd42ae304b9d1e4e00a0fb2bd114bb5461ef4a2c429c20e40004/"
               "surya_ocr-0.23.0-py3-none-any.whl")
SURYA_WHEEL_SHA256 = "42d84a89c5eb81db9d2e681dd181bc451a1e0d70ed14bd797a33beb553a0264f"
# github.com/datalab-to/surya main at 9d5aadfebb07939755c24e8a80d0e61b3be80895 ("Surya 2.5 support,
# release 0.23.0" merge) carries byte-identical copies of the four pinned files below.
SURYA_GIT_COMMIT = "9d5aadfebb07939755c24e8a80d0e61b3be80895"
SURYA_SOURCE_SHA256 = {
    "surya/inference/prompts.py": "e5557f0fb72283f86aea36e7a2bfeab885f962d324251e241983f87365b015d2",
    "surya/inference/util.py": "d853246db9767d01c8648058b779047ad0d93b3c74c0f7783dcf9faeef33b11b",
    "surya/scripts/build_gguf.py": "d20a1cd1526243ffd63798484463dce8f7377ae497a8f4cb99b8937b9856c383",
    "surya/inference/backends/openai_client.py": "431672016c553f1b30ba611d3bdcb24fb139c369d30ec622dfe65f4a72db9981",
}
METADATA_SUFFIXES = (".json", ".jinja", ".txt", ".md", ".model", ".py")
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
# Representative scale_to_fit outputs: a block crop, a minimum-size strip, a
# 192-DPI letter page and the 3072x2048 page cap documented by the package.
IMAGE_SIZES = [(448, 112), (1792, 28), (1632, 2112), (2048, 3072), (3072, 2048)]

OUT = Path(os.environ.get("HEAVY_OUT", "surya25-out"))
STEPS = {}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def redact(text):
    text = re.sub(r"https?://\S+", "<url>", str(text))
    text = re.sub(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]+", "<credential>", text)
    text = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s\]\)'\"]+)", "<path>", text)
    return text[:1500]


def step(name):
    def wrap(fn):
        def run(*args, **kwargs):
            try:
                value = fn(*args, **kwargs)
                STEPS[name] = {"ok": True}
                return value
            except Exception as error:  # Record and continue: one run, full picture.
                STEPS[name] = {"ok": False, "error": type(error).__name__, "detail": redact(error),
                               "trace": redact("".join(traceback.format_exc().splitlines(True)[-6:]))}
                print(f"STEP {name} FAILED: {type(error).__name__}: {redact(error)}", flush=True)
                return None
        return run
    return wrap


def fetch(url, byte_range=None, limit=MAX_METADATA_BYTES):
    headers = {"User-Agent": "crispembed-surya25-metadata"}
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = "Bearer " + token
    if byte_range:
        headers["Range"] = "bytes=%d-%d" % byte_range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Response exceeds bound")
        return response.status, dict(response.headers), data


@step("surya_package_source")
def surya_source():
    _, _, wheel = fetch(SURYA_WHEEL)
    if sha_bytes(wheel) != SURYA_WHEEL_SHA256:
        raise ValueError("surya-ocr 0.23.0 wheel digest differs")
    files = {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name, expected in SURYA_SOURCE_SHA256.items():
            data = archive.read(name)
            if sha_bytes(data) != expected:
                raise ValueError("Pinned package source differs: " + name)
            files[name] = data.decode()
    prompts = {}
    tree = ast.parse(files["surya/inference/prompts.py"])
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target.endswith("_PROMPT"):
                prompts[target] = ast.literal_eval(node.value)
    mapping = {"layout": "LAYOUT_PROMPT", "block": "BLOCK_PROMPT", "table_rec": "TABLE_REC_PROMPT",
               "high_accuracy_bbox": "HIGH_ACCURACY_BBOX_PROMPT"}
    from PIL import Image
    from typing import Tuple
    namespace = {"Image": Image, "Tuple": Tuple}
    util = ast.parse(files["surya/inference/util.py"])
    function = next(n for n in util.body if isinstance(n, ast.FunctionDef) and n.name == "scale_to_fit")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-scale-to-fit", "exec"), namespace)
    return {"wheel_sha256": SURYA_WHEEL_SHA256, "git_commit": SURYA_GIT_COMMIT, "sources": SURYA_SOURCE_SHA256,
            "prompts": {key: prompts[value] for key, value in mapping.items()},
            "scale_to_fit": namespace["scale_to_fit"]}


@step("revision_and_inventory")
def inventory():
    from huggingface_hub import HfApi
    info = HfApi().model_info(MODEL, files_metadata=True)
    files = [{"path": s.rfilename, "size": s.size,
              "lfs_sha256": (s.lfs.sha256 if getattr(s, "lfs", None) else None)} for s in info.siblings]
    card = getattr(info, "card_data", None)
    return {"revision": info.sha, "gated": getattr(info, "gated", None), "private": getattr(info, "private", None),
            "license": (card.get("license") if card else None),
            "license_name": (card.get("license_name") if card else None),
            "pipeline_tag": getattr(info, "pipeline_tag", None), "tags": sorted(getattr(info, "tags", []) or []),
            "library_name": getattr(info, "library_name", None),
            "files": sorted(files, key=lambda item: item["path"])}


@step("metadata_files")
def metadata(revision, files):
    from huggingface_hub import hf_hub_download
    root = None
    observed = {}
    for item in files:
        path = item["path"]
        if not path.endswith(METADATA_SUFFIXES) and not path.startswith("LICENSE"):
            continue
        if item["size"] and item["size"] > MAX_METADATA_BYTES:
            observed[path] = {"skipped": "size bound", "size": item["size"]}
            continue
        local = Path(hf_hub_download(MODEL, path, revision=revision))
        root = local.parents[len(Path(path).parts) - 1]
        data = local.read_bytes()
        observed[path] = {"size": len(data), "sha256": sha_bytes(data)}
    return root, observed


@step("safetensors_headers")
def headers(revision, files):
    shards = [item["path"] for item in files if item["path"].endswith(".safetensors")]
    tensors = {}
    for shard in shards:
        url = f"https://huggingface.co/{MODEL}/resolve/{revision}/{shard}"
        status, _, prefix = fetch(url, (0, 7), 8)
        length = int.from_bytes(prefix[:8], "little")
        if status != 206 or len(prefix) != 8 or not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError("Header length range not honored or out of bound")
        status, _, payload = fetch(url, (8, 8 + length - 1), length)
        if status != 206 or len(payload) != length:
            raise ValueError("Header range not honored")
        header = json.loads(payload)
        header.pop("__metadata__", None)
        for name, entry in header.items():
            if name in tensors:
                raise ValueError("Duplicate tensor across shards: " + name)
            tensors[name] = {"shard": shard, "dtype": entry["dtype"], "shape": entry["shape"]}
    return tensors


def summarize_tensors(tensors):
    groups, layer_zero, other, dtypes = {}, {}, {}, {}
    for name, entry in tensors.items():
        dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + 1
        generic = re.sub(r"\.(\d+)\.", ".N.", name)
        groups[generic] = groups.get(generic, 0) + 1
        if re.search(r"\.layers\.0\.|\.blocks\.0\.", name):
            layer_zero[name] = entry["shape"]
        elif not re.search(r"\.(layers|blocks)\.\d+\.", name):
            other[name] = entry["shape"]
    return {"count": len(tensors), "dtypes": dtypes, "group_counts": dict(sorted(groups.items())),
            "layer0_shapes": dict(sorted(layer_zero.items())), "nonlayer_shapes": dict(sorted(other.items())),
            "mtp_tensors": sorted(name for name in tensors if "mtp" in name.lower())}


@step("tokenizer_json_summary")
def tokenizer_summary(root):
    data = json.loads((root / "tokenizer.json").read_text())
    model = data["model"]
    vocab = model.get("vocab", {})
    lengths = {}
    for token in vocab:
        lengths[len(token)] = lengths.get(len(token), 0) + 1
    ends_lt = sorted(t for t in vocab if len(t) > 1 and t.endswith("<"))
    ids = sorted(vocab.values()) if isinstance(vocab, dict) else []
    return {"model_type": model.get("type"), "unk_token": model.get("unk_token"),
            "vocab_entries": len(vocab), "vocab_id_min": ids[0] if ids else None,
            "vocab_id_max": ids[-1] if ids else None,
            "vocab_ids_contiguous": bool(ids) and ids == list(range(ids[0], ids[0] + len(ids))),
            "token_char_length_histogram": dict(sorted(lengths.items())),
            "multichar_ending_lt_count": len(ends_lt), "multichar_ending_lt_sample": ends_lt[:40],
            "normalizer": data.get("normalizer"), "pre_tokenizer": data.get("pre_tokenizer"),
            "decoder": data.get("decoder"), "post_processor": data.get("post_processor"),
            "truncation": data.get("truncation"), "padding": data.get("padding"),
            "added_tokens": data.get("added_tokens", []),
            "model_keys": sorted(k for k in model if k != "vocab"),
            "merges_present": "merges" in model}


@step("official_schema")
def official_schema(root, tensors):
    import transformers
    from transformers import AutoConfig
    raw = json.loads((root / "config.json").read_text())
    patched = Path(OUT / "patched-config")
    patched.mkdir(parents=True, exist_ok=True)
    (patched / "config.json").write_text(json.dumps(dict(raw, architectures=["Qwen3_5ForConditionalGeneration"])))
    config = AutoConfig.from_pretrained(patched, local_files_only=True, trust_remote_code=False)
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    with torch.device("meta"):
        official = Qwen3_5ForConditionalGeneration(config)
    expected = {name: list(t.shape) for name, t in official.state_dict().items()}
    actual = {name: entry["shape"] for name, entry in tensors.items()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(n for n in actual if n in expected and actual[n] != expected[n])
    return {"transformers": transformers.__version__, "config_class": type(config).__name__,
            "source_architectures": raw.get("architectures"), "model_type": raw.get("model_type"),
            "expected": len(expected), "actual": len(actual), "missing": missing[:64], "missing_count": len(missing),
            "extra_count": len(extra), "extra_sample": extra[:64],
            "extra_groups": sorted({re.sub(r"\.(\d+)\.", ".N.", n) for n in extra})[:64],
            "shape_mismatch": [[n, actual[n], expected[n]] for n in wrong[:32]]}


@step("tokenizer_and_prompts")
def prompts(root, package):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    eos = tokenizer.eos_token_id
    report = {"tokenizer_class": type(tokenizer).__name__, "len": len(tokenizer), "eos_token": tokenizer.eos_token,
              "eos_token_id": eos, "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
              "bos_token": tokenizer.bos_token, "special_ids": {}, "prompts": {}}
    for text in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|vision_end|>",
                 "<|image_pad|>", "<think>", "</think>", "<|im_end!>"]:
        report["special_ids"][text] = tokenizer.convert_tokens_to_ids(text)
    for name, text in package["prompts"].items():
        ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
        report["prompts"][name] = {"text": text, "eos_spelled": eos not in ids,
                                   "ids_with_end_marker": ids,
                                   "pieces_tail": tokenizer.convert_ids_to_tokens(ids[-12:])}
    sample = "Grüße, 東京 — ½ x² <b>bold</b>\n\tTab"
    sample_ids = tokenizer.encode(sample, add_special_tokens=False)
    report["roundtrip_sample"] = {"ids": sample_ids, "pieces": tokenizer.convert_ids_to_tokens(sample_ids),
                                  "decoded": tokenizer.decode(sample_ids),
                                  "decoded_skip_special": tokenizer.decode(sample_ids, skip_special_tokens=True)}
    return report


@step("processor_render")
def processor_render(root, package):
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    image_processor = processor.image_processor
    result = {"processor_class": type(processor).__name__, "image_processor_class": type(image_processor).__name__,
              "image_processor": {k: v for k, v in image_processor.to_dict().items()
                                  if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
              "rendered": {}, "grids_default": {}, "grids_vllm_kwargs": {}}
    for name, text in package["prompts"].items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
        result["rendered"][name] = {"text": rendered, "sha256": sha_bytes(rendered.encode()), "ids": ids}
    scale = package["scale_to_fit"]
    for width, height in IMAGE_SIZES:
        image = Image.new("RGB", (width, height), (255, 255, 255))
        scaled = scale(image)
        key = f"{width}x{height}"
        for label, kwargs in (("grids_default", {}),
                              ("grids_vllm_kwargs", {"min_pixels": 3136, "max_pixels": 6291456})):
            try:
                out = image_processor(images=[scaled], return_tensors="pt", **kwargs)
                result[label][key] = {"scaled": list(scaled.size), "grid_thw": out["image_grid_thw"].tolist(),
                                      "pixel_values": list(out["pixel_values"].shape)}
            except Exception as error:
                result[label][key] = {"scaled": list(scaled.size), "error": redact(error)}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"model": MODEL, "scope": "weight-free source observation; no runtime/quality claim"}
    package = surya_source()
    if package:
        report["surya_package"] = {k: v for k, v in package.items() if k != "scale_to_fit"}
    listing = inventory()
    report["inventory"] = listing
    if listing:
        root, observed = metadata(listing["revision"], listing["files"]) or (None, None)
        report["metadata_files"] = observed
        tensors = headers(listing["revision"], listing["files"])
        if tensors:
            report["tensors"] = summarize_tensors(tensors)
            (OUT / "tensor-header.json").write_text(json.dumps(tensors, indent=1, sort_keys=True))
        if root:
            for name in ["config.json", "generation_config.json", "preprocessor_config.json",
                         "processor_config.json", "video_preprocessor_config.json", "tokenizer_config.json",
                         "special_tokens_map.json", "chat_template.jinja", "chat_template.json"]:
                path = root / name
                if path.exists():
                    text = path.read_text()
                    report.setdefault("files", {})[name] = json.loads(text) if name.endswith(".json") else text
            report["tokenizer_json"] = tokenizer_summary(root)
            if tensors:
                report["official_schema"] = official_schema(root, tensors)
            if package:
                report["tokenizer"] = prompts(root, package)
                report["processor"] = processor_render(root, package)
    report["steps"] = STEPS
    text = json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False, default=str)
    (OUT / "surya25-metadata.json").write_text(text + "\n")
    # Job logs are the review channel when artifacts are not reachable: print in bounded chunks.
    compact = dict(report)
    if "tokenizer_json" in compact and compact["tokenizer_json"]:
        compact["tokenizer_json"] = dict(compact["tokenizer_json"])
        compact["tokenizer_json"]["added_tokens"] = [[t.get("id"), t.get("content"), t.get("special")]
                                                     for t in compact["tokenizer_json"]["added_tokens"]]
    body = json.dumps(compact, sort_keys=True, ensure_ascii=True, default=str)
    print("SURYA25_REPORT_SHA256", sha_bytes(text.encode()))
    for index in range(0, len(body) if os.environ.get("SURYA25_LOG_FULL") == "1" else 0, 3000):
        print("SURYA25_REPORT_CHUNK %05d %s" % (index // 3000, body[index:index + 3000]))
    tensors = report.get("tensors") or {}
    summary = {"revision": (listing or {}).get("revision"), "official_schema": report.get("official_schema"),
               "tensor_count": tensors.get("count"), "dtypes": tensors.get("dtypes"),
               "mtp_tensors": tensors.get("mtp_tensors"), "nonlayer_shapes": tensors.get("nonlayer_shapes"),
               "group_counts": tensors.get("group_counts")}
    print("SURYA25_SUMMARY", json.dumps(summary, sort_keys=True, default=str))
    print("SURYA25_LAYER0", json.dumps(tensors.get("layer0_shapes"), sort_keys=True))
    print("SURYA25_STEPS", json.dumps(STEPS, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
