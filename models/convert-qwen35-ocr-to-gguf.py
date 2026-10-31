#!/usr/bin/env python3
"""Export dense Qwen3.5 OCR checkpoints for the pending native backend.

This is an experimental weight format, not a supported registry model. Compare
checkpoint names/shapes with the official Transformers model on meta before
writing anything. Do not silently drop unknown weights or fabricate defaults.
The source layout is preserved (including zero-centered RMSNorm weights).
"""

import argparse
import hashlib
import importlib.util
import inspect
from functools import lru_cache
import json
import re
from pathlib import Path


ARCH = "qwen35_ocr"
LIGHTON_MODEL = "lightonai/LightOnOCR-3-0.8B"
LIGHTON_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"


@lru_cache(maxsize=1)
def _ovis_profiles():
    spec=importlib.util.spec_from_file_location('qwen35_ovis_profile',Path(__file__).with_name('qwen35_ovis_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _ovis3_profile():
    spec=importlib.util.spec_from_file_location('qwen35_ovis3_profile',Path(__file__).with_name('qwen35_ovis3_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _ovis3_hf_profile():
    spec=importlib.util.spec_from_file_location('qwen35_ovis3_hf_profile',Path(__file__).with_name('qwen35_ovis3_hf_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _surya25_profile():
    spec=importlib.util.spec_from_file_location('qwen35_surya25_profile',Path(__file__).with_name('qwen35_surya25_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write_surya25_metadata(writer, root):
    """Format 4: character-level WordLevel vocabulary for the native core_wordlevel encoder."""
    import gguf
    policy = _surya25_profile()
    data = policy.read_tokenizer(root)
    writer.add_string("tokenizer.ggml.model", "surya25-wordlevel")
    writer.add_string("tokenizer.ggml.pre", "surya25")
    # ggml's gguf_get_arr_str() returns a C string, so the NUL token (U+0000) would read
    # back empty. Write it empty and name its id explicitly; native restores "\0".
    nul = [i for i, token in enumerate(data["tokens"]) if "\x00" in token]
    if nul != [policy.NUL_ID] or data["tokens"][policy.NUL_ID] != "\x00":
        raise ValueError("Unexpected NUL-bearing Surya tokens")
    writer.add_array("tokenizer.ggml.tokens", ["" if i == policy.NUL_ID else t for i, t in enumerate(data["tokens"])])
    writer.add_uint32(f"{ARCH}.nul_token_id", policy.NUL_ID)
    for key, values in [("wordlevel_model_ids", data["model_ids"]), ("added_token_ids", data["raw_added_ids"]),
                        ("normalized_added_token_ids", data["normalized_added_ids"]),
                        ("special_token_ids", data["special_ids"]), ("stop_token_ids", [policy.EOS_ID])]:
        writer.add_key_value(f"{ARCH}.{key}", values, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_uint32(f"{ARCH}.generation_eos_id", policy.EOS_ID)
    writer.add_uint32("tokenizer.ggml.eos_token_id", policy.EOS_ID)
    writer.add_uint32("tokenizer.ggml.padding_token_id", policy.PAD_ID)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", policy.UNK_ID)
    writer.add_string(f"{ARCH}.tokenizer_json_sha256", data["tokenizer_json_sha256"])
    writer.add_string(f"{ARCH}.profile", "surya25")
    writer.add_string(f"{ARCH}.profile.render_prefix", policy.RENDER_PREFIX)
    writer.add_string(f"{ARCH}.profile.render_suffix", policy.RENDER_SUFFIX)
    writer.add_string(f"{ARCH}.profile.prompts", json.dumps(policy.PROMPTS, sort_keys=True))
    writer.add_string(f"{ARCH}.checkpoint_sha256", policy.CHECKPOINT_SHA256)
    writer.add_uint32(f"{ARCH}.image_min_pixels", policy.IMAGE_MIN_PIXELS)
    writer.add_uint32(f"{ARCH}.image_max_pixels", policy.IMAGE_MAX_PIXELS)
    writer.add_string("general.license", "openrail")
    writer.add_bool(f"{ARCH}.image_max_qualified", False)
    writer.add_bool(f"{ARCH}.native_alias_qualified", False)


def precision_recipe(name):
    if name == "original-f16":
        return []
    if name != "vision-f32":
        raise ValueError(f"Unknown precision recipe: {name}")
    return ["v.patch_embed.proj.weight", "v.pos_embed.weight", "v.merger.linear_fc1.weight", "v.merger.linear_fc2.weight"] + [
        f"v.blocks.{i}.{part}.weight" for i in range(12)
        for part in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
    ]


def read_tokenizer_metadata(root, vocab_size, generation=None, allow_generation_eos_list=False):
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "BPE":
        raise ValueError("Expected an audited BPE tokenizer")
    vocab = dict(tokenizer["model"]["vocab"])
    for token in tokenizer["added_tokens"]:
        vocab[token["content"]] = token["id"]
    tokens = [""] * vocab_size
    occupied = set()
    for token, index in vocab.items():
        if index < 0 or index >= len(tokens) or index in occupied:
            raise ValueError(f"Invalid/duplicate tokenizer id: {index}")
        occupied.add(index)
        tokens[index] = token
    merges = tokenizer["model"]["merges"]
    merges = [item if isinstance(item, str) else " ".join(item) for item in merges]

    if tokenizer.get("normalizer") != {"type": "NFC"} or tokenizer["model"].get("dropout") is not None:
        raise ValueError("Unimplemented tokenizer normalization/dropout")
    if any(t.get("single_word") or t.get("lstrip") or t.get("rstrip") or t.get("normalized") for t in tokenizer["added_tokens"]):
        raise ValueError("Unimplemented added-token matching policy")
    if generation is None:
        generation = json.loads((root / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    if allow_generation_eos_list:
        if (not isinstance(eos, list) or eos != [248044, 248046]
                or any(type(value) is not int or value < 0 or value >= vocab_size for value in eos)):
            raise ValueError("OvisOCR3 source EOS list differs")
        eos = eos[0]  # Legacy scalar key; full source list is written under the OvisOCR3 profile.
    if not isinstance(eos, int) or eos < 0 or eos >= vocab_size:
        raise ValueError("Only a valid single generation EOS supported")
    config = json.loads((root / "tokenizer_config.json").read_text())
    return {"tokens": tokens, "merges": merges,
            "added": [t["id"] for t in tokenizer["added_tokens"]],
            "special": [t["id"] for t in tokenizer["added_tokens"] if t["special"]],
            "generation_eos": eos, "tokenizer_eos": vocab[config["eos_token"]],
            "pad": vocab[config["pad_token"]],
            "normalizer": "NFC", "pattern": tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]}


def write_tokenizer_metadata(writer, data):
    import gguf
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen35")
    writer.add_string(f"{ARCH}.normalizer", data["normalizer"])
    writer.add_string(f"{ARCH}.pretokenizer_pattern", data["pattern"])
    writer.add_array("tokenizer.ggml.tokens", data["tokens"])
    writer.add_array("tokenizer.ggml.merges", data["merges"])
    for key in ["added", "special"]:
        writer.add_key_value(f"{ARCH}.{key}_token_ids", data[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_uint32(f"{ARCH}.generation_eos_id", data["generation_eos"])
    writer.add_uint32("tokenizer.ggml.eos_token_id", data["tokenizer_eos"])
    writer.add_uint32("tokenizer.ggml.padding_token_id", data["pad"])


def validate_config(config, profile=None):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Only dense qwen3_5 checkpoints are supported")
    text = config["text_config"]
    vision = config["vision_config"]
    kinds = text["layer_types"]
    if len(kinds) != text["num_hidden_layers"] or not set(kinds) <= {
        "linear_attention", "full_attention"
    }:
        raise ValueError("Invalid or unsupported decoder layer schedule")
    if profile is not None and profile.get("name") == "surya25":
        _surya25_profile().validate_config(config)
    elif profile is not None:
        if profile.get("qualification_kind") == "hf5.16.1-original-reference":
            _ovis3_hf_profile().validate_config(config,profile)
        elif profile.get("name") == "ovisocr3":
            _ovis3_profile().validate_config(config,profile)
        else:
            _ovis_profiles().validate_config(config,profile)
    if text.get("mtp_num_hidden_layers", 0) and profile is None:
        raise ValueError("MTP checkpoints need a separately audited export policy")
    if vision.get("deepstack_visual_indexes", []):
        raise ValueError("DeepStack is not part of the audited Qwen3.5 OCR format")
    if text.get("attention_bias", False) or not text.get("attn_output_gate", False):
        raise ValueError("Expected bias-free gated full attention")
    rope = text["rope_parameters"]
    rotary = int(text["head_dim"] * rope["partial_rotary_factor"])
    if not rope["mrope_interleaved"] or sum(rope["mrope_section"]) * 2 != rotary:
        raise ValueError("Inconsistent partial/interleaved MRoPE configuration")
    return text, vision


def tensor_name(name):
    if name.startswith("model.language_model."):
        result = "l." + name.removeprefix("model.language_model.")
    elif name.startswith("model.visual."):
        result = "v." + name.removeprefix("model.visual.")
    elif name == "lm_head.weight":
        result = name
    else:
        raise ValueError(f"Unaudited checkpoint tensor prefix: {name}")
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"Tensor name exceeds native ggml limit: {result}")
    return result


def convert(model_dir, output, source_model, source_revision, dtype="f16", keep_f32=(), profile=None):
    root = Path(model_dir)
    output = Path(output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("An immutable 40-hex source revision is required")
    if profile not in (None,"ovisocr2","ovisocr3","ovisocr3-hf-reference","surya25"):
        raise ValueError("Unknown export profile")
    surya = profile == "surya25"
    if surya:
        policy = _surya25_profile()
        if (source_model, source_revision) != (policy.MODEL, policy.REVISION):
            raise ValueError("Surya 2.5 export requires its pinned source/revision")
        policy.verify_metadata(root)
        pinned = {"name": "surya25", "format_version": 4}
    else:
        policy=(_ovis3_hf_profile() if profile=="ovisocr3-hf-reference" else
                _ovis3_profile() if profile=="ovisocr3" else _ovis_profiles())
        pinned=policy.load_profile(root,source_model,source_revision) if profile else None
    config = json.loads((root / "config.json").read_text())
    text, vision = validate_config(config,pinned)
    if dtype not in {"f32", "f16"}:
        raise ValueError("Establish full-precision parity before adding quantized exports")
    if surya:
        if (root/"model.safetensors.index.json").exists() or \
                _surya25_profile().file_sha256(root/"model.safetensors") != _surya25_profile().CHECKPOINT_SHA256:
            raise ValueError("Surya 2.5 checkpoint differs from the pinned single file")
    elif pinned:
        if (root/"model.safetensors.index.json").exists():
            raise ValueError("Pinned Ovis checkpoint is a single file; unexpected shard index")
        policy.verify_checkpoint(root/"model.safetensors",pinned)

    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open
    import transformers
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    if pinned and not surya:
        expected=pinned['qualification']
        implementation=Path(inspect.getfile(Qwen3_5ForConditionalGeneration)).read_bytes().replace(b'\r\n',b'\n')
        if (transformers.__version__!=expected['transformers_version']
                or hashlib.sha256(implementation).hexdigest()!=expected['loaded_source_sha256']['model']):
            raise ValueError("Ovis meta-model implementation differs from qualification")

    # Official implementation supplies the independent expected parameter schema.
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if pinned and not surya: hf_config._attn_implementation="eager"
    with torch.device("meta"):
        official = Qwen3_5ForConditionalGeneration(hf_config)
    expected = {name: list(tensor.shape) for name, tensor in official.state_dict().items()}
    del official

    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(mapping.values()))
    else:
        shards = ["model.safetensors"]
        mapping = None
    actual = {}
    for shard in shards:
        # Check index paths before opening checkpoint-controlled file names.
        if Path(shard).name != shard:
            raise ValueError("Shard paths must be plain file names")
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in actual:
                    raise ValueError(f"Duplicate tensor: {name}")
                if mapping is not None and mapping.get(name) != shard:
                    raise ValueError(f"Shard/index mismatch: {name}")
                actual[name] = (shard, list(handle.get_slice(name).get_shape()))
    if mapping is not None and set(actual) != set(mapping):
        raise ValueError("Checkpoint index has missing or extra tensors")
    tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    missing = set(expected) - set(actual)
    if tied and missing == {"lm_head.weight"}:
        missing.clear()
    extra = set(actual) - set(expected)
    mismatched = [name for name in actual if name in expected and actual[name][1] != expected[name]]
    if missing or extra or mismatched:
        raise ValueError(f"Tensor schema mismatch: missing={sorted(missing)}, "
                         f"extra={sorted(extra)}, shape={mismatched}")
    names = [tensor_name(name) for name in actual]
    if len(names) != len(set(names)):
        raise ValueError("Export tensor names collide")
    keep_f32 = frozenset(keep_f32)
    exported_names = set(names) - ({"lm_head.weight"} if tied else set())
    if keep_f32 - exported_names:
        raise ValueError(f"Unknown F32 retention tensors: {sorted(keep_f32 - exported_names)}")
    if tied and "lm_head.weight" in actual:
        with safe_open(root / actual["lm_head.weight"][0], framework="pt") as head_file, \
             safe_open(root / actual["model.language_model.embed_tokens.weight"][0], framework="pt") as embed_file:
            if not torch.equal(head_file.get_tensor("lm_head.weight"),
                               embed_file.get_tensor("model.language_model.embed_tokens.weight")):
                raise ValueError("Checkpoint claims tied weights but stores unequal head/embedding")

    tokenizer_data = None if surya else read_tokenizer_metadata(
        root, text["vocab_size"],pinned["generation"] if pinned else None,
        allow_generation_eos_list=profile in ("ovisocr3","ovisocr3-hf-reference"))

    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".partial")
    writer = gguf.GGUFWriter(str(pending), ARCH, use_temp_file=True)
    writer.add_string("general.name", source_model)
    writer.add_string("general.source.huggingface.repository", source_model)
    writer.add_string("general.source.huggingface.revision", source_revision)
    writer.add_string(f"{ARCH}.status", "experimental-export-native-backend-pending")
    writer.add_uint32(f"{ARCH}.format_version", pinned["format_version"] if pinned else 1)
    writer.add_bool(f"{ARCH}.tie_word_embeddings", tied)
    writer.add_bool(f"{ARCH}.zero_centered_text_norm", True)
    # Preserve every config field, rather than treating nested EOS ids as synonyms.
    for filename in ["config.json", "processor_config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "generation_config.json"]:
        if pinned and filename in pinned.get('missing_source_files', ()):
            continue  # Preserve absence, never fabricate an upstream source file.
        writer.add_string(f"{ARCH}.{filename.removesuffix('.json')}", (root / filename).read_text())
    writer.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    if surya:
        write_surya25_metadata(writer, root)
    else:
        write_tokenizer_metadata(writer, tokenizer_data)
    if profile=="ovisocr2":
        writer.add_string(f"{ARCH}.profile",pinned['name'])
        writer.add_string(f"{ARCH}.profile.rendered_prompt",pinned['rendered_prompt'])
        writer.add_key_value(f"{ARCH}.stop_token_ids",pinned['stop_ids'],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
        writer.add_uint32(f"{ARCH}.primary_eos_id",pinned['primary_eos'])
        writer.add_array(f"{ARCH}.missing_source_files",pinned['missing_source_files'])
        writer.add_string(f"{ARCH}.resolved_generation_config",json.dumps(pinned['generation'],sort_keys=True))
        writer.add_string(f"{ARCH}.resolved_processor_config",json.dumps(pinned['processor'],sort_keys=True))
        writer.add_string(f"{ARCH}.metadata_qualification",json.dumps(pinned['qualification'],sort_keys=True))
        writer.add_string(f"{ARCH}.checkpoint_sha256",pinned['checkpoint_sha256'])
        writer.add_uint32(f"{ARCH}.image_min_pixels",pinned['image_min_pixels'])
        writer.add_uint32(f"{ARCH}.image_max_pixels",pinned['image_max_pixels'])
        writer.add_bool(f"{ARCH}.image_max_qualified",False)
        writer.add_bool(f"{ARCH}.native_alias_qualified",False)
    elif profile in ("ovisocr3","ovisocr3-hf-reference"):
        writer.add_array(f"{ARCH}.missing_source_files",pinned["missing_source_files"])
        writer.add_string(f"{ARCH}.profile",pinned['name'])
        writer.add_string(f"{ARCH}.profile.task",pinned['task'])
        writer.add_string(f"{ARCH}.profile.task_prompt",pinned['task_prompt'])
        writer.add_key_value(f"{ARCH}.source_eos_token_ids",pinned['source_eos_ids'],
                             gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
        writer.add_string(f"{ARCH}.source_generation_config",json.dumps(pinned['generation'],sort_keys=True))
        writer.add_string(f"{ARCH}.source_qualification",json.dumps(pinned['qualification'],sort_keys=True))
        writer.add_string(f"{ARCH}.checkpoint_sha256",pinned['checkpoint_sha256'])
        writer.add_uint32(f"{ARCH}.image_min_pixels",pinned['image_min_pixels'])
        writer.add_uint32(f"{ARCH}.image_max_pixels",pinned['image_max_pixels'])
        writer.add_bool(f"{ARCH}.image_max_qualified",False)
        if profile=="ovisocr3-hf-reference":
            writer.add_string(f"{ARCH}.profile.rendered_prompt",pinned['rendered_prompt'])
            writer.add_key_value(f"{ARCH}.stop_token_ids",pinned['stop_ids'],
                                 gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
            writer.add_uint32(f"{ARCH}.primary_eos_id",pinned['primary_eos'])
            writer.add_string(f"{ARCH}.metadata_qualification",pinned['metadata_producer_canonical_sha256'])
            writer.add_string(f"{ARCH}.original_reference_qualification",pinned['original_reference_producer_canonical_sha256'])
            writer.add_bool(f"{ARCH}.rendered_prompt_qualified",True)
            writer.add_bool(f"{ARCH}.processor_qualified",True)
            writer.add_bool(f"{ARCH}.hf_stop_policy_qualified",True)
            writer.add_string(f"{ARCH}.hf_stop_runtime_proof_sha256",pinned['hf_stop_runtime_proof_sha256'])
            writer.add_string(f"{ARCH}.stop_qualification_scope",pinned['stop_qualification_scope'])
            writer.add_bool(f"{ARCH}.effective_stops_qualified",False)
        else:
            writer.add_bool(f"{ARCH}.rendered_prompt_qualified",False)
            writer.add_bool(f"{ARCH}.processor_qualified",False)
            writer.add_bool(f"{ARCH}.hf_stop_policy_qualified",False)
            writer.add_bool(f"{ARCH}.effective_stops_qualified",False)
        writer.add_bool(f"{ARCH}.native_alias_qualified",False)

    records = []
    try:
        for shard in shards:
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for source in sorted(handle.keys()):
                    if tied and source == "lm_head.weight":
                        continue  # Native head must alias the exported embedding.
                    name = tensor_name(source)
                    tensor = handle.get_tensor(source).float().numpy()
                    source_shape = list(tensor.shape)
                    if name == "v.patch_embed.proj.weight":
                        # ggml supports at most four axes. HF applies this Conv3d
                        # to processor-flattened patches; preserve its C,T,H,W order.
                        tensor = tensor.reshape(tensor.shape[0], -1)
                    # Scalars, recurrent coefficients, convs and norms remain F32.
                    target = np.float16 if dtype == "f16" and name not in keep_f32 and tensor.ndim == 2 and "norm" not in name else np.float32
                    array = np.ascontiguousarray(tensor, dtype=target)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite or overflowing export: {source}")
                    writer.add_tensor(name, array)
                    records.append({"source": source, "name": name, "shape": list(array.shape),
                                    "source_shape": source_shape,
                                    "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except BaseException:
        writer.close()
        pending.unlink(missing_ok=True)
        raise
    writer.close()
    pending.replace(output)
    manifest = {"source_model": source_model, "source_revision": source_revision,
                "architecture": ARCH, "dtype": dtype, "keep_f32": sorted(keep_f32), "tie_word_embeddings": tied,
                "checkpoint_tensor_count": len(actual), "export_tensor_count": len(records),
                "records": records, "bytes": output.stat().st_size}
    if pinned:
        manifest["profile"]={k:v for k,v in pinned.items() if k!="proof_marker"}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LIGHTON_MODEL)
    parser.add_argument("--profile",choices=["ovisocr2","ovisocr3","ovisocr3-hf-reference","surya25"],
                        help="Pinned experimental Ovis export-only profile; not a native alias")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    parser.add_argument("--precision-recipe", choices=["original-f16", "vision-f32"], default="original-f16",
                        help="Experimental F16 text/F32 vision recipe verified on two owned RGB fixtures")
    parser.add_argument("--keep-f32", action="append", default=[], metavar="GGUF_NAME",
                        help="Keep this exact tensor in F32; repeat for multiple tensors (unknown names reject)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-hex commit")
    if args.profile == "surya25" or args.model == _surya25_profile().MODEL:
        policy = _surya25_profile()
        if args.profile != "surya25" or args.model != policy.MODEL or args.revision != policy.REVISION:
            parser.error("Surya 2.5 export requires its exact source/revision and --profile surya25")
        directory = snapshot_download(args.model, revision=args.revision,
                                      allow_patterns=list(policy.METADATA_FILES) + ["model.safetensors"])
        manifest = convert(directory, args.output, args.model, args.revision, args.dtype,
                           sorted(set(precision_recipe(args.precision_recipe) + args.keep_f32)), profile="surya25")
        print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
        return
    if args.profile or args.model in ("ATH-MaaS/OvisOCR2", "ATH-MaaS/OvisOCR3"):
        policy=(_ovis3_hf_profile() if args.profile=="ovisocr3-hf-reference" else
                _ovis3_profile() if args.profile=="ovisocr3" else _ovis_profiles())
        if args.profile not in ("ovisocr2","ovisocr3","ovisocr3-hf-reference") or args.model!=policy.MODEL or args.revision!=policy.REVISION:
            parser.error("Ovis export requires its exact source/revision and matching --profile")
        metadata=snapshot_download(args.model,revision=args.revision,
                                  allow_patterns=list(policy.metadata_files()))
        policy.load_profile(metadata,args.model,args.revision)  # Before any weight download.
    directory = snapshot_download(args.model, revision=args.revision,
                                  allow_patterns=(list(_ovis3_profile().metadata_files())+["*.safetensors"]
                                                  if args.profile in ("ovisocr3","ovisocr3-hf-reference") else
                                                  ["*.json", "*.jinja", "*.safetensors", "LICENSE*"]))
    retained = sorted(set(precision_recipe(args.precision_recipe) + args.keep_f32))
    manifest = convert(directory, args.output, args.model, args.revision, args.dtype, retained,profile=args.profile)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
