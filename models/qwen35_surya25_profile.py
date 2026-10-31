"""Pinned Surya OCR 2.5 (datalab-to/surya-ocr-2.5) source policy for the qwen35_ocr export.

Weight-free observations (config, tokenizer, prompts, processor) are recorded in
tests/results/surya25/metadata.json. This module validates a downloaded snapshot
against those pins and turns the character-level WordLevel tokenizer into the
native core_wordlevel arrays. It makes no runtime, quality or alias claim.
"""

import hashlib
import json
from pathlib import Path

MODEL = "datalab-to/surya-ocr-2.5"
REVISION = "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3"
CHECKPOINT_SHA256 = "6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17"
CHECKPOINT_BYTES = 1378772208
FILE_SHA256 = {
    "config.json": "37b7e37ff4ca2e6ec7c67e2808b61274c0d08aeef80bb03349637743ae7db2ec",
    "generation_config.json": "b3bb9900f84f37fe8686fdb70f3a435d30ea89076810b5332b70fd21edde81e4",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer_config.json": "273af4267d997926aea04103971a6f317c16c503f2ef924150f83ed0eb3bdcfd",
    "tokenizer.json": "49753fd82a5800f975d18d301203450d42f1a46c2b0247343c93239f122684c1",
    "chat_template.jinja": "86f17a85672e7f367b5e6c6de6f67f53ede0fba4abb3d67c583a6ef647c1aa85",
    "LICENSE": "636dfb8eb19efc5835ed6efb878f8b694a93f028c715d2547d831e2a55440276",
}
VOCAB_SIZE = 88526
EOS_ID = 2            # <|im_end|>; generation_config eos_token_id (single id)
PAD_ID = 0            # <|endoftext|>
UNK_ID = 65424        # <unk>
NUL_ID = 1947         # the U+0000 character token (needs NUL-safe GGUF transport)
IMAGE_PAD_ID, VISION_START_ID, VISION_END_ID = 11, 9, 10
# vLLM serving contract in surya-ocr 0.23.0 (backends/vllm.py mm_processor_kwargs).
IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS = 3136, 6291456
# Package prompts (surya/inference/prompts.py, sha256 e5557f0f...). Training-time contract.
PROMPTS = {
    "block": "OCR this block image to HTML.",
    "high_accuracy_bbox": "OCR this image to HTML. Each block is a div with data-label and data-bbox "
                          "(x0 y0 x1 y1, normalized 0-1000).",
    "layout": 'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" '
              'fields. Bbox is x0 y0 x1 y1, normalized 0-1000.',
    "table_rec": 'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") '
                 'and "bbox" (x0 y0 x1 y1, normalized 0-1000).',
}
# Chat template rendering for one image followed by the text prompt (image first, as the package sends it).
RENDER_PREFIX = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
RENDER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
METADATA_FILES = tuple(FILE_SHA256)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_metadata(root):
    root = Path(root)
    for name, expected in FILE_SHA256.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Surya 2.5 source file differs from the pinned revision: {name}")


def render_prompt(prompt):
    return RENDER_PREFIX + prompt + RENDER_SUFFIX


def read_tokenizer(root):
    """Strictly validate tokenizer.json and return the native arrays."""
    root = Path(root)
    raw = (root / "tokenizer.json").read_bytes()
    data = json.loads(raw)
    model = data["model"]
    if (model.get("type") != "WordLevel" or model.get("unk_token") != "<unk>" or set(model) != {"type", "vocab", "unk_token"}
            or data.get("normalizer") is not None
            or data.get("pre_tokenizer") != {"type": "Split", "pattern": {"Regex": "[\\s\\S]"},
                                             "behavior": "Isolated", "invert": False}
            or data.get("decoder") != {"type": "Fuse"}
            or data.get("truncation") is not None or data.get("padding") is not None):
        raise ValueError("Unaudited Surya tokenizer pipeline")
    post = data.get("post_processor") or {}
    if post.get("type") != "TemplateProcessing" or post.get("special_tokens") != {} or \
            post.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}]:
        raise ValueError("Surya post-processor would add tokens")
    tokens = [None] * VOCAB_SIZE
    model_ids = []
    for text, index in model["vocab"].items():
        if not 0 <= index < VOCAB_SIZE or tokens[index] is not None or not text:
            raise ValueError("Invalid WordLevel vocabulary id")
        tokens[index] = text
        model_ids.append(index)
    raw_added, normalized_added, special = [], [], []
    for entry in data["added_tokens"]:
        index, text = entry["id"], entry["content"]
        if entry.get("single_word") or entry.get("lstrip") or entry.get("rstrip"):
            raise ValueError("Unimplemented added-token matching flag")
        if not 0 <= index < VOCAB_SIZE or not text or (tokens[index] is not None and tokens[index] != text):
            raise ValueError("Invalid added token id/content")
        tokens[index] = text
        (normalized_added if entry["normalized"] else raw_added).append(index)
        if entry["special"]:
            special.append(index)
    if any(t is None for t in tokens):
        raise ValueError("Vocabulary has holes")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate token texts")
    if (tokens[EOS_ID] != "<|im_end|>" or tokens[PAD_ID] != "<|endoftext|>" or tokens[UNK_ID] != "<unk>"
            or tokens[IMAGE_PAD_ID] != "<|image_pad|>" or tokens[VISION_START_ID] != "<|vision_start|>"
            or tokens[VISION_END_ID] != "<|vision_end|>" or UNK_ID not in special or EOS_ID not in special):
        raise ValueError("Surya special-token ids differ")
    return {"tokens": tokens, "model_ids": sorted(model_ids), "raw_added_ids": raw_added,
            "normalized_added_ids": normalized_added, "special_ids": special, "unk_id": UNK_ID,
            "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest()}


def validate_config(config):
    text, vision = config["text_config"], config["vision_config"]
    expected_text = {"hidden_size": 1024, "num_hidden_layers": 24, "intermediate_size": 3584, "head_dim": 256,
                     "num_attention_heads": 8, "num_key_value_heads": 2, "vocab_size": VOCAB_SIZE,
                     "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128,
                     "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "tie_word_embeddings": True,
                     "mtp_num_hidden_layers": 0, "rms_norm_eps": 1e-06}
    expected_vision = {"depth": 12, "hidden_size": 768, "intermediate_size": 3072, "num_heads": 12,
                       "out_hidden_size": 1024, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 2304, "deepstack_visual_indexes": []}
    for source, expected in ((text, expected_text), (vision, expected_vision)):
        for key, value in expected.items():
            if source.get(key) != value:
                raise ValueError(f"Surya 2.5 config field differs: {key}")
    if (config.get("eos_token_id") != EOS_ID or config.get("image_token_id") != IMAGE_PAD_ID
            or config.get("vision_start_token_id") != VISION_START_ID
            or config.get("vision_end_token_id") != VISION_END_ID):
        raise ValueError("Surya 2.5 multimodal token ids differ")
