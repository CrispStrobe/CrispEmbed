#!/usr/bin/env python3
"""End-to-end InternVL2 OCR test via CrispEmbed Python bindings.

Uses HuggingFace tokenizer for prompt encoding/decoding, CrispEmbed
C library for vision + LLM inference.

Usage:
    PYTHONNOUSERSITE=1 python tests/test_internvl2_ocr.py \
        --model /mnt/storage/gguf-models/internvl2.5-2b-q8_0.gguf \
        --image test_images/document.png \
        --tokenizer OpenGVLab/InternVL2_5-2B
"""

import argparse
import ctypes
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--image", default=None, help="Image path (optional)")
    parser.add_argument("--tokenizer", default="OpenGVLab/InternVL2_5-2B",
                        help="HF tokenizer model ID")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Describe this image in detail.",
                        help="User prompt")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Build chat prompt with image placeholders
    # InternVL2.5 template: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    n_image_tokens = 256  # 1 tile × 256 tokens/tile

    system_msg = "You are a helpful assistant."
    image_placeholder = IMG_CONTEXT_TOKEN * n_image_tokens

    conversation = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholder}\n{args.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    token_ids = tok.encode(conversation, add_special_tokens=False)
    print(f"Prompt: {len(token_ids)} tokens")

    # Find image token positions
    img_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    n_img_found = sum(1 for t in token_ids if t == img_token_id)
    print(f"Image tokens: {n_img_found} (id={img_token_id})")

    # Load CrispEmbed library
    lib_path = Path("build/libcrispembed.so")
    if not lib_path.exists():
        lib_path = Path("build/libcrispembed-static.a")
    # Try the shared lib from the ggml build
    for candidate in [
        "build/libcrispembed.so",
        "build/libcrispembed.dylib",
    ]:
        if Path(candidate).exists():
            lib_path = Path(candidate)
            break

    print(f"\nNote: C library integration requires the Python crispembed bindings.")
    print(f"For now, this test outputs the tokenized prompt for manual testing.")
    print(f"\nTokenized prompt ({len(token_ids)} tokens):")
    print(f"  First 10: {token_ids[:10]}")
    print(f"  Image range: [{token_ids.index(img_token_id)}..{len(token_ids) - 1 - token_ids[::-1].index(img_token_id)}]")
    print(f"  Last 10: {token_ids[-10:]}")

    # Decode a sample output for verification
    sample_output = [14016, 350, 12819, 364, 882, 413, 2226, 1122, 388, 821]
    decoded = tok.decode(sample_output)
    print(f"\nSample decode of generated tokens {sample_output}:")
    print(f"  → '{decoded}'")

    print("\nTo run full inference:")
    print(f"  ./build/test-internvl2-e2e {args.model} {args.max_tokens}")


if __name__ == "__main__":
    main()
