#!/usr/bin/env python3
"""Tokenize a Qwen2.5-VL chat prompt and output token IDs.

Builds the chat-format prompt with image_pad placeholders, tokenizes
via HuggingFace, and outputs the token IDs as a binary file or JSON
for the C++ engine to consume.

Usage:
    # Text-only prompt:
    python tools/qwen2vl_tokenize.py --prompt "Describe this image." \
        --output /tmp/tokens.bin

    # With image (inserts image_pad tokens):
    python tools/qwen2vl_tokenize.py --prompt "Describe this image." \
        --n-image-tokens 391 --output /tmp/tokens.bin

    # JSON output (for debugging):
    python tools/qwen2vl_tokenize.py --prompt "OCR this." \
        --n-image-tokens 391 --format json
"""

import argparse
import json
import struct
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--prompt", required=True)
    p.add_argument("--system", default=None, help="System prompt")
    p.add_argument("--n-image-tokens", type=int, default=0,
                   help="Number of merged image tokens (inserts image_pad)")
    p.add_argument("--output", default=None, help="Output binary file")
    p.add_argument("--format", choices=["bin", "json"], default="bin")
    args = p.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Build chat messages
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.n_image_tokens > 0:
        # Insert image placeholder in the user message
        content = [
            {"type": "image"},
            {"type": "text", "text": args.prompt},
        ]
    else:
        content = args.prompt

    messages.append({"role": "user", "content": content})

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)

    # If we have image tokens, the template inserts <|vision_start|><|image_pad|><|vision_end|>
    # but only ONE image_pad. We need to replace it with n_image_tokens copies.
    if args.n_image_tokens > 0:
        image_pad = "<|image_pad|>"
        # Find the single image_pad and replace with the right count
        idx = text.find(image_pad)
        if idx >= 0:
            # Remove the single placeholder
            before = text[:idx]
            after = text[idx + len(image_pad):]
            # Insert n copies
            text = before + (image_pad * args.n_image_tokens) + after

    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Output
    if args.format == "json" or not args.output:
        result = {
            "text": text,
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
        }
        # Annotate special tokens
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        result["special_ids"] = {
            "image_pad": image_pad_id,
            "vision_start": vision_start_id,
            "vision_end": vision_end_id,
        }
        n_img = sum(1 for t in token_ids if t == image_pad_id)
        result["n_image_pad_tokens"] = n_img

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
    else:
        # Binary: 4-byte int count, then count × 4-byte int32 token IDs
        with open(args.output, 'wb') as f:
            f.write(struct.pack('<I', len(token_ids)))
            for tid in token_ids:
                f.write(struct.pack('<i', tid))
        print(f"Wrote {len(token_ids)} tokens to {args.output}")


if __name__ == "__main__":
    main()
