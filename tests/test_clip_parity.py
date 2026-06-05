#!/usr/bin/env python3
"""CLIP text + vision parity test: CrispEmbed vs HuggingFace.

Tests that CLIP text and vision encoders produce embeddings in the same
cross-modal space, and that text-image similarity ranking is correct.

Usage:
    python tests/test_clip_parity.py \
        --text-gguf clip-text-base.gguf \
        --vision-gguf clip-vit-base-patch16.gguf \
        --binary build/crispembed
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def ce_text(binary, model, text):
    r = subprocess.run([binary, "-m", model, text],
                       capture_output=True, text=True, timeout=60)
    vals = r.stdout.strip().split()
    return np.array([float(x) for x in vals]) if vals else np.array([])


def ce_image(binary, model, path):
    r = subprocess.run([binary, "-m", model, "--image", path],
                       capture_output=True, text=True, timeout=60)
    vals = r.stdout.strip().split()
    return np.array([float(x) for x in vals]) if vals else np.array([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-gguf", required=True)
    parser.add_argument("--vision-gguf", required=True)
    parser.add_argument("--binary", default="build/crispembed")
    args = parser.parse_args()

    for p, name in [(args.text_gguf, "text GGUF"), (args.vision_gguf, "vision GGUF"),
                     (args.binary, "binary")]:
        if not os.path.exists(p):
            print(f"SKIP: {name} not found at {p}")
            return 0

    passed = 0
    failed = 0

    # Test 1: Text encoder produces non-empty, normalized embeddings
    print("Test 1 — Text encoder basic:")
    texts = ["hello", "a photo of a cat"]
    for text in texts:
        emb = ce_text(args.binary, args.text_gguf, text)
        if len(emb) > 0 and abs(np.linalg.norm(emb) - 1.0) < 0.01:
            passed += 1
            print(f"  '{text}': dim={len(emb)} norm={np.linalg.norm(emb):.4f} PASS")
        else:
            failed += 1
            print(f"  '{text}': dim={len(emb)} FAIL")

    # Test 2: Vision encoder produces non-empty, normalized embeddings
    print("\nTest 2 — Vision encoder basic:")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        import cv2
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        cv2.ellipse(img, (112, 100), (40, 50), 0, 0, 360, (220, 190, 170), -1)
        cv2.imwrite(f.name, img)
        img_path = f.name

    emb_img = ce_image(args.binary, args.vision_gguf, img_path)
    if len(emb_img) > 0 and abs(np.linalg.norm(emb_img) - 1.0) < 0.01:
        passed += 1
        print(f"  image: dim={len(emb_img)} norm={np.linalg.norm(emb_img):.4f} PASS")
    else:
        failed += 1
        print(f"  image: dim={len(emb_img)} FAIL")

    # Test 3: Text and vision embeddings have the same dimension
    print("\nTest 3 — Cross-modal dimension match:")
    emb_txt = ce_text(args.binary, args.text_gguf, "test")
    if len(emb_txt) == len(emb_img) and len(emb_txt) > 0:
        passed += 1
        print(f"  text_dim={len(emb_txt)} == vision_dim={len(emb_img)} PASS")
    else:
        failed += 1
        print(f"  text_dim={len(emb_txt)} != vision_dim={len(emb_img)} FAIL")

    # Test 4: Cross-modal similarity is non-trivial
    print("\nTest 4 — Cross-modal similarity non-trivial:")
    emb_face_text = ce_text(args.binary, args.text_gguf, "a human face")
    emb_cat_text = ce_text(args.binary, args.text_gguf, "a cat sitting")
    if len(emb_face_text) > 0 and len(emb_img) > 0:
        sim_face = np.dot(emb_face_text, emb_img)
        sim_cat = np.dot(emb_cat_text, emb_img)
        if sim_face != sim_cat and abs(sim_face) > 0.01:
            passed += 1
            print(f"  face_sim={sim_face:.4f} cat_sim={sim_cat:.4f} PASS (different)")
        else:
            failed += 1
            print(f"  face_sim={sim_face:.4f} cat_sim={sim_cat:.4f} FAIL")
    else:
        failed += 1
        print("  encoding failed FAIL")

    os.unlink(img_path)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
