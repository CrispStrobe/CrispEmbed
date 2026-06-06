#!/usr/bin/env python3
"""Face search example: index faces from a directory, query by image.

Usage:
    # Index all images in a directory, then query
    python examples/face_search.py --index-dir photos/ --query query.jpg

    # Use registry names (auto-download)
    python examples/face_search.py --det yunet --rec auraface-v1 \
        --index-dir photos/ --query query.jpg

    # Specify GGUF paths directly
    python examples/face_search.py --det yunet.gguf --rec auraface-v1.gguf \
        --index-dir photos/ --query query.jpg

Environment:
    CRISPEMBED_LIB   Path to libcrispembed.{so,dylib,dll}
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from crispembed import CrispFacePipeline


def index_faces(pipe: CrispFacePipeline, image_dir: str, conf: float = 0.5):
    """Index all faces in *image_dir*. Returns list of (path, face_idx, embedding)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    entries = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        path = os.path.join(image_dir, fname)
        results = pipe.run(path, conf=conf)
        for i, r in enumerate(results):
            entries.append((path, i, r["embedding"]))
        if results:
            print(f"  {fname}: {len(results)} face(s)")
        else:
            print(f"  {fname}: no faces")
    return entries


def search(entries, query_emb: np.ndarray, top_n: int = 5):
    """Return top-N matches by cosine similarity."""
    scores = []
    for path, face_idx, emb in entries:
        cos = float(np.dot(query_emb, emb))
        scores.append((cos, path, face_idx))
    scores.sort(key=lambda x: -x[0])
    return scores[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Face search with CrispEmbed")
    parser.add_argument("--det", default="yunet", help="Detection model (registry name or GGUF path)")
    parser.add_argument("--rec", default="auraface-v1", help="Recognition model (registry name or GGUF path)")
    parser.add_argument("--index-dir", required=True, help="Directory of images to index")
    parser.add_argument("--query", required=True, help="Query image path")
    parser.add_argument("--top-n", type=int, default=5, help="Number of results")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--lib", default=os.environ.get("CRISPEMBED_LIB"), help="Path to shared library")
    args = parser.parse_args()

    # Resolve models — if they look like registry names, auto-download
    det_path = args.det
    rec_path = args.rec
    if not os.path.isfile(det_path):
        from crispembed import CrispEmbed
        det_path = CrispEmbed.resolve_model(det_path, auto_download=True, lib_path=args.lib)
    if not os.path.isfile(rec_path):
        from crispembed import CrispEmbed
        rec_path = CrispEmbed.resolve_model(rec_path, auto_download=True, lib_path=args.lib)

    pipe = CrispFacePipeline(det_path, rec_path, n_threads=4, lib_path=args.lib)

    # Index
    print(f"Indexing faces in {args.index_dir}...")
    t0 = time.time()
    entries = index_faces(pipe, args.index_dir, conf=args.conf)
    t_index = time.time() - t0
    print(f"Indexed {len(entries)} face(s) in {t_index:.1f}s\n")

    if not entries:
        print("No faces found in index directory.")
        return

    # Query
    print(f"Query: {args.query}")
    query_results = pipe.run(args.query, conf=args.conf)
    if not query_results:
        print("No faces detected in query image.")
        return

    query_emb = query_results[0]["embedding"]
    matches = search(entries, query_emb, top_n=args.top_n)

    print(f"\nTop-{args.top_n} matches:")
    for rank, (cos, path, face_idx) in enumerate(matches, 1):
        label = "MATCH" if cos > 0.4 else "no match"
        print(f"  {rank}. cos={cos:.4f} [{label}]  {os.path.basename(path)} face#{face_idx}")


if __name__ == "__main__":
    main()
