#!/usr/bin/env python3
"""BidirLM-Omni image ↔ text retrieval demo.

Encodes an image and a set of candidate captions independently and ranks
them by cosine similarity. Both modalities land in the same 2048-d
output space (vision tower mean-pools merged tokens; text encoder
mean-pools content tokens — both L2-normalize).

Usage:
    python examples/demo/bidirlm_image_text.py [path/to/image.jpg]

Requires:
    - A vision-enabled GGUF (re-converted with the patch_embed flatten
      and visual.* export — see commit feat(vision): vision tower
      forward + parity).
    - transformers + torchvision + Pillow (HF Qwen2VLImageProcessorFast).

Caveat: this is *unconditioned* cross-modal retrieval. BidirLM-Omni's
trained behaviour is text-conditioned via DeepStack injection — running
the text encoder with vision features added at the first few layers is
the proper alignment path. The simpler mean-pool-both-then-cosine
shortcut here is mostly a sanity check that the two modalities land in
the same space at all; do not extrapolate retrieval-quality numbers
from it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from crispembed._binding import CrispEmbed


DEFAULT_MODEL = "bidirlm-omni-2.5b"
DEFAULT_IMAGE = "/tmp/cat.jpg"
DEFAULT_CAPTIONS = [
    "A photograph of a cat sitting on a chair.",
    "A car driving on a highway at sunset.",
    "Engineers reviewing a circuit diagram.",
    "A bowl of ramen on a wooden table.",
    "Schematic of an electrical wiring panel.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                    help=f"Image path (default: {DEFAULT_IMAGE})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"GGUF model identifier or path (default: {DEFAULT_MODEL})")
    ap.add_argument("--lib", default=None,
                    help="libcrispembed shared-library path (optional)")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Loading {args.model}…", flush=True)
    ce = CrispEmbed(args.model, lib_path=args.lib) if args.lib else CrispEmbed(args.model)
    if not ce.has_vision:
        print("ERROR: build has no vision support compiled in.", file=sys.stderr)
        return 1

    print(f"Encoding image: {args.image}")
    img_vec = ce.encode_image(args.image)
    if img_vec.size == 0:
        print("ERROR: vision tower returned no data — is this GGUF vision-enabled?",
              file=sys.stderr)
        return 1

    print(f"Encoding {len(DEFAULT_CAPTIONS)} candidate captions…")
    txt_vecs = np.stack([ce.encode(t) for t in DEFAULT_CAPTIONS])

    sims = txt_vecs @ img_vec
    order = np.argsort(-sims)

    print()
    print(f"{'rank':>4s}  {'cosine':>8s}  caption")
    print("-" * 70)
    for r, idx in enumerate(order, 1):
        print(f"{r:>4d}  {sims[idx]:>8.4f}  {DEFAULT_CAPTIONS[idx]}")
    print()
    print("(Mean-pool image_embeds vs encode(text). Unconditioned — see file docstring.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
