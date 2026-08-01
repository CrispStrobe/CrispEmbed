#!/usr/bin/env python3
"""Face verification example: compare two images (1:1).

Detects the largest face in each of two images, encodes both, and prints the
cosine similarity between the templates.

This example is deliberately 1:1 (verification: "are these two images the same
person?") and not 1:N (identification: "who, in this gallery, is this?").
Searching a gallery of faces builds a biometric identification system, which
carries obligations this repo cannot discharge on your behalf — see POLICY.md.
Extending this to 1:N is a handful of lines, and that is your decision to make
knowingly, with a legal basis, rather than by copying an example.

No accept/reject verdict is printed. A usable threshold depends on the model,
the population you run it on, and the false-match rate you can tolerate; it has
to be calibrated on your own data and documented. A hardcoded constant would be
a false sense of rigour.

Usage:
    # Registry names (auto-download)
    python examples/face_verify.py --a id_photo.jpg --b selfie.jpg

    # Explicit GGUF paths
    python examples/face_verify.py --det yunet.gguf --rec auraface-v1.gguf \
        --a id_photo.jpg --b selfie.jpg

Environment:
    CRISPEMBED_LIB                 Path to libcrispembed.{so,dylib,dll}
    CRISPEMBED_ACCEPT_BIOMETRIC=1  Acknowledge biometric processing
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from crispembed import CrispFacePipeline, accept_biometric_use

BIOMETRIC_NOTICE = """\
Face recognition produces a biometric template — special-category personal data
under GDPR Art. 9, which generally needs an Art. 9(2) basis (e.g. explicit
consent) before you process it. See POLICY.md.

Re-run with --accept-biometric or set CRISPEMBED_ACCEPT_BIOMETRIC=1 to
acknowledge.\
"""


def largest_face(pipe: CrispFacePipeline, path: str, conf: float):
    """Return the embedding of the largest detected face, or None."""
    results = pipe.run(path, conf=conf)
    if not results:
        return None
    # Largest bbox by area — the conventional choice for a portrait/ID shot.
    best = max(results, key=lambda r: r["bbox"][2] * r["bbox"][3])
    return np.asarray(best["embedding"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Face verification (1:1) with CrispEmbed")
    parser.add_argument("--det", default="yunet", help="Detection model (registry name or GGUF path)")
    parser.add_argument("--rec", default="auraface-v1", help="Recognition model (registry name or GGUF path)")
    parser.add_argument("--a", required=True, help="First image")
    parser.add_argument("--b", required=True, help="Second image")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--lib", default=os.environ.get("CRISPEMBED_LIB"), help="Path to shared library")
    parser.add_argument("--accept-biometric", action="store_true",
                        help="Acknowledge biometric processing (see POLICY.md)")
    args = parser.parse_args()

    env_ack = os.environ.get("CRISPEMBED_ACCEPT_BIOMETRIC", "0") not in ("", "0")
    if not (args.accept_biometric or env_ack):
        print(BIOMETRIC_NOTICE, file=sys.stderr)
        return 1

    # Pass the acknowledgement down to the library, which gates recognition
    # models in crispembed_face_init() regardless of how it was entered.
    accept_biometric_use(lib_path=args.lib)

    # Resolve models — if they look like registry names, auto-download
    det_path = args.det
    rec_path = args.rec
    if not os.path.isfile(det_path):
        from crispembed import CrispEmbed
        det_path = CrispEmbed.resolve_model(det_path, auto_download=True, lib_path=args.lib)
    if not os.path.isfile(rec_path):
        from crispembed import CrispEmbed
        rec_path = CrispEmbed.resolve_model(rec_path, auto_download=True, lib_path=args.lib)

    pipe = CrispFacePipeline(det_path, rec_path, n_threads=1, lib_path=args.lib)

    emb_a = largest_face(pipe, args.a, args.conf)
    if emb_a is None:
        print(f"No face detected in {args.a}", file=sys.stderr)
        return 1
    emb_b = largest_face(pipe, args.b, args.conf)
    if emb_b is None:
        print(f"No face detected in {args.b}", file=sys.stderr)
        return 1

    # Embeddings are L2-normalized, so the dot product is the cosine.
    cos = float(np.dot(emb_a, emb_b))
    print(f"{os.path.basename(args.a)} vs {os.path.basename(args.b)}: cosine {cos:.4f}")
    print("(no verdict — calibrate and document a threshold on your own data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
