#!/usr/bin/env python3
"""Convert only the authenticated German spaCy 3.8.0 model; never downloads."""
import argparse
from pathlib import Path
import sys

from spacy_bundle import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New bundle directory (must not exist)")
    args = parser.parse_args()
    try:
        manifest = convert(args.wheel, args.output)
    except Exception as error:
        print(f"spaCy conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(manifest['tensors'])} exact F32 tensors; "
          f"GGUF SHA-256 {manifest['files']['model.gguf']['sha256']}")
    print("Conversion is not native linguistic-analysis qualification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
