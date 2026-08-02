#!/usr/bin/env python3
"""Run the checked-out EasyOCR Python page pipeline and emit a manifest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyocr-repo", type=Path, required=True)
    parser.add_argument("--craft-checkpoint", type=Path, required=True)
    parser.add_argument("--recognizer-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.easyocr_repo))
    from PIL import Image

    from easyocr_postprocess_reference import make_manifest

    import easyocr

    # Reader validates the official MD5 values from EasyOCR's config. Symlink
    # only the requested artifacts into an isolated model directory so this
    # runner cannot accidentally use a user-global ~/.EasyOCR cache.
    with tempfile.TemporaryDirectory(prefix="crispembed-easyocr-models-") as model_dir:
        model_dir = Path(model_dir)
        (model_dir / "craft_mlt_25k.pth").symlink_to(args.craft_checkpoint.resolve())
        (model_dir / "english_g2.pth").symlink_to(args.recognizer_checkpoint.resolve())
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(model_dir),
            download_enabled=False,
            detector=True,
            recognizer=True,
            verbose=False,
            quantize=False,
        )
        result = reader.readtext(
            str(args.image),
            decoder="greedy",
            batch_size=1,
            workers=0,
            detail=1,
            paragraph=False,
        )

    with Image.open(args.image) as image:
        width, height = image.size
    source = {
        "image": str(args.image),
        "width": width,
        "height": height,
        "items": [
            {
                "box": item[0],
                "text": item[1],
                "confidence": item[2],
                # EasyOCR's public readtext tuple does not expose detector
                # confidence. Keep this explicit for downstream comparison.
                "detector_confidence": 0.0,
            }
            for item in result
        ],
        "detector_confidence_source": "unavailable in EasyOCR readtext detail=1",
    }
    manifest = make_manifest({**source, "mode": "lines"})
    manifest["detector_confidence_source"] = source["detector_confidence_source"]
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}; mode=lines; records={len(manifest['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
