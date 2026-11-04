#!/usr/bin/env python3
"""Bounded original PP-DocLayoutV3 stage for Xiaomi document qualification.

This is an external hosted reference dependency, not a native layout alias.
It invokes the pinned upstream detector/postprocessor and preserves its region
order. The native Xiaomi HF route remains independently qualified from vLLM.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("xiaomi0_layout_policy", ROOT / "models/qwen35_xiaomi0_profile.py")
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)
# Additional transitive imports from the same audited upstream Git tree. These
# are canonical Git blob IDs, checked from the exact bytes before import.
EXTRA_BLOBS = {"postprocess/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
               "evaluation_pipeline/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
               "postprocess/repetition_guard.py": "4b85e5cf5eba357141eaed23efc96e4743aa107b",
               "postprocess/otsl.py": "1acbe769397246f9d26ee21b1122e2136d240c1f"}
REQUIRED_SOURCE = ("evaluation_pipeline/twostage_img2md.py", "evaluation_pipeline/cropping.py", "postprocess/region.py")


def verify_layout_assets(directory):
    layout = policy.evidence()["layout"]
    root = Path(directory)
    for name, digest in layout["metadata_files"].items():
        if policy.file_sha256(root / name) != digest:
            raise ValueError("Actual PP-DocLayoutV3 metadata differs: " + name)
    for name, record in layout["checkpoint_files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or policy.file_sha256(path) != record["sha256"]:
            raise ValueError("Actual PP-DocLayoutV3 checkpoint differs: " + name)
    return layout


def load_upstream(directory):
    root = Path(directory).resolve()
    expected = policy.evidence()["upstream"]["files"]
    for name in REQUIRED_SOURCE:
        if policy.file_sha256(root / name) != expected[name]:
            raise ValueError("Xiaomi original layout source differs: " + name)
    for name, digest in EXTRA_BLOBS.items():
        data = (root / name).read_bytes()
        if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != digest:
            raise ValueError("Xiaomi transitive source differs: " + name)
    for name in ("postprocess", "postprocess.otsl", "postprocess.region", "postprocess.repetition_guard",
                 "evaluation_pipeline", "evaluation_pipeline.cropping"):
        if name in sys.modules:
            loaded = Path(sys.modules[name].__file__).resolve()
            if loaded != root / (name.replace(".", "/") + ("/__init__.py" if name in ("postprocess", "evaluation_pipeline") else ".py")):
                raise ValueError("Unrelated module shadows pinned Xiaomi source: " + name)
    spec = importlib.util.spec_from_file_location("xiaomi0_original_twostage", root / REQUIRED_SOURCE[0])
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    return module


def region_requests(regions, detector, *, maximum=16):
    """Validate actual detector identity and map original labels to exact tasks."""
    evidence = policy.evidence()
    layout = evidence["layout"]
    if (detector.get("model") != layout["model"] or detector.get("revision") != layout["revision"]
            or detector.get("checkpoint_files") != layout["checkpoint_files"]
            or detector.get("threshold") != layout["transformers_threshold"]
            or detector.get("forward_executed") is not True
            or detector.get("source_revision") != evidence["upstream"]["revision"]):
        raise ValueError("Two-stage Xiaomi requires the actual pinned PP-DocLayoutV3 producer")
    if type(maximum) is not int or maximum < 1 or maximum > 64 or len(regions) > maximum:
        raise ValueError("Bounded layout region budget exceeded; no silent truncation")
    mapping = {label: task for task, labels in layout["label_task_mapping"].items() for label in labels}
    result = []
    for index, region in enumerate(regions):
        label = region.get("label")
        bbox = region.get("bbox_2d")
        if (region.get("index") != index or label not in mapping or region.get("task_type") != mapping[label]
                or not isinstance(bbox, list) or len(bbox) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000 for value in bbox)
                or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            raise ValueError("Invalid original layout label/order/coordinates")
        task = mapping[label]
        # Preserve visual regions as explicit skipped entries; do not fabricate OCR.
        result.append({"index": index, "label": label, "task": task,
                       "bbox": [value / 1000 for value in bbox], "bbox_2d": bbox,
                       "prompt": evidence["prompts"].get(task), "ocr_required": task != "skip",
                       "native_prompt_order": "image-first", "original_vllm_prompt_order": "text-first"})
    return result


def detect(image_path, layout_dir, upstream_dir, *, maximum=16):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise ValueError("Actual layout weights/inference run on hosted CI only")
    import transformers
    from PIL import Image
    if transformers.__version__ != "5.17.0":
        raise ValueError("Pinned PP-DocLayoutV3 Transformers blueprint required")
    layout = verify_layout_assets(layout_dir)
    upstream = load_upstream(upstream_dir)
    image = Image.open(image_path).convert("RGB")
    processor, model = upstream.load_layout_model(device="cpu", model_dir=str(layout_dir))
    if type(model).__name__ != "PPDocLayoutV3ForObjectDetection" or type(processor).__name__ != "PPDocLayoutV3ImageProcessor":
        raise ValueError("Substitute layout detector is not the Xiaomi dependency")
    regions = upstream.run_layout_detection([image], processor, model, device="cpu", threshold=0.3)[0]
    detector = {"model": layout["model"], "revision": layout["revision"],
                "checkpoint_files": layout["checkpoint_files"], "threshold": 0.3, "forward_executed": True,
                "source_revision": policy.evidence()["upstream"]["revision"],
                "image_sha256": policy.file_sha256(image_path)}
    requests = region_requests(regions, detector, maximum=maximum)
    return {"schema": "xiaomi0-original-layout/v1", "detector": detector, "regions": regions,
            "requests": requests, "native_layout_implemented": False, "document_qualified": False}, upstream, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layout-model-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    args = parser.parse_args()
    report, upstream, image = detect(args.image, args.layout_model_dir, args.upstream_dir, maximum=args.max_regions)
    args.output.mkdir(parents=True, exist_ok=True)
    for request in report["requests"]:
        if not request["ocr_required"]:
            continue
        crop = upstream.crop_region(image, request["bbox_2d"])
        # Keep the original crop; native bound rejection is evidence, not an implicit resize.
        file = args.output / ("region-%03d.png" % request["index"])
        crop.save(file)
        request["crop_file"] = file.name
        request["crop_sha256"] = policy.file_sha256(file)
    (args.output / "layout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
