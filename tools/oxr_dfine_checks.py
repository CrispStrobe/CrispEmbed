#!/usr/bin/env python3
"""Hosted-CI helpers for the OXR D-FINE layout qualification lane.

Subcommands
  refdiff A.gguf B.gguf      stage-by-stage drift between two *reference* runs
                             (e.g. torch threads 4 vs 1): the PyTorch noise floor.
  api --lib L --model M --image P --reference R.json
                             native C API JSON vs the service response: exact
                             key/format contract (compact JSON, Python float repr),
                             OXR parse_layout_batch blocks on both sides.
  heron --lib L --model M --images ...  --output O.json
                             rt_detr_v2 (heron) regions through crispembed_layout_detect
                             with repr-exact floats, for base-vs-branch identity.
  lineage --checkpoint last.pth --output O.json
                             compare backbone/full tensors with public D-FINE /
                             HGNetv2 checkpoints (Peterande releases, Apache-2.0).
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import sys
from pathlib import Path


def refdiff(args):
    import gguf
    import numpy as np
    a = {t.name: np.array(t.data, dtype=np.float64) for t in gguf.GGUFReader(args.a).tensors}
    b = {t.name: np.array(t.data, dtype=np.float64) for t in gguf.GGUFReader(args.b).tensors}
    rows = []
    for name in sorted(set(a) & set(b)):
        x, y = a[name].reshape(-1), b[name].reshape(-1)
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[ok] - y[ok])
        ref = np.abs(y[ok]).max() if ok.any() else 0.0
        rows.append({"name": name, "max_abs": float(d.max()) if d.size else 0.0,
                     "rel": float(d.max() / ref) if d.size and ref > 0 else 0.0,
                     "nonfinite_equal": bool(np.array_equal(~np.isfinite(x), ~np.isfinite(y)))})
    Path(args.output).write_text(json.dumps({"schema": "crispembed.oxr-dfine.refdiff.v1", "a": Path(args.a).name,
                                             "b": Path(args.b).name, "stages": rows}, indent=1) + "\n")


def api(args):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from crispembed._oxr_layout import CrispOXRLayout
    reference = json.loads(Path(args.reference).read_text())["response"]
    rgb = np.ascontiguousarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    lib = C.CDLL(args.lib)
    lib.crispembed_oxr_layout_init.restype = C.c_void_p
    lib.crispembed_oxr_layout_init.argtypes = [C.c_char_p, C.c_int]
    lib.crispembed_oxr_layout_detect.restype = C.c_char_p
    lib.crispembed_oxr_layout_detect.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int, C.c_int,
                                                 C.c_float]
    ctx = lib.crispembed_oxr_layout_init(args.model.encode(), 4)
    if not ctx:
        raise SystemExit("crispembed_oxr_layout_init failed")
    h, w = rgb.shape[:2]
    raw = lib.crispembed_oxr_layout_detect(ctx, rgb.ctypes.data_as(C.POINTER(C.c_uint8)), w, h, 3, float("nan"))
    text = raw.decode()
    native = json.loads(text)
    raw_c = lib.crispembed_oxr_layout_detect(ctx, rgb.ctypes.data_as(C.POINTER(C.c_uint8)), w, h, 3, 0.5)
    native_c = json.loads(raw_c.decode())
    checks = {
        # Starlette JSONResponse: json.dumps(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        "compact_python_json_text": text == json.dumps(native, ensure_ascii=False, allow_nan=False,
                                                        separators=(",", ":")),
        "top_level_keys": list(native) == list(reference),
        "data_keys": list(native["data"]) == list(reference["data"]),
        "params_empty": native["params"] == reference["params"] == {},
        "page_count": all(len(native["data"][k]) == 1 for k in ("labels", "boxes", "scores")),
        "candidate_count": len(native["data"]["labels"][0]) == len(reference["data"]["labels"][0]),
        "label_types": all(type(v) is int for v in native["data"]["labels"][0]),
        "box_arity": all(len(b) == 4 for b in native["data"]["boxes"][0]),
        "confidence_params": native_c["params"] == {"confidence": 0.5, "class_confidences": {}},
        "confidence_filter": all(s >= 0.5 for s in native_c["data"]["scores"][0]),
    }
    # The Python wrapper and OXR's own parser on both responses.
    with CrispOXRLayout(args.model, 4, args.lib) as model:
        wrapped = model.detect(rgb)
    checks["python_wrapper_equal"] = wrapped == native
    parsed = {}
    if args.blueprint:
        sys.path.insert(0, str(Path(args.blueprint)))
        from oxr.pipeline.postprocess.layout_postprocess import parse_layout_batch
        for side, payload in (("native", native), ("reference", reference)):
            blocks = parse_layout_batch(payload, [(w, h)], 0.5, 0.6)[0]
            parsed[side] = [{"type": str(b.type.value if hasattr(b.type, "value") else b.type),
                             "bbox": b.bbox, "score": b.layout_score} for b in blocks]
        same = len(parsed["native"]) == len(parsed["reference"]) and all(
            n["type"] == r["type"] and max(abs(x - y) for x, y in zip(n["bbox"], r["bbox"])) <= 0.5
            for n, r in zip(parsed["native"], parsed["reference"]))
        checks["oxr_parse_layout_batch_equivalent"] = same
    result = {"schema": "crispembed.oxr-dfine.api.v1", "image": Path(args.image).name, "checks": checks,
              "pass": all(checks.values()), "parsed": parsed}
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(checks))
    return 0 if result["pass"] else 1


def heron(args):
    class Region(C.Structure):
        _fields_ = [("x1", C.c_float), ("y1", C.c_float), ("x2", C.c_float), ("y2", C.c_float),
                    ("score", C.c_float), ("label", C.c_int), ("label_name", C.c_char_p)]
    lib = C.CDLL(args.lib)
    lib.crispembed_layout_init.restype = C.c_void_p
    lib.crispembed_layout_init.argtypes = [C.c_char_p, C.c_int]
    lib.crispembed_layout_detect.restype = C.POINTER(Region)
    lib.crispembed_layout_detect.argtypes = [C.c_void_p, C.c_char_p, C.c_float, C.POINTER(C.c_int)]
    ctx = lib.crispembed_layout_init(args.model.encode(), 4)
    if not ctx:
        raise SystemExit("heron init failed")
    out = {}
    for image in args.images:
        n = C.c_int(0)
        regions = lib.crispembed_layout_detect(ctx, image.encode(), 0.0, C.byref(n))
        out[Path(image).name] = [[repr(regions[i].x1), repr(regions[i].y1), repr(regions[i].x2),
                                  repr(regions[i].y2), repr(regions[i].score), regions[i].label,
                                  regions[i].label_name.decode()] for i in range(n.value)]
    text = json.dumps(out, indent=0, sort_keys=True)
    Path(args.output).write_text(text + "\n")
    print(Path(args.output).name, hashlib.sha256(text.encode()).hexdigest(), {k: len(v) for k, v in out.items()})


PUBLIC = {
    "PPHGNetV2_B2_stage1.pth": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
    "dfine_m_obj365.pth": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth",
    "dfine_m_coco.pth": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
    "dfine_m_obj2coco.pth": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco_e25.pth",
}


def lineage(args):
    import urllib.request
    import torch
    oxr = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed.oxr-dfine.lineage.v1", "oxr_tensors": len(oxr), "candidates": {}}
    for name, url in PUBLIC.items():
        path = work / name
        entry = {"url": url}
        try:
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            obj = torch.load(path, map_location="cpu", weights_only=True)
            state = obj.get("ema", {}).get("module") if isinstance(obj, dict) and "ema" in obj else None
            state = state or (obj.get("model") if isinstance(obj, dict) and "model" in obj else obj)
            prefix = "" if any(k.startswith("backbone.") for k in state) else "backbone."
            groups = {"backbone": 0, "encoder": 0, "decoder": 0}
            equal = {"backbone": 0, "encoder": 0, "decoder": 0}
            cos = {"backbone": [], "encoder": [], "decoder": []}
            for key, tensor in oxr.items():
                group = key.split(".")[0]
                if group not in groups or not tensor.is_floating_point():
                    continue
                other = state.get(key if prefix == "" else key[len("backbone."):] if key.startswith("backbone.") else None)
                if other is None or other.shape != tensor.shape:
                    continue
                groups[group] += 1
                if torch.equal(other.float(), tensor.float()):
                    equal[group] += 1
                a, b = tensor.float().flatten(), other.float().flatten()
                if a.norm() > 0 and b.norm() > 0:
                    cos[group].append(float(torch.dot(a, b) / (a.norm() * b.norm())))
            entry.update(shape_matched=groups, bit_equal=equal,
                         mean_cosine={g: (sum(v) / len(v) if v else None) for g, v in cos.items()})
        except Exception as error:  # record download/format failures verbatim
            entry["error"] = f"{type(error).__name__}: {error}"
        report["candidates"][name] = entry
    backbone = sum(1 for k, t in oxr.items() if k.startswith("backbone.") and t.is_floating_point())
    report["oxr_backbone_float_tensors"] = backbone
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: {x: v.get(x) for x in ("bit_equal", "shape_matched", "error")}
                      for k, v in report["candidates"].items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refdiff"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--output", required=True)
    p = sub.add_parser("api")
    for k in ("--lib", "--model", "--image", "--reference", "--output"):
        p.add_argument(k, required=True)
    p.add_argument("--blueprint")
    p = sub.add_parser("heron")
    p.add_argument("--lib", required=True); p.add_argument("--model", required=True)
    p.add_argument("--output", required=True); p.add_argument("--images", nargs="+", required=True)
    p = sub.add_parser("lineage")
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--workdir", required=True)
    args = ap.parse_args()
    return {"refdiff": refdiff, "api": api, "heron": heron, "lineage": lineage}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
