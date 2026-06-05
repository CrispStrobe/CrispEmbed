#!/usr/bin/env python3
"""YuNet face detection parity test: CrispEmbed C++ vs OpenCV reference.

Verifies that the GGUF graph replay + YuNet decode produces detections
matching OpenCV's FaceDetectorYN on the same ONNX model.

Usage:
    python tests/test_yunet_parity.py [--onnx PATH] [--gguf PATH] [--binary PATH]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def make_test_image(path: str, w: int = 640, h: int = 480) -> np.ndarray:
    """Create a synthetic test image with a face-like pattern."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    # Face oval
    cv2.ellipse(img, (cx, cy - 40), (80, 100), 0, 0, 360, (220, 190, 170), -1)
    # Eyes
    cv2.circle(img, (cx - 30, cy - 60), 8, (60, 60, 60), -1)
    cv2.circle(img, (cx + 30, cy - 60), 8, (60, 60, 60), -1)
    # Nose
    cv2.circle(img, (cx, cy - 30), 5, (180, 150, 140), -1)
    # Mouth
    cv2.ellipse(img, (cx, cy), (25, 8), 0, 0, 360, (150, 100, 100), -1)
    cv2.imwrite(path, img)
    return img


def opencv_detect(onnx_path: str, img: np.ndarray, conf: float = 0.3):
    """Run OpenCV FaceDetectorYN and return list of (x,y,w,h,score,landmarks)."""
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(onnx_path, "", (w, h),
                                     score_threshold=conf, nms_threshold=0.4)
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None:
        return []
    results = []
    for f in faces:
        results.append({
            "x": float(f[0]), "y": float(f[1]),
            "w": float(f[2]), "h": float(f[3]),
            "score": float(f[14]),
            "landmarks": [float(f[4+i]) for i in range(10)],
        })
    return results


def crispembed_detect(binary: str, gguf_path: str, img_path: str, conf: float = 0.3):
    """Run CrispEmbed CLI detection and parse JSON output."""
    cmd = [binary, "-m", gguf_path, "--detect", img_path, "--json",
           "--conf", str(conf)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"CrispEmbed stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"CrispEmbed failed with code {result.returncode}")
    data = json.loads(result.stdout)
    faces = []
    for f in data.get("faces", []):
        faces.append({
            "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"],
            "score": f["conf"],
            "landmarks": f["landmarks"],
        })
    return faces


def compute_iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / (union + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="/mnt/volume1/yunet/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--gguf", default="/mnt/storage/yunet.gguf")
    parser.add_argument("--binary", default="build/crispembed")
    args = parser.parse_args()

    # Check files exist
    for p, name in [(args.onnx, "ONNX"), (args.gguf, "GGUF"), (args.binary, "binary")]:
        if not os.path.exists(p):
            print(f"SKIP: {name} not found at {p}")
            return 0

    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Single face
        img_path = os.path.join(tmpdir, "single.jpg")
        img = make_test_image(img_path)
        cv_faces = opencv_detect(args.onnx, img)
        ce_faces = crispembed_detect(args.binary, args.gguf, img_path)

        print(f"Test 1 - Single face: OpenCV={len(cv_faces)} CrispEmbed={len(ce_faces)}")
        if len(cv_faces) == len(ce_faces) and len(cv_faces) > 0:
            cv_faces.sort(key=lambda f: -f["score"])
            ce_faces.sort(key=lambda f: -f["score"])
            for i, (cv, ce) in enumerate(zip(cv_faces, ce_faces)):
                iou = compute_iou(cv, ce)
                score_diff = abs(cv["score"] - ce["score"])
                lm_diffs = [abs(cv["landmarks"][k] - ce["landmarks"][k])
                            for k in range(10)]
                max_lm_diff = max(lm_diffs)
                print(f"  Face {i}: IoU={iou:.4f} score_diff={score_diff:.4f} "
                      f"max_landmark_diff={max_lm_diff:.1f}px")
                if iou > 0.85 and score_diff < 0.01 and max_lm_diff < 5.0:
                    passed += 1
                    print("    PASS")
                else:
                    failed += 1
                    print(f"    FAIL (IoU>{0.85}, score_diff<{0.01}, lm<{5.0}px)")
        else:
            failed += 1
            print("  FAIL: face count mismatch")

        # Test 2: Multi-face
        img2_path = os.path.join(tmpdir, "multi.jpg")
        img2 = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Two faces
        cv2.ellipse(img2, (200, 200), (60, 80), 0, 0, 360, (220, 190, 170), -1)
        cv2.circle(img2, (175, 180), 6, (60, 60, 60), -1)
        cv2.circle(img2, (225, 180), 6, (60, 60, 60), -1)
        cv2.circle(img2, (200, 210), 4, (180, 150, 140), -1)
        cv2.ellipse(img2, (200, 230), (20, 6), 0, 0, 360, (150, 100, 100), -1)
        cv2.ellipse(img2, (440, 220), (55, 75), 0, 0, 360, (210, 180, 160), -1)
        cv2.circle(img2, (415, 200), 6, (50, 50, 50), -1)
        cv2.circle(img2, (465, 200), 6, (50, 50, 50), -1)
        cv2.circle(img2, (440, 225), 4, (170, 140, 130), -1)
        cv2.ellipse(img2, (440, 250), (18, 5), 0, 0, 360, (140, 90, 90), -1)
        cv2.imwrite(img2_path, img2)

        cv_faces2 = opencv_detect(args.onnx, img2)
        ce_faces2 = crispembed_detect(args.binary, args.gguf, img2_path)

        print(f"\nTest 2 - Multi-face: OpenCV={len(cv_faces2)} CrispEmbed={len(ce_faces2)}")
        if len(cv_faces2) == len(ce_faces2):
            passed += 1
            print("    PASS (count match)")
        else:
            failed += 1
            print("    FAIL (count mismatch)")

        # Test 3: No face (landscape)
        img3_path = os.path.join(tmpdir, "noface.jpg")
        img3 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        # Add some edges/textures but no face-like pattern
        for y in range(0, 480, 40):
            cv2.line(img3, (0, y), (640, y), (50, 50, 50), 2)
        cv2.imwrite(img3_path, img3)

        cv_faces3 = opencv_detect(args.onnx, img3, conf=0.5)
        ce_faces3 = crispembed_detect(args.binary, args.gguf, img3_path, conf=0.5)

        print(f"\nTest 3 - No face: OpenCV={len(cv_faces3)} CrispEmbed={len(ce_faces3)}")
        # Both should detect 0 or very few spurious faces
        if abs(len(cv_faces3) - len(ce_faces3)) <= 1:
            passed += 1
            print("    PASS")
        else:
            failed += 1
            print("    FAIL")

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
