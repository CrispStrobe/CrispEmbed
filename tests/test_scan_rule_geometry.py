#!/usr/bin/env python3
"""Independent projective-grid oracle for native ruled-photo rectification.

OpenCV creates known camera homographies and measures output lines. It is a
test-only dependency; the production detector and warp use only C++ stdlib.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--library', required=True)
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args()
args.artifacts.mkdir(parents=True, exist_ok=True)
lib = C.CDLL(str(Path(args.library).resolve()))
FloatPointer = C.POINTER(C.c_float)
fn = lib.scan_cleanup_rectify_rules
fn.argtypes = [FloatPointer, C.c_int, C.c_int, C.POINTER(FloatPointer), C.POINTER(C.c_int), C.POINTER(C.c_int)]
fn.restype = C.c_int
free = lib.scan_cleanup_free_image
free.argtypes = [C.c_void_p]

def rectify(image):
    gray = np.ascontiguousarray(image.astype(np.float32) / 255)
    out = FloatPointer()
    width, height = C.c_int(), C.c_int()
    rc = fn(gray.ctypes.data_as(FloatPointer), image.shape[1], image.shape[0], C.byref(out), C.byref(width), C.byref(height))
    if rc != 1:
        assert not out
        return rc, image
    try:
        result = np.ctypeslib.as_array(out, shape=(width.value * height.value,)).copy().reshape(height.value, width.value)
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
        return rc, np.rint(result * 255).astype(np.uint8)
    finally:
        free(out)

def angles(image):
    height, width = image.shape
    detected = cv2.HoughLinesP(cv2.Canny(image, 30, 100), 1, np.pi / 1800,
                              threshold=60, minLineLength=min(width, height) * .35, maxLineGap=12)
    horizontal, vertical = [], []
    if detected is not None:
        for x1, y1, x2, y2 in detected[:, 0, :]:
            dx, dy = int(x2 - x1), int(y2 - y1)
            if abs(dx) > width * .35 and abs(dy) < abs(dx) * .2:
                horizontal.append(abs(float(np.degrees(np.arctan2(dy, dx))) % 180))
                horizontal[-1] = min(horizontal[-1], 180 - horizontal[-1])
            if abs(dy) > height * .35 and abs(dx) < abs(dy) * .2:
                vertical.append(abs(float(np.degrees(np.arctan2(dx, dy))) % 180))
                vertical[-1] = min(vertical[-1], 180 - vertical[-1])
    assert len(horizontal) >= 6 and len(vertical) >= 4, (len(horizontal), len(vertical))
    return {'horizontal_p90': float(np.quantile(horizontal, .9)),
            'vertical_p90': float(np.quantile(vertical, .9)),
            'horizontal_lines': len(horizontal), 'vertical_lines': len(vertical)}

height, width = 1000, 800
grid = np.full((height, width), 242, np.uint8)
for y in range(90, 930, 70):
    cv2.line(grid, (70, y), (730, y), 20, 2)
for x in range(70, 750, 60):
    cv2.line(grid, (x, 90), (x, 860), 20, 2)
normal = np.array([[1/width, 0, -.5], [0, 1/height, -.5], [0, 0, 1.]])
receipt = {'production_uses_opencv': False, 'cases': [], 'controls': []}
for name, camera in [
    ('perspective-a', [[1., -.03, 0], [.025, 1., 0], [.07, -.10, 1.]]),
    ('perspective-b', [[1., .02, 0], [-.035, 1., 0], [-.05, .08, 1.]]),
    ('shear', [[1., -.04, 0], [.025, 1., 0], [0, 0, 1.]]),
]:
    matrix = np.linalg.inv(normal) @ np.array(camera) @ normal
    photo = cv2.warpPerspective(grid, matrix, (width, height), borderValue=255)
    rc, prepared = rectify(photo)
    assert rc == 1, name
    before, after = angles(photo), angles(prepared)
    cv2.imwrite(str(args.artifacts / f'{name}-input.png'), photo)
    cv2.imwrite(str(args.artifacts / f'{name}-output.png'), prepared)
    receipt['cases'].append({'case': name, 'before': before, 'after': after})
    assert after['horizontal_p90'] < .35 and after['vertical_p90'] < .35, (name, after)
for name, image in [('blank', np.full_like(grid, 240)), ('aligned', grid),
                    ('horizontal-only', np.where(np.indices(grid.shape)[0] % 70 < 2, 20, 240).astype(np.uint8))]:
    rc, prepared = rectify(image)
    assert rc == 0 and np.array_equal(prepared, image), name
    receipt['controls'].append({'case': name, 'no_warp': True})
(args.artifacts / 'geometry.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
