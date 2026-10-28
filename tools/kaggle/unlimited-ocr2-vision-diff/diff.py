"""Unlimited-OCR-2 native-vs-reference vision diff (stdlib + numpy + Pillow; run on Kaggle).

    diff.py <page.png> <native vis.f32> <native global-pixels.f32> <reference projector.npy> <out.json>

1. Preprocessing: recomputes the reference views exactly like `_uocr_build_inputs`
   (dynamic_preprocess tiles, ImageOps.pad global, ToTensor/Normalize(0.5), bf16) and
   compares them with the native dumps (UOCR_DUMP_PIXELS, <vis>.tiles.f32).
2. Features: compares the native assembled IMG span (UOCR2_DUMP_VISION) block by block
   with the reference projector array. The reference hook keeps the FIRST projector call
   of a case: local tiles [wt*ht, 100, D] when tiled, the global view [1, 256, D] when
   untiled. Newline/separator rows and a tiled page's global view are not in the
   reference; they get native-internal consistency checks only.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps


def bf16(a):
    b = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def to_tensor(img):  # ToTensor + Normalize(0.5, 0.5) + .to(bfloat16)
    x = np.asarray(img.convert('RGB'), np.float32).transpose(2, 0, 1) / np.float32(255)
    return bf16((x - np.float32(0.5)) / np.float32(0.5))


def grid(w, h, min_num=2, max_num=16, isz=640):  # dynamic_preprocess + find_closest_aspect_ratio
    tr = sorted(set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                    if min_num <= i * j <= max_num), key=lambda x: x[0] * x[1])
    ar, best, bd = w / h, (1, 1), float('inf')
    for r in tr:
        d = abs(ar - r[0] / r[1])
        if d < bd:
            bd, best = d, r
        elif d == bd and w * h > 0.5 * isz * isz * r[0] * r[1]:
            best = r
    return best


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return dict(max_abs=float(np.max(np.abs(a - b))), mean_abs=float(np.mean(np.abs(a - b))), cosine=cos,
                ref_max_abs=float(np.max(np.abs(b))), exact=bool(np.array_equal(a, b)))


def main(img_path, vis_path, gpx_path, proj_path, out):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    tiled = w > 640 or h > 640
    rep = dict(size=[w, h], tiled=tiled)
    g_ref = to_tensor(ImageOps.pad(img, (1024, 1024), color=(127, 127, 127)))
    if os.path.exists(gpx_path):
        g_nat = np.fromfile(gpx_path, np.float32).reshape(3, 1024, 1024)
        rep['global_pixels'] = stats(g_nat, g_ref) | dict(n_diff=int(np.sum(g_nat != g_ref)))
    wt = ht = 1
    if tiled:
        wt, ht = grid(w, h)
        big = img.resize((640 * wt, 640 * ht))
        tiles = [to_tensor(big.crop(((i % wt) * 640, (i // wt) * 640, (i % wt + 1) * 640, (i // wt + 1) * 640)))
                 for i in range(wt * ht)]
        tp = vis_path + '.tiles.f32'
        if os.path.exists(tp):
            nat = np.fromfile(tp, np.float32).reshape(wt * ht, 3, 640, 640)
            rows = [stats(nat[i], tiles[i]) | dict(tile=i, n_diff=int(np.sum(nat[i] != tiles[i])))
                    for i in range(wt * ht)]
            rep['tile_pixels'] = dict(first_differing_tile=next((r['tile'] for r in rows if not r['exact']), None),
                                      max_abs=max(r['max_abs'] for r in rows), per_tile=rows)
    rep['grid'] = [wt, ht]
    proj = np.load(proj_path).astype(np.float32)
    D = proj.shape[-1]
    vis = np.fromfile(vis_path, np.float32).reshape(-1, D)
    rep['native_n_vis'] = int(vis.shape[0])
    rep['reference_projector_shape'] = list(proj.shape)
    nq, nqb = 10, 16
    n_local = (nq * ht) * (nq * wt + 1) if tiled else 0
    rep['n_vis_expected'] = n_local + nqb * (nqb + 1) + 1
    rep['n_vis_match'] = rep['native_n_vis'] == rep['n_vis_expected']
    if not rep['n_vis_match']:
        # Reference layout cannot be applied to a dump with a different token count
        # (e.g. 256 instead of 100 tokens per 640 tile): report it, do not slice garbage.
        rep['error'] = 'native n_vis %d != reference %d' % (rep['native_n_vis'], rep['n_vis_expected'])
        with open(out, 'w') as f:
            json.dump(rep, f, indent=1)
        sys.exit(3)
    blocks = []
    if tiled:
        loc = vis[:n_local].reshape(nq * ht, nq * wt + 1, D)
        if proj.shape[0] == wt * ht:  # reference = local tiles, row-major
            for i in range(wt * ht):
                ty, tx = divmod(i, wt)
                nb = loc[ty * nq:(ty + 1) * nq, tx * nq:(tx + 1) * nq].reshape(nq * nq, D)
                rb = proj[i].reshape(-1, D)
                worst = int(np.argmax(np.max(np.abs(nb - rb), axis=1)))
                blocks.append(dict(block='tile', tile=i, ty=ty, tx=tx, **stats(nb, rb), worst_token=worst,
                                   worst_token_rc=list(divmod(worst, nq))))
        nl = loc[:, nq * wt]
        rep['local_newlines_identical'] = bool(np.all(nl == nl[0]))
    glob_ = vis[n_local:n_local + nqb * (nqb + 1)].reshape(nqb, nqb + 1, D)
    if not tiled and proj.shape[-2] == nqb * nqb:
        blocks.append(dict(block='global', **stats(glob_[:, :nqb].reshape(nqb * nqb, D), proj.reshape(-1, D))))
    rep['global_newlines_identical'] = bool(np.all(glob_[:, nqb] == glob_[0, nqb]))
    if tiled:
        rep['global_newline_equals_local_newline'] = bool(np.array_equal(glob_[0, nqb], loc[0, nq * wt]))
    rep['blocks'] = blocks
    rep['first_block_over_1e-3'] = next((i for i, b in enumerate(blocks) if b['max_abs'] > 1e-3), None)
    rep['max_abs'] = max((b['max_abs'] for b in blocks), default=None)
    rep['min_cosine'] = min((b['cosine'] for b in blocks), default=None)
    with open(out, 'w') as f:
        json.dump(rep, f, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:6])
