#!/usr/bin/env python3
"""LFM2.5-VL tiling oracle — HF's own functions, extracted verbatim.

Multi-tile NaFlex is not implemented in the C++ engine yet (see
/mnt/volume1/naflex-todos.md). This script exists so the guard test can be
written BEFORE that code, per dev-guide HARD RULE 2c: it produces the golden
grid layouts to pin against.

These are VERBATIM extracts from
transformers/models/lfm2_vl/image_processing_lfm2_vl.py -- pure math, no torch
or torchvision needed, which is what makes them a real oracle rather than a
reimplementation of the thing under test. Do not "clean them up": the value is
that they are byte-for-byte what upstream runs. In particular Python's round()
is banker's rounding (half-to-even) and C++ std::round is not -- that
difference is the single most likely silent divergence.

Usage:  python tools/lfm2_vl_tiling_oracle.py
"""
import math

def round_by_factor(number, factor):
    return round(number / factor) * factor

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf"); best_ratio = (1, 1); area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff; best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio

def target_ratios(min_tiles, max_tiles):
    r = [(w, h) for n in range(min_tiles, max_tiles + 1)
                for w in range(1, n + 1) for h in range(1, n + 1)
                if min_tiles <= w * h <= max_tiles]
    return sorted(set(r), key=lambda x: x[0] * x[1])

def get_grid_layout(height, width, min_tiles, max_tiles, tile_size):
    ar = width / height
    gw, gh = find_closest_aspect_ratio(ar, target_ratios(min_tiles, max_tiles), width, height, tile_size)
    return gw, gh, tile_size * gw, tile_size * gh, gw * gh

def smart_resize(height, width, ds, min_tok, max_tok, P):
    tf = P * ds
    mn = min_tok * P**2 * ds**2; mx = max_tok * P**2 * ds**2
    h = max(tf, round_by_factor(height, tf)); w = max(tf, round_by_factor(width, tf))
    if h * w > mx:
        b = math.sqrt((height * width) / mx)
        h = max(tf, math.floor(height / b / tf) * tf); w = max(tf, math.floor(width / b / tf) * tf)
    elif h * w < mn:
        b = math.sqrt(mn / (height * width))
        h = math.ceil(height * b / tf) * tf; w = math.ceil(width * b / tf) * tf
    return w, h

def is_too_large(height, width, max_tok, P, ds, tol):
    tf = P * ds
    h = max(P, round_by_factor(height, tf)); w = max(P, round_by_factor(width, tf))
    return h * w > max_tok * P**2 * ds**2 * tol

P, DS, TILE = 16, 2, 512
MINT, MAXT, MINTOK, MAXTOK, TOL = 1, 10, 64, 256, 2.0

print("target_ratios(1,10) =", target_ratios(1, 10))
print()
hdr = f"{'W x H':>12} {'split?':>7} {'grid wxh':>9} {'tiles':>5} {'+thumb':>7} {'resized(WxH)':>14} {'thumb(WxH)':>12} {'img_tokens':>10}"
print(hdr); print("-" * len(hdr))
for (w, h, label) in [(500,650,"the fixture"),(150,200,"thumbnail"),(300,1000,"tall strip"),
                      (3000,4000,"A4 scan 300dpi"),(1000,300,"wide banner"),(2048,2048,"square"),
                      (1700,2200,"letter 200dpi"),(4000,1000,"panorama")]:
    big = is_too_large(h, w, MAXTOK, P, DS, TOL)
    tw, th = smart_resize(h, w, DS, MINTOK, MAXTOK, P)
    if big:
        gw, gh, twd, thd, n = get_grid_layout(h, w, MINT, MAXT, TILE)
        imgs = n + (1 if gw*gh != 1 else 0)
        tok = n*256 + (256 if gw*gh != 1 else 0)
        print(f"{w:5d} x{h:5d} {'YES':>7} {f'{gw}x{gh}':>9} {n:5d} {imgs:7d} {f'{twd}x{thd}':>14} {f'{tw}x{th}':>12} {tok:10d}")
    else:
        print(f"{w:5d} x{h:5d} {'no':>7} {'1x1':>9} {1:5d} {1:7d} {f'{tw}x{th}':>14} {'-':>12} {(tw//P)*(th//P)//(DS*DS):10d}")
