#!/usr/bin/env python3
"""Generate a 1200x630 OG social-preview image for the CrispEmbed WASM OCR demo.

Portable Pillow-only; no machine paths. Uses the demo's exact palette:
  - ground #0f172a (slate-900, dark) with #2563eb (blue-600) accent
  - white text
Motif: a document/scan card with a magnifier + text lines -> OCR concept.
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
GROUND = (15, 23, 42)      # #0f172a
ACCENT = (37, 99, 235)     # #2563eb
ACCENT_DARK = (29, 78, 216)  # #1d4ed8
WHITE = (255, 255, 255)
MUTED = (148, 163, 184)    # #94a3b8
CARD = (30, 41, 59)        # #1e293b

def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def main():
    img = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(img)

    # subtle top accent bar
    d.rectangle([0, 0, W, 8], fill=ACCENT)

    # left: document card with text lines
    cx, cy = 300, 300
    d.rounded_rectangle([cx-190, cy-150, cx+190, cy+150], radius=18, fill=CARD,
                        outline=ACCENT, width=3)
    # text lines on card
    for i, (w, col) in enumerate([(240, WHITE), (200, WHITE), (220, MUTED),
                                  (180, MUTED), (230, WHITE), (150, MUTED)]):
        y = cy - 100 + i * 40
        d.rounded_rectangle([cx-140, y, cx-140+w, y+18], radius=4, fill=col)

    # magnifier over the card (OCR concept)
    mx, my = 430, 250
    d.ellipse([mx-55, my-55, mx+55, my+55], outline=ACCENT, width=8)
    d.line([mx+40, my+40, mx+95, my+95], fill=ACCENT, width=12)
    d.ellipse([mx-30, my-30, mx+30, my+30], fill=ACCENT)

    # right: headline
    f_title = font(64)
    f_sub = font(30)
    f_tag = font(22)
    tx = 560
    d.text((tx, 200), "CrispEmbed OCR", font=f_title, fill=WHITE)
    d.text((tx, 300), "Client-side OCR powered by", font=f_sub, fill=MUTED)
    d.text((tx, 340), "WebAssembly — no server needed.", font=f_sub, fill=MUTED)
    d.text((tx, 420), "Text · Math → LaTeX · Music · Scans", font=f_tag, fill=ACCENT)

    # bottom tagline
    d.text((60, 560), "crispstrobe.github.io/CrispEmbed", font=f_tag, fill=MUTED)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "og-image.png")
    img.save(out, "PNG")
    print("wrote", out, img.size)

if __name__ == "__main__":
    main()
