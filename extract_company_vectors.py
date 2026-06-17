"""
Re-extract 10-d color vectors per company for {logo, web-image} sets, save to CSV.

Outputs: figure_rebuild/data/company_vectors.csv with columns
  company, side (logo|web), red, orange, yellow, green, blue, purple, pink,
  white, grey, black
plus a separate companion CSV with cosine(logo, web) per company.

Replicates the extraction pipeline from LogoAndScreenshotOfBrands.ipynb without
needing to run the notebook. Pure data, no third-party imagery in output.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/Users/qgroup/Desktop/word_color_association-main/brand materials")
LOGO_DIR = ROOT / "company_logos_resize"
WEB_DIR = ROOT / "Resize_Image"
OUT_DIR = Path("/Users/qgroup/Desktop/word_color_association-main/company_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_THRESHOLD = 15.930582082686795

HUE_RANGES = {
    "red":    [(19.3270, 45.6767)],
    "orange": [(45.6767, 80.9054)],
    "yellow": [(80.9054, 120.1811)],
    "green":  [(120.1811, 203.2800)],
    "blue":   [(203.2800, 293.8690)],
    "purple": [(293.8690, 331.2452)],
    "pink":   [(331.2452, 360.0), (0.0, 19.3270)],
}
LIGHTNESS_RANGES = {
    "black": [(0.00, 41.6538)],
    "grey":  [(41.6538, 82.8975)],
    "white": [(82.8975, 100.00)],
}

CHROMATIC_KEYS = list(HUE_RANGES.keys())
ACHROMATIC_KEYS = list(LIGHTNESS_RANGES.keys())
ALL_KEYS = CHROMATIC_KEYS + ACHROMATIC_KEYS


def rgb_to_lch(rgb):
    """RGB (0-255) → CIE Lch (L 0-100, c 0-~150, h 0-360 degrees)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    r = r if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    c = (a * a + b_ * b_) ** 0.5
    h = (np.degrees(np.arctan2(b_, a)) + 360.0) % 360.0
    return L, c, h


def image_to_lch_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] == 4:
        pixels = pixels[pixels[:, 3] >= 10][:, :3]
    out = np.zeros((len(pixels), 3))
    for i, px in enumerate(pixels):
        out[i] = rgb_to_lch(px)
    return out


def vectorize(lch: np.ndarray) -> dict[str, float]:
    counts = {k: 0 for k in ALL_KEYS}
    n = max(1, len(lch))
    chromatic = lch[lch[:, 1] >= CHROMA_THRESHOLD]
    for _, _, h in chromatic:
        for color, ranges in HUE_RANGES.items():
            for lo, hi in ranges:
                if lo <= h <= hi:
                    counts[color] += 1
                    break
    for L, _, _ in lch:
        for color, ranges in LIGHTNESS_RANGES.items():
            for lo, hi in ranges:
                if lo <= L <= hi:
                    counts[color] += 1
                    break
    return {k: counts[k] / n for k in ALL_KEYS}


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def main():
    files = sorted(LOGO_DIR.glob("*.png"))
    companies = [f.stem for f in files if (WEB_DIR / f.name).exists()]
    print(f"Found {len(companies)} companies in both logo and web directories.")

    rows = []
    sims = []
    for i, c in enumerate(companies, 1):
        lpath, wpath = LOGO_DIR / f"{c}.png", WEB_DIR / f"{c}.png"
        l_lch = image_to_lch_array(lpath)
        w_lch = image_to_lch_array(wpath)
        l_vec = vectorize(l_lch)
        w_vec = vectorize(w_lch)
        rows.append({"company": c, "side": "logo", **l_vec})
        rows.append({"company": c, "side": "web", **w_vec})
        lu = np.array([l_vec[k] for k in ALL_KEYS])
        wu = np.array([w_vec[k] for k in ALL_KEYS])
        sims.append({"company": c, "cosine": cosine(lu, wu)})
        if i % 10 == 0:
            print(f"  {i}/{len(companies)} processed")

    df = pd.DataFrame(rows, columns=["company", "side"] + ALL_KEYS)
    df.to_csv(OUT_DIR / "company_vectors.csv", index=False)
    print(f"Saved: {OUT_DIR / 'company_vectors.csv'}  ({len(df)} rows)")

    sim_df = pd.DataFrame(sims).sort_values("cosine", ascending=False).reset_index(drop=True)
    sim_df.to_csv(OUT_DIR / "company_cosine.csv", index=False)
    print(f"Saved: {OUT_DIR / 'company_cosine.csv'}  ({len(sim_df)} rows)")
    print("\nTop 5:")
    print(sim_df.head(5).to_string(index=False))
    print("\nBottom 5:")
    print(sim_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
