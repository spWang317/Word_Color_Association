"""
Utility module: CIELch color space, color binning into named base
colors, pixel-to-vector reduction, and image resizing helpers.

Used by the per-word color vector pipeline.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from revision_utils import RESULTS_DIR

# Raw screenshots captured for the R2.6 word set (800x438 full-res).
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "Images", "r26_raw")

# Pipeline target: each (cropped) screenshot is resized to ~10,000 pixels
# total, maintaining aspect ratio (Lanczos) — exactly as in word2image.py.
PIXEL_COUNT = 10_000

# Constants from calculation.py
WILD_MEAN = 0.3141835271415166      # fraction of brightest pixels (white grid) trimmed
CHROMA_THRESHOLD = 15.930582082686795

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
          "black", "grey", "white"]

HUE_RANGES = {
    "red":    [(19.326959847036328, 45.67667984189723)],
    "orange": [(45.67667984189723, 80.90534979423869)],
    "yellow": [(80.90534979423869, 120.1811320754717)],
    "green":  [(120.1811320754717, 203.28)],
    "blue":   [(203.28, 293.86897590361446)],
    "purple": [(293.86897590361446, 331.2451923076923)],
    "pink":   [(331.2451923076923, 360.0), (0.0, 19.326959847036328)],
}
LIGHT_RANGES = {
    "black": (0.0, 41.65384615384615),
    "grey":  (41.65384615384615, 82.89746682750301),
    "white": (82.89746682750301, 100.0),
}

# Crop fractions of image height (top portion kept)
SEED = 0


# ---- vectorized RGB -> LCH --------------------------------------------------

def rgb_to_lch(pixels: np.ndarray) -> np.ndarray:
    """pixels: (N, 3) uint8 -> (N, 3) array of (L, C, H)."""
    rgb = pixels.astype(float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    # normalize by D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    C = np.sqrt(a * a + bb * bb)
    H = np.degrees(np.arctan2(bb, a))
    H = np.where(H < 0, H + 360.0, H)
    return np.stack([L, C, H], axis=1)


def color_vector_from_pixels(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replicates the manuscript pipeline's color-ratio extraction."""
    lch = rgb_to_lch(pixels)
    L, C, H = lch[:, 0], lch[:, 1], lch[:, 2]

    # --- Lightness: trim brightest WILD_MEAN fraction (white grid), redistribute ---
    thr = np.percentile(L, 100.0 - WILD_MEAN * 100.0)
    L_cut = L[L <= thr]
    n_cut = len(L) - len(L_cut)
    hist, bins = np.histogram(L_cut, bins=np.arange(0, 101, 1))
    if hist.sum() > 0 and n_cut > 0:
        p = hist / hist.sum()
        add = rng.choice(bins[:-1], size=n_cut, p=p)
        L_final = np.concatenate([L_cut, add])
    else:
        L_final = L_cut

    # --- Hue: pixels with C < threshold have undefined hue; redistribute ---
    valid = H[C >= CHROMA_THRESHOLD]
    n_low = len(H) - len(valid)
    hist_h, bins_h = np.histogram(valid, bins=np.linspace(0, 360, 361))
    if hist_h.sum() > 0 and n_low > 0:
        ratios = hist_h / hist_h.sum()
        counts = np.round(ratios * n_low).astype(int)
        extra = np.repeat(bins_h[:-1] + 0.5, counts)
        H_final = np.concatenate([valid, extra])
    else:
        H_final = valid

    # --- chromatic ratios ---
    vec = np.zeros(len(COLORS))
    n_h = len(H_final)
    if n_h > 0:
        hmod = H_final % 360
        for j, c in enumerate(COLORS[:7]):
            cnt = 0
            for (lo, hi) in HUE_RANGES[c]:
                cnt += np.sum((hmod >= lo) & (hmod <= hi))
            vec[j] = cnt / n_h

    # --- achromatic ratios ---
    n_l = len(L_final)
    if n_l > 0:
        for j, c in enumerate(["black", "grey", "white"], start=7):
            lo, hi = LIGHT_RANGES[c]
            vec[j] = np.sum((L_final >= lo) & (L_final <= hi)) / n_l
    return vec


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def resize_to_pixel_count(img: Image.Image, pixel_count: int = PIXEL_COUNT) -> Image.Image:
    """Resize so width*height ≈ pixel_count, keeping aspect ratio (Lanczos),
    matching the manuscript pipeline's resize step."""
    W, H = img.size
    if W * H == 0:
        return img
    scale = (pixel_count / (W * H)) ** 0.5
    new_W = max(1, int(round(W * scale)))
    new_H = max(1, int(round(H * scale)))
    return img.resize((new_W, new_H), Image.LANCZOS)
