"""
Helper module: COLOR_LEXICON, BASE_RGB, CIELch conversions, lexical lookup.
Imported by setlur_scrape.py and setlur_analyze.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from nltk.corpus import wordnet as wn

OUT_DIR = Path("/Users/qgroup/Desktop/[word-color-association]final_draft")
NRC_TABLE = OUT_DIR / "SI File" / "csv" / "analyses" / "nrc_validation_summary_table.csv"

SEED = 0
N_SAMPLE = 100

CHROMA_THRESHOLD = 15.930582082686795
HUE_RANGES = {
    "red":    (19.3270, 45.6767),
    "orange": (45.6767, 80.9054),
    "yellow": (80.9054, 120.1811),
    "green":  (120.1811, 203.2800),
    "blue":   (203.2800, 293.8690),
    "purple": (293.8690, 331.2452),
    "pink":   (331.2452, 379.3270),
}
LIGHTNESS_RANGES = {
    "black": (0.0, 41.6538),
    "grey":  (41.6538, 82.8975),
    "white": (82.8975, 100.0),
}

CATS = list(HUE_RANGES.keys()) + list(LIGHTNESS_RANGES.keys())

COLOR_LEXICON = {
    "red": (255, 0, 0), "crimson": (220, 20, 60), "scarlet": (255, 36, 0),
    "maroon": (128, 0, 0), "ruby": (224, 17, 95), "burgundy": (128, 0, 32),
    "orange": (255, 165, 0), "amber": (255, 191, 0), "coral": (255, 127, 80),
    "tangerine": (242, 133, 0), "rust": (183, 65, 14),
    "yellow": (255, 255, 0), "gold": (255, 215, 0), "lemon": (255, 247, 0),
    "mustard": (255, 219, 88), "saffron": (244, 196, 48),
    "green": (0, 128, 0), "olive": (128, 128, 0), "lime": (0, 255, 0),
    "emerald": (0, 168, 107), "forest": (34, 139, 34), "mint": (152, 251, 152),
    "blue": (0, 0, 255), "navy": (0, 0, 128), "azure": (0, 127, 255),
    "cyan": (0, 255, 255), "cobalt": (0, 71, 171), "indigo": (75, 0, 130),
    "sapphire": (15, 82, 186),
    "purple": (128, 0, 128), "violet": (143, 0, 255),
    "lavender": (181, 126, 220), "magenta": (255, 0, 255),
    "pink": (255, 192, 203), "rose": (255, 0, 127),
    "salmon": (250, 128, 114), "fuchsia": (255, 0, 255),
    "black": (0, 0, 0), "white": (255, 255, 255),
    "grey": (128, 128, 128), "gray": (128, 128, 128),
    "charcoal": (54, 69, 79), "silver": (192, 192, 192),
    "ivory": (255, 255, 240), "cream": (255, 253, 208),
}

BASE_RGB = {
    "red":    (255,   0,   0),
    "orange": (255, 165,   0),
    "yellow": (255, 255,   0),
    "green":  (  0, 128,   0),
    "blue":   (  0,   0, 255),
    "purple": (128,   0, 128),
    "pink":   (255, 192, 203),
    "black":  (  0,   0,   0),
    "grey":   (128, 128, 128),
    "white":  (255, 255, 255),
}


def rgb_to_lch(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    def srgb_lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = srgb_lin(r), srgb_lin(g), srgb_lin(b)
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y =  r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    def f(t): return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a, bb = 500 * (fx - fy), 200 * (fy - fz)
    c = (a * a + bb * bb) ** 0.5
    h = (np.degrees(np.arctan2(bb, a)) + 360.0) % 360.0
    return L, c, h


def rgb_to_our_bin(rgb):
    L, c, h = rgb_to_lch(rgb)
    if c < CHROMA_THRESHOLD:
        for color, (lo, hi) in LIGHTNESS_RANGES.items():
            if lo <= L < hi:
                return color
        return "grey"
    for color, (lo, hi) in HUE_RANGES.items():
        if lo <= h <= hi:
            return color
    return "pink"


def lch_distance(rgb1, rgb2):
    L1, c1, h1 = rgb_to_lch(rgb1)
    L2, c2, h2 = rgb_to_lch(rgb2)
    a1, b1 = c1 * np.cos(np.radians(h1)), c1 * np.sin(np.radians(h1))
    a2, b2 = c2 * np.cos(np.radians(h2)), c2 * np.sin(np.radians(h2))
    return float(np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))


def nearest_w3c_name(rgb):
    return min(COLOR_LEXICON.keys(), key=lambda c: lch_distance(rgb, COLOR_LEXICON[c]))


def collect_linguistic_text(word):
    chunks = []
    syns = wn.synsets(word)
    if not syns:
        return ""
    for s in syns[:8]:
        chunks.append(s.definition() or "")
        for lemma in s.lemmas():
            chunks.append(lemma.name().replace("_", " "))
        for related in (s.hyponyms() + s.hypernyms())[:5]:
            chunks.append(related.definition() or "")
            for lemma in related.lemmas():
                chunks.append(lemma.name().replace("_", " "))
    return " ".join(chunks).lower()


def lexical_lookup(word):
    text = collect_linguistic_text(word)
    if not text:
        return None, None
    scores = {c: len(re.compile(r"\b" + re.escape(c) + r"\b", re.I).findall(text))
              for c in COLOR_LEXICON}
    if not any(v > 0 for v in scores.values()):
        return None, None
    top = max(scores, key=scores.get)
    return top, COLOR_LEXICON[top]
