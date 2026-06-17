"""
R2.3 polysemy demonstration.

(1) Coexistence: a polysemous query's color vector is a visible *blend* of
    its senses (e.g., 'crane' = grey bird + yellow machine).
(2) Controllability: adding a sense-cue word shifts the vector toward the
    targeted sense ('crane bird' -> grey, 'crane machine' -> yellow).

No statistics — purely illustrative. Same extraction as the main pipeline
(resize to 10,000 px, then color-ratio over the 10 base colors).

Output: csv/revision_results/polysemy.{csv,png,pdf}
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from revision_utils import RESULTS_DIR
from color_utils import (
    rgb_to_lch, color_vector_from_pixels, resize_to_pixel_count, COLORS,
)

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images", "r23_raw")

# display order grouped by base word: (filename stem, label, group)
ITEMS = [
    # brand polysemy: bare token vs full registered name (reviewer's examples)
    ("apple",       "apple",            "apple"),
    ("apple_inc",   "Apple Inc.",       "apple"),
    ("amazon",      "amazon",           "amazon"),
    ("amazon_inc",  "Amazon.com Inc.",  "amazon"),
    ("visa",        "visa",             "visa"),
    ("visa_inc",    "Visa Inc.",        "visa"),
    ("shell",       "shell",            "shell"),
    ("shell_plc",   "Shell plc",        "shell"),
    # general object polysemy: base + two sense-cue words
    ("nail",          "nail",          "nail"),
    ("nail_metal",    "metal nail",    "nail"),
    ("nail_finger",   "finger nail",   "nail"),
    ("chip",          "chip",          "chip"),
    ("chip_potato",   "potato chip",   "chip"),
    ("chip_computer", "computer chip", "chip"),
]

COLOR_HEX = {
    "red": "#d62728", "orange": "#ff7f0e", "yellow": "#e6c200",
    "green": "#2ca02c", "blue": "#1f77b4", "purple": "#9467bd",
    "pink": "#e377c2", "black": "#111111", "grey": "#888888",
    "white": "#eaeaea",
}


def vector_for(stem, rng):
    path = os.path.join(RAW, f"{stem}.png")
    if not os.path.exists(path):
        return None
    raw = Image.open(path).convert("RGB")
    arr = np.array(resize_to_pixel_count(raw)).reshape(-1, 3)
    return color_vector_from_pixels(arr, rng)


def main():
    rng = np.random.default_rng(0)
    rows, labels, groups = [], [], []
    for stem, label, group in ITEMS:
        v = vector_for(stem, rng)
        if v is None:
            print(f"missing: {stem}")
            continue
        s = v.sum()
        vn = v / s if s > 0 else v          # normalize to 1 for display
        rows.append(vn)
        labels.append(label)
        groups.append(group)

    import pandas as pd
    df = pd.DataFrame(rows, columns=COLORS)
    df.insert(0, "query", labels)
    out_csv = os.path.join(RESULTS_DIR, "polysemy.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df.round(3).to_string(index=False))

    # ---- two panels: chromatic (7) | achromatic (3), each renormalized ----
    # Chromatic and achromatic are normalized independently in the pipeline,
    # so we show them as two separate stacked bars (avoids a spurious 50/50
    # split). The achromatic panel reveals senses like the black bat / grey
    # seal that the hue panel cannot show.
    CHROM = COLORS[:7]
    ACHR = COLORS[7:]            # black, grey, white
    n = len(rows)
    fig, (axc, axa) = plt.subplots(
        1, 2, figsize=(13, 0.55 * n + 1.2),
        gridspec_kw={"width_ratios": [1, 1]}, sharey=True)
    ypos = []
    y = 0
    prev_group = None
    for vec, label, group in zip(rows, labels, groups):
        if prev_group is not None and group != prev_group:
            y -= 0.5
        ypos.append(y)
        for ax, names in ((axc, CHROM), (axa, ACHR)):
            part = np.array([vec[COLORS.index(c)] for c in names])
            s = part.sum()
            part = part / s if s > 0 else part
            left = 0.0
            for c, w in zip(names, part):
                if w <= 0:
                    continue
                ax.barh(y, w, left=left, color=COLOR_HEX[c],
                        edgecolor="white", linewidth=0.4, height=0.7)
                left += w
        prev_group = group
        y -= 1

    for ax in (axc, axa):
        ax.set_xlim(0, 1)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axc.set_yticks(ypos)
    axc.set_yticklabels(labels, fontsize=10)
    axc.invert_yaxis()
    axc.set_xlabel("Chromatic (hue) composition — 7 chromatic bases", fontsize=10)
    axa.set_xlabel("Achromatic composition — black / grey / white", fontsize=10)
    fig.suptitle("A polysemous query blends its senses' colors; a sense-cue "
                 "word shifts the composition", fontsize=13, y=1.02)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR_HEX[c], edgecolor="grey", label=c)
               for c in COLORS]
    axa.legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5),
               fontsize=8, frameon=False)
    plt.tight_layout()
    png = os.path.join(RESULTS_DIR, "polysemy.png")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    print("Saved:", png, "(+ .pdf)")


if __name__ == "__main__":
    main()
