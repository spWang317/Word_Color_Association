"""
Standalone SERP-like grid generators for the *moss* and *lilac* queries,
used as schematic components of Fig 1 and Fig 2.

Layout:
  - Multiple rows; each row has a uniform height (rows can have different heights)
  - Within each row, cells have variable widths (justified to fill the canvas)
  - White gutters separate every cell
  - Each cell is filled with a pixel mosaic sampled from a query-typical CIELch
    palette (moss → greens/browns/darks; lilac → purples/greens/blues)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIRS = [
    Path("/Users/qgroup/Desktop/[word-color-association]final_draft"),
    Path("/Users/qgroup/Desktop/PLOS_Submission_Final/01_For_Upload/B_Figures"),
]


def lch_to_rgb(L, c, h_deg):
    h = np.radians(h_deg)
    a, b = c * np.cos(h), c * np.sin(h)
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787

    x, y, z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_ = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def gamma(u):
        return 1.055 * u ** (1 / 2.4) - 0.055 if u > 0.0031308 else 12.92 * u

    return np.clip([gamma(r), gamma(g), gamma(b_)], 0, 1)


def sample_moss_palette(n, seed):
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(int(0.60 * n)):
        tiles.append(lch_to_rgb(
            rng.uniform(20, 70), rng.uniform(15, 50), rng.uniform(120, 200)))
    for _ in range(int(0.15 * n)):
        tiles.append(lch_to_rgb(
            rng.uniform(25, 55), rng.uniform(20, 50), rng.uniform(45, 90)))
    for _ in range(int(0.15 * n)):
        tiles.append(lch_to_rgb(
            rng.uniform(5, 30), rng.uniform(0, 8), rng.uniform(0, 360)))
    while len(tiles) < n:
        tiles.append(lch_to_rgb(
            rng.uniform(40, 80), rng.uniform(0, 10), rng.uniform(120, 180)))
    rng.shuffle(tiles)
    return np.array(tiles[:n])


def sample_lilac_palette(n, seed):
    rng = np.random.default_rng(seed)
    tiles = []
    mix = [
        (0.43, (294, 332), (35, 80)),
        (0.25, (120, 203), (25, 80)),
        (0.24, (203, 294), (30, 75)),
        (0.05, (332, 380), (45, 85)),
    ]
    for prop, (h_lo, h_hi), (L_lo, L_hi) in mix:
        k = max(1, int(prop * n * 0.8))
        for _ in range(k):
            tiles.append(lch_to_rgb(
                rng.uniform(L_lo, L_hi),
                rng.uniform(20, 55),
                rng.uniform(h_lo, h_hi) % 360))
    while len(tiles) < n:
        L = rng.uniform(20, 95)
        tiles.append([L / 100, L / 100, L / 100])
    rng.shuffle(tiles)
    return np.array(tiles[:n])


def build_masonry(palette_func, W=1400, H=900, n_rows=6, gutter=10,
                  row_h_jitter=0.18, cell_w_mean=210, cell_w_std=80,
                  pixel_size=8, seed=42):
    """Masonry SERP-like grid array (H, W, 3) in 0..1, white background."""
    rng = np.random.default_rng(seed)
    arr = np.ones((H, W, 3))

    base = (H - (n_rows + 1) * gutter) // n_rows
    row_heights = [int(base * rng.uniform(1 - row_h_jitter, 1 + row_h_jitter))
                   for _ in range(n_rows)]
    scale = (H - (n_rows + 1) * gutter) / sum(row_heights)
    row_heights = [int(h * scale) for h in row_heights]
    diff = H - sum(row_heights) - (n_rows + 1) * gutter
    row_heights[-1] += diff

    y0 = gutter
    for row_h in row_heights:
        remaining = W - 2 * gutter
        cell_widths = []
        min_w = 90
        while remaining > min_w + gutter:
            w = max(min_w, int(rng.normal(cell_w_mean, cell_w_std)))
            if w + gutter > remaining:
                if cell_widths:
                    cell_widths[-1] += remaining + gutter
                    remaining = 0
                    break
                else:
                    cell_widths.append(remaining)
                    remaining = 0
                    break
            cell_widths.append(w)
            remaining -= (w + gutter)
        if remaining > 0 and cell_widths:
            cell_widths[-1] += remaining + gutter

        x0 = gutter
        for cell_w in cell_widths:
            n_cols = max(1, cell_w // pixel_size)
            n_rows_cell = max(1, row_h // pixel_size)
            n_total = n_cols * n_rows_cell
            cell_seed = int(rng.integers(0, 100000))
            palette = palette_func(n_total, seed=cell_seed)
            for k in range(n_total):
                pr, pc = divmod(k, n_cols)
                py = y0 + pr * pixel_size
                px_ = x0 + pc * pixel_size
                py_end = min(y0 + row_h, py + pixel_size)
                px_end = min(x0 + cell_w, px_ + pixel_size)
                arr[py:py_end, px_:px_end] = palette[k]
            x0 += cell_w + gutter
        y0 += row_h + gutter

    return arr


def save_array(arr, path_base, dpi=300):
    H, W = arr.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(f"{path_base}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    moss = build_masonry(sample_moss_palette, W=1400, H=900, seed=7,
                         pixel_size=8)
    # Same outer seed + identical layout params -> same row heights & cell
    # widths; only the pixel chunkiness changes (downsampled feel).
    moss_resized = build_masonry(sample_moss_palette, W=1400, H=900, seed=7,
                                  pixel_size=28)
    lilac = build_masonry(sample_lilac_palette, W=1400, H=900, seed=11,
                          pixel_size=8)
    for d in OUT_DIRS:
        save_array(moss,         d / "moss_serp_grid")
        save_array(moss_resized, d / "moss_serp_grid_resized")
        save_array(lilac,        d / "lilac_serp_grid")
        print(f"Saved: {d / 'moss_serp_grid'}.{{png,pdf}}")
        print(f"Saved: {d / 'moss_serp_grid_resized'}.{{png,pdf}}")
        print(f"Saved: {d / 'lilac_serp_grid'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
