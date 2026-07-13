"""Depth-to-depth color-vector correlation for the aggregate-stability analysis.

For each word we already store per-page Google-image-search screenshots at
`Images/r26b_pages/{word}/page_{N}.png` for N = 1..max_page. This script
extracts the 10-d named base-color vector from every page, then reports:

(A) a mean 10x10 depth-to-depth cosine matrix aggregating across the 30
    words (words with fewer than 10 pages are included for the pairs
    they cover), and

(B) a random-sampling analysis in which K pages are drawn without
    replacement from the {2..max_page} pool for each word, their color
    vectors are averaged, and the cosine between the sampled aggregate
    and the top viewport (page 1) is recorded. B bootstrap draws per
    (word, K) yield a distribution.

Outputs
-------
csv/revision_results/page_correlation_matrix.csv   mean cosine matrix
csv/revision_results/page_random_sampling.csv      per-draw records
figures/page_correlation_analysis.pdf              two-panel figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from color_utils import (
    PIXEL_COUNT, SEED, color_vector_from_pixels, cosine, resize_to_pixel_count,
)
from revision_utils import RESULTS_DIR

SCREENSHOT_DIR = Path(__file__).resolve().parent / "Images" / "r26b_pages"
META_PATH = Path(RESULTS_DIR) / "page_metadata.csv"

MATRIX_CSV = Path(RESULTS_DIR) / "page_correlation_matrix.csv"
SAMPLING_CSV = Path(RESULTS_DIR) / "page_random_sampling.csv"
DISTANCE_CSV = Path(RESULTS_DIR) / "page_correlation_by_distance.csv"
FIG_PATH = Path(__file__).resolve().parent / "figures" / "page_correlation_analysis.pdf"

SAMPLING_K_VALUES = [2, 3, 5, 7]
B_BOOTSTRAP = 100


def color_vec(png_path: Path, rng: np.random.Generator) -> np.ndarray:
    img = Image.open(png_path).convert("RGB")
    resized = resize_to_pixel_count(img)
    arr = np.array(resized).reshape(-1, 3)
    return color_vector_from_pixels(arr, rng)


def per_word_vectors(word: str, max_page: int, rng: np.random.Generator):
    """Return {page: 10-d vector} for pages 1..max_page."""
    vectors = {}
    for p in range(1, max_page + 1):
        path = SCREENSHOT_DIR / word / f"page_{p}.png"
        if not path.exists():
            continue
        vectors[p] = color_vec(path, rng)
    return vectors


def depth_matrix(all_vectors, n_pages: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate mean pairwise cosine across words, with per-cell counts."""
    total = np.zeros((n_pages, n_pages))
    counts = np.zeros((n_pages, n_pages), dtype=int)
    for vecs in all_vectors.values():
        pages = sorted(vecs.keys())
        for i in pages:
            for j in pages:
                if i > n_pages or j > n_pages:
                    continue
                total[i - 1, j - 1] += float(cosine(vecs[i], vecs[j]))
                counts[i - 1, j - 1] += 1
    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, total / counts, np.nan)
    return mean, counts


def random_sampling(all_vectors, k_values, n_draws, rng: np.random.Generator):
    """For each K, draw N bootstrap subsets of K pages (excluding page 1),
    average their vectors, and record the cosine to page 1."""
    rows = []
    for word, vecs in all_vectors.items():
        if 1 not in vecs:
            continue
        v1 = vecs[1]
        deeper_pages = [p for p in vecs if p != 1]
        if not deeper_pages:
            continue
        deeper_mat = np.vstack([vecs[p] for p in deeper_pages])
        for K in k_values:
            if K > len(deeper_pages):
                continue
            for draw in range(n_draws):
                idx = rng.choice(len(deeper_pages), size=K, replace=False)
                sampled = deeper_mat[idx].mean(axis=0)
                cos = float(cosine(v1, sampled))
                rows.append({
                    "word": word, "K": K, "draw": draw,
                    "cos_sampled_vs_p1": cos,
                    "sampled_pages": ",".join(str(deeper_pages[i]) for i in sorted(idx)),
                })
    return pd.DataFrame(rows)


def plot_analysis(matrix, counts, sampling_df, top_vs_last, out_path):
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.55)

    ax1 = fig.add_subplot(gs[0, 0])
    n = matrix.shape[0]
    im = ax1.imshow(matrix, cmap="viridis", vmin=np.nanmin(matrix), vmax=1.0,
                    origin="lower", aspect="equal")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(range(1, n + 1))
    ax1.set_yticklabels(range(1, n + 1))
    ax1.set_xlabel("page depth $j$")
    ax1.set_ylabel("page depth $i$")
    # Title placement is handled at the figure level below so panels (a) and
    # (b) share the same title y-coordinate.
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue
            val = matrix[i, j]
            color = "white" if val < 0.85 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color=color)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.045, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    if not sampling_df.empty:
        by_K = sampling_df.groupby("K")["cos_sampled_vs_p1"]
        Ks = sorted(sampling_df["K"].unique())
        data = [by_K.get_group(K).to_numpy() for K in Ks]
        positions = list(range(len(Ks)))
        parts = ax2.boxplot(data, positions=positions, widths=0.55,
                            patch_artist=True, showfliers=False)
        for patch in parts["boxes"]:
            patch.set(facecolor="#a6c8ff", edgecolor="#1f4e79", linewidth=0.9)
        for line in parts["medians"]:
            line.set(color="#c00000", linewidth=1.4)
        ax2.set_xticks(positions)
        ax2.set_xticklabels([str(K) for K in Ks])
        ax2.set_ylabel("cosine similarity to top viewport (page 1)")
        ax2.set_xlabel("number of randomly sampled deeper viewports, $K$")
        # Title placement is handled at the figure level below.
        ax2.axhline(top_vs_last, color="#c00000", linestyle="--", linewidth=0.9,
                    label=f"top vs page-10 mean = {top_vs_last:.3f}")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax2.legend(loc="lower right", fontsize=9)

    fig.canvas.draw()
    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()

    # 1. Shrink panel (b) vertically to 0.8x, centered around its current midline.
    new_height = ax2_bbox.height * 0.8
    new_y0 = ax2_bbox.y0 + (ax2_bbox.height - new_height) / 2

    # 2. Move panel (b) left so the gap between (a)'s right edge and (b)'s left
    #    edge is halved, while (a)'s position and both panel widths stay fixed.
    current_gap = ax2_bbox.x0 - (ax1_bbox.x0 + ax1_bbox.width)
    new_x0 = ax2_bbox.x0 - current_gap * 0.2

    ax2.set_position([new_x0, new_y0, ax2_bbox.width, new_height])
    fig.canvas.draw()

    title_y = 0.84
    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()
    fig.text(ax1_bbox.x0, title_y,
             "(a) Depth-to-depth mean cosine ($N=30$ words)",
             fontsize=11, ha="left", va="bottom")
    fig.text(ax2_bbox.x0, title_y,
             "(b) Random-sampled $K$ viewports vs. top viewport",
             fontsize=11, ha="left", va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pages", type=int, default=10)
    parser.add_argument("--k-values", type=str, default="2,3,5,7")
    parser.add_argument("--bootstrap", type=int, default=B_BOOTSTRAP)
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",") if k]

    meta = pd.read_csv(META_PATH)
    rng = np.random.default_rng(SEED)

    all_vectors = {}
    for _, m in meta.iterrows():
        word = m["word"]
        max_page = int(m["max_page"])
        if max_page < 2:
            continue
        vecs = per_word_vectors(word, max_page, rng)
        if vecs:
            all_vectors[word] = vecs
        print(f"  {word}: {len(vecs)} pages")

    print(f"\nProcessed {len(all_vectors)} words")

    matrix, counts = depth_matrix(all_vectors, n_pages=args.n_pages)
    matrix_df = pd.DataFrame(matrix,
                             index=[f"page_{i}" for i in range(1, args.n_pages + 1)],
                             columns=[f"page_{j}" for j in range(1, args.n_pages + 1)])
    matrix_df.to_csv(MATRIX_CSV)
    print(f"Wrote {MATRIX_CSV}")

    from scipy.stats import pearsonr, spearmanr, linregress
    by_distance_pairs = {}
    pooled_d, pooled_c = [], []
    for vecs in all_vectors.values():
        pages = sorted(vecs.keys())
        for i in pages:
            for j in pages:
                if j <= i:
                    continue
                d = j - i
                c = float(cosine(vecs[i], vecs[j]))
                by_distance_pairs.setdefault(d, []).append(c)
                pooled_d.append(d)
                pooled_c.append(c)
    ds = sorted(by_distance_pairs.keys())
    dist_rows = []
    for d in ds:
        arr = np.array(by_distance_pairs[d])
        dist_rows.append({
            "page_distance": d,
            "n_pairs": len(arr),
            "mean_cosine": float(arr.mean()),
            "median_cosine": float(np.median(arr)),
            "std_cosine": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        })
    dist_df = pd.DataFrame(dist_rows)
    pool_d_arr = np.array(pooled_d)
    pool_c_arr = np.array(pooled_c)
    r_agg, p_agg = pearsonr([r["page_distance"] for r in dist_rows],
                            [r["mean_cosine"] for r in dist_rows])
    r_pool, p_pool = pearsonr(pool_d_arr, pool_c_arr)
    slope, intercept, _, p_slope, se = linregress(pool_d_arr, pool_c_arr)
    dist_df.attrs["pearson_r_aggregated"] = r_agg
    dist_df.attrs["pearson_p_aggregated"] = p_agg
    dist_df.attrs["pearson_r_pooled"] = r_pool
    dist_df.attrs["pearson_p_pooled"] = p_pool
    dist_df.attrs["linreg_intercept"] = intercept
    dist_df.attrs["linreg_slope"] = slope
    dist_df.attrs["linreg_slope_se"] = se
    dist_df.attrs["linreg_slope_p"] = p_slope
    dist_df.to_csv(DISTANCE_CSV, index=False)
    print(f"Wrote {DISTANCE_CSV}  ({len(dist_df)} distances)")
    print(f"  Pearson r on aggregated means: {r_agg:.3f} (p = {p_agg:.3g})")
    print(f"  Pearson r on pooled per-pair : {r_pool:.3f} (p = {p_pool:.3g})")
    print(f"  Linear fit: cosine = {intercept:.4f} + {slope:.4f} × distance "
          f"(slope p = {p_slope:.3g})")

    sampling_df = random_sampling(all_vectors, k_values, args.bootstrap, rng)
    sampling_df.to_csv(SAMPLING_CSV, index=False)
    print(f"Wrote {SAMPLING_CSV}  ({len(sampling_df)} rows)")

    top_vs_last = float(np.nanmean([matrix[0, i] for i in range(1, args.n_pages)]))
    plot_analysis(matrix, counts, sampling_df, top_vs_last, FIG_PATH)
    print(f"Wrote {FIG_PATH}")

    print("\n--- summary ---")
    print(matrix_df.round(3).to_string())
    print()
    if not sampling_df.empty:
        summary = (sampling_df.groupby("K")["cos_sampled_vs_p1"]
                   .agg(["mean", "median", "std", "min", "max"]))
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
