"""
R2.6b — Analyze page 1 vs page N (deepest reachable) color vectors.

For each word in the R2.6b screenshot set:
  - Read page_1.png and page_{max_page}.png.
  - Apply the standard color-vector extraction (same pipeline as crop test).
  - Compute cosine similarity between page-1 and deepest-page vectors.

Group results by three cases:
  A. max_page == 10  (target reached)
  B. 1 < max_page < 10 (limited — hit Google's last page early)
  C. max_page == 1   (single-page query, trivially cosine = 1)

Outputs:
  csv/revision_results/page_compare.csv
  csv/revision_results/page_compare.png
  csv/revision_results/page_compare.pdf
  csv/revision_results/page_compare_summary.md
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from revision_utils import RESULTS_DIR
from color_utils import (
    PIXEL_COUNT, SEED, color_vector_from_pixels, cosine, resize_to_pixel_count,
)

SCREENSHOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Images", "r26b_pages"
)
META_PATH = os.path.join(RESULTS_DIR, "page_metadata.csv")


def color_vec(png_path: str, rng: np.random.Generator) -> np.ndarray:
    img = Image.open(png_path).convert("RGB")
    resized = resize_to_pixel_count(img)
    arr = np.array(resized).reshape(-1, 3)
    return color_vector_from_pixels(arr, rng)


def main():
    meta = pd.read_csv(META_PATH)
    print(f"Loaded metadata for {len(meta)} words")
    print(meta.groupby("max_page").size().to_string())

    rng = np.random.default_rng(SEED)
    rows = []
    for _, m in meta.iterrows():
        w = m["word"]
        max_page = int(m["max_page"])
        if max_page < 1:
            rows.append({"word": w, "max_page": 0, "cos_p1_pN": np.nan,
                         "category": "FAIL"})
            continue
        p1_path = os.path.join(SCREENSHOT_DIR, w, "page_1.png")
        pN_path = os.path.join(SCREENSHOT_DIR, w, f"page_{max_page}.png")
        if not os.path.exists(p1_path) or not os.path.exists(pN_path):
            rows.append({"word": w, "max_page": max_page, "cos_p1_pN": np.nan,
                         "category": "MISSING_FILE"})
            continue
        v1 = color_vec(p1_path, rng)
        vN = color_vec(pN_path, rng)
        cos = float(cosine(v1, vN)) if max_page > 1 else 1.0
        if max_page == 10:
            cat = "A_full_10"
        elif max_page > 1:
            cat = "B_partial"
        else:
            cat = "C_single_page"
        rows.append({"word": w, "max_page": max_page, "cos_p1_pN": cos,
                     "category": cat})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "page_compare.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ------ Summary stats ------
    by_cat = df.dropna(subset=["cos_p1_pN"]).groupby("category")
    print("\n=== Per-category cosine summary ===")
    print(by_cat["cos_p1_pN"].agg(["count", "mean", "std", "min", "max"]).to_string())

    summary_lines = [
        "# R2.6b — Page 1 vs Page 10 (or last reachable) color vector comparison",
        "",
        f"- Sample: {len(meta)} frequency-random common nouns (same as R2.6 v2).",
        "- For each word, page 1 = top viewport at scroll position 0; "
        "page N = viewport at scroll position (N-1) × viewport_height "
        "after triggering Google's infinite scroll / 'Show more' button as needed.",
        "- N target = 10. If the scroll hits the bottom of results before N=10, "
        "the deepest reachable page (max_page) is used instead.",
        "",
        "## Category breakdown",
        "",
        "| Category | Description | n |",
        "|---|---|---:|",
        f"| A | max_page == 10 (target reached) | {(df['category']=='A_full_10').sum()} |",
        f"| B | 1 < max_page < 10 (limited — bottom hit early) | {(df['category']=='B_partial').sum()} |",
        f"| C | max_page == 1 (single-page query, cosine ≡ 1 trivially) | {(df['category']=='C_single_page').sum()} |",
        f"| FAIL/MISSING | scrape error or missing screenshot | {df['category'].isin(['FAIL', 'MISSING_FILE']).sum()} |",
        "",
        "## Cosine similarity summary by category",
        "",
        "| Category | n | mean cos | SD | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat in ["A_full_10", "B_partial", "C_single_page"]:
        sub = df[df["category"] == cat]
        if len(sub):
            summary_lines.append(
                f"| {cat} | {len(sub)} | "
                f"{sub['cos_p1_pN'].mean():.4f} | "
                f"{sub['cos_p1_pN'].std():.4f} | "
                f"{sub['cos_p1_pN'].min():.4f} | "
                f"{sub['cos_p1_pN'].max():.4f} |"
            )

    # All-category (excluding C since trivial) aggregate
    main_df = df[df["category"].isin(["A_full_10", "B_partial"])]
    if len(main_df):
        summary_lines += [
            "",
            "## Combined A+B (non-trivial: page 1 vs deepest reached, max_page ≥ 2)",
            "",
            f"- n = {len(main_df)}",
            f"- mean cosine(page1, page_N) = {main_df['cos_p1_pN'].mean():.4f}",
            f"- SD = {main_df['cos_p1_pN'].std():.4f}",
            f"- median = {main_df['cos_p1_pN'].median():.4f}",
            f"- min = {main_df['cos_p1_pN'].min():.4f}  "
            f"(word: {main_df.loc[main_df['cos_p1_pN'].idxmin(), 'word']})",
        ]

    summary_lines += ["", "## Per-word table (sorted by cosine ascending)", ""]
    sorted_df = df.sort_values("cos_p1_pN", ascending=True, na_position="last")
    summary_lines.append("| Word | max_page | cosine(page1, pageN) | Category |")
    summary_lines.append("|---|---:|---:|---|")
    for _, r in sorted_df.iterrows():
        cos_str = f"{r['cos_p1_pN']:.4f}" if pd.notna(r['cos_p1_pN']) else "—"
        summary_lines.append(
            f"| {r['word']} | {r['max_page']} | {cos_str} | {r['category']} |"
        )

    out_md = os.path.join(RESULTS_DIR, "page_compare_summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {out_md}")

    # ------ Figure: single panel, y truncated to data-relevant range ------
    # No broken axis — statistical-paper convention is to truncate the y-axis
    # to the data-relevant range and state the truncation in the caption.
    valid = df.dropna(subset=["cos_p1_pN"]).copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    cat_color = {"A_full_10": "#1f77b4", "B_partial": "#ff7f0e"}
    pos_jitter = np.random.default_rng(0).normal(0, 0.07, len(valid))

    for cat in ["A_full_10", "B_partial"]:
        mask = valid["category"].values == cat
        if not mask.any():
            continue
        n = int(mask.sum())
        label = (f"page 10 reached (n = {n})"
                 if cat == "A_full_10"
                 else f"result list ended at page 8–9 (n = {n})")
        ax.scatter(
            pos_jitter[mask],
            valid["cos_p1_pN"].values[mask],
            color=cat_color[cat], alpha=0.8, s=52, zorder=3,
            edgecolor="black", linewidth=0.5,
            label=label,
        )

    ax.boxplot(
        [valid["cos_p1_pN"].values], positions=[0], widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="none", linewidth=1),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        patch_artist=True, zorder=2,
    )

    # y-axis: truncated to [0.85, 1.00], ticks every 0.05
    ax.set_ylim(0.85, 1.00)
    ax.set_yticks(np.arange(0.85, 1.001, 0.05))
    ax.set_xlim(-0.6, 0.6)
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":")

    ax.set_xticks([0])
    ax.set_xticklabels(["all 30 words"], fontsize=11)
    ax.set_ylabel(
        "cosine(page 1 color vector, page-N color vector)",
        fontsize=11,
    )
    ax.set_title(
        "Page 1 vs deepest reachable page (target = page 10)\n"
        "30 frequency-random common nouns",
        fontsize=12,
    )
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=10, frameon=False,
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, "page_compare.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
