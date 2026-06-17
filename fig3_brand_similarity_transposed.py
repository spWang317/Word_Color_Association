"""
Fig 3 — sorted-leaderboard scatter of the top 100 publicly traded companies
by logo–web-image color cosine similarity.

Layout:
  - y-axis: company rank (1 at top = highest cosine; 100 at bottom = lowest)
  - x-axis: cosine similarity (low at left, high at right)
  - Dots colored by extracted dominant brand color of the official logo
  - All 100 companies labeled with their full registered name
    (rank <= 50: label to the left of the dot; rank > 50: label to the right;
    short connector lines, labels allowed inside the plot area)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("/Users/qgroup/Desktop/word_color_association-main/company_data")
OUT_DRAFT = Path(
    "/Users/qgroup/Desktop/[word-color-association]final_draft/Fig 3.pdf"
)
OUT_SUBMIT = Path(
    "/Users/qgroup/Desktop/PLOS_Submission_Final/01_For_Upload/B_Figures/Fig 3.pdf"
)

CHROM = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]
ACHROM = ["black", "grey", "white"]

HEX = {
    "red": "#d62728", "orange": "#ff7f0e", "yellow": "#e6c200",
    "green": "#2ca02c", "blue": "#1f77b4", "purple": "#9467bd",
    "pink": "#e377c2", "black": "#1a1a1a", "grey": "#888888",
    "white": "#f5f5f5",
}

OFFSET = 0.015  # tiny connector length in x-units
FONT_LABEL = 8.5


def dominant(row, min_chrom=0.05):
    chrom = {c: row[c] for c in CHROM}
    if max(chrom.values()) >= min_chrom:
        return max(chrom, key=chrom.get)
    ach = {c: row[c] for c in ACHROM}
    return max(ach, key=ach.get)


def main():
    vec = pd.read_csv(DATA / "company_vectors.csv")
    sim = pd.read_csv(DATA / "company_cosine.csv")
    logos = vec[vec["side"] == "logo"].set_index("company")
    df = sim.merge(
        logos.apply(dominant, axis=1).rename("dom"),
        left_on="company", right_index=True,
    )
    df = df.sort_values("cosine", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(10.24, 16))

    median = float(df["cosine"].median())
    q1, q3 = df["cosine"].quantile([0.25, 0.75])
    ax.axvspan(q1, q3, color="#cccccc", alpha=0.28, zorder=1,
               label=f"IQR  [{q1:.2f}, {q3:.2f}]")
    ax.axvline(median, color="#444444", linestyle="--", linewidth=1.0,
               zorder=2, label=f"Median  {median:.2f}")

    for _, r in df.iterrows():
        face = HEX[r["dom"]]
        edge = "#202020" if r["dom"] != "white" else "#404040"
        ax.scatter(
            r["cosine"], r["rank"],
            s=80, color=face, edgecolor=edge, linewidth=0.6,
            alpha=0.95, zorder=3,
        )

    for _, r in df.iterrows():
        name = r["company"]
        cos = r["cosine"]
        rank = r["rank"]
        if rank <= 50:
            ax.annotate(
                name,
                xy=(cos, rank),
                xytext=(cos - OFFSET, rank),
                fontsize=FONT_LABEL, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.4),
                zorder=4,
            )
        else:
            ax.annotate(
                name,
                xy=(cos, rank),
                xytext=(cos + OFFSET, rank),
                fontsize=FONT_LABEL, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.4),
                zorder=4,
            )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(101, 0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xticklabels(
        ["0" if abs(t) < 1e-9 else f"{t:.1f}"
         for t in np.arange(0.0, 1.01, 0.1)]
    )
    ax.set_yticks([1, 20, 40, 60, 80, 100])

    ax.set_xlabel(
        "Color similarity (cosine) between official logo and web-image color vectors",
        fontsize=11,
    )
    ax.set_ylabel(
        "Company rank (1 = highest cosine similarity, 100 = lowest)",
        fontsize=11,
    )
    ax.set_title(
        "Logo vs. web-image color similarity for the world's 100 largest publicly traded companies\n"
        "(dot color = extracted dominant color of each official logo)",
        fontsize=12,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DRAFT, bbox_inches="tight")
    plt.savefig(str(OUT_DRAFT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.savefig(OUT_SUBMIT, bbox_inches="tight")
    plt.savefig(str(OUT_SUBMIT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved (draft):  {OUT_DRAFT}")
    print(f"Saved (submit): {OUT_SUBMIT}")


if __name__ == "__main__":
    main()
