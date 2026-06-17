"""
Fig 6 — Lowell − Johnson Δ per base color.

Error bars: ± SD from the permutation null distribution (B = 10,000
size-matched partitions). Δ values at α = 1.0 match Table tab:color_usage.

Data source: csv/revision_results/permutation_null.csv
Output: Fig 6.pdf (draft + submit copies).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(
    "/Users/qgroup/Desktop/word_color_association-main/"
    "csv/revision_results/permutation_null.csv"
)
OUT_DRAFT = Path("/Users/qgroup/Desktop/[word-color-association]final_draft/Fig 6.pdf")
OUT_SUBMIT = Path(
    "/Users/qgroup/Desktop/PLOS_Submission_Final/01_For_Upload/B_Figures/Fig 6.pdf"
)

ORDER = ["red", "orange", "yellow", "green", "blue", "purple",
         "pink", "black", "grey", "white"]

# Bar colors — solid base colors, with light grey/white needing visible edges.
FACE = {
    "red":    "#e23030",
    "orange": "#f08020",
    "yellow": "#f0d020",
    "green":  "#28a028",
    "blue":   "#2050d0",
    "purple": "#7b3aa6",
    "pink":   "#f0a8c0",
    "black":  "#0a0a0a",
    "grey":   "#a0a0a0",
    "white":  "#ffffff",
}
EDGE = {c: "#000000" for c in ORDER}
EDGE["white"] = "#000000"
EDGE["grey"] = "#404040"


def main():
    df = pd.read_csv(DATA).set_index("color").loc[ORDER]
    delta = df["observed_delta"].to_numpy()
    nullsd = df["null_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(ORDER))
    bars = ax.bar(
        x, delta,
        yerr=nullsd, capsize=4,
        color=[FACE[c] for c in ORDER],
        edgecolor=[EDGE[c] for c in ORDER],
        linewidth=1.0,
        error_kw=dict(ecolor="#1a1a1a", lw=1.0, capthick=1.0),
    )

    ax.axhline(0, color="#000000", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["red", "orange", "yellow", "green", "blue", "purple",
         "pink", "black", "gray", "white"],
        fontsize=12,
    )
    ax.set_ylabel(
        r"Difference in color usage fraction (AL $-$ GJ) $\pm$ permutation null SD",
        fontsize=12,
    )
    ax.tick_params(axis="y", labelsize=11)
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
