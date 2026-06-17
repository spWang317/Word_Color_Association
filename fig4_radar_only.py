"""
Fig 4 — per-company comparison of the 10-d color vectors of the official
logo and of the screenshot of the associated web images, for four illustrative
companies.

Layout per row:
  - Left column: company name + high/low cosine-similarity classification
  - Right column: 10-d radar chart of the logo (solid) and the web images
    (dotted), with the cosine similarity between the two reported above

Output: Fig 4.pdf in both draft and submission folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

DATA = Path("/Users/qgroup/Desktop/word_color_association-main/company_data")
OUT_DRAFT = Path(
    "/Users/qgroup/Desktop/[word-color-association]final_draft/"
    "Fig 4 (radar only preview).pdf"
)
OUT_SUBMIT = Path(
    "/Users/qgroup/Desktop/PLOS_Submission_Final/01_For_Upload/B_Figures/"
    "Fig 4 (radar only preview).pdf"
)

CHROM = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]
ACHROM = ["black", "grey", "white"]
ALL = CHROM + ACHROM

RADAR_ORDER = ["green", "yellow", "orange", "red", "pink",
               "white", "grey", "black", "purple", "blue"]

COMPANIES = [
    ("AT&T Inc.",            "high"),
    ("Caterpillar Inc.",     "high"),
    ("Walt Disney Company",  "low"),
    ("Nestlé S.A.",          "low"),
]


def draw_radar(ax, logo_vec, web_vec, cos):
    angles = np.linspace(0, 2 * np.pi, len(RADAR_ORDER), endpoint=False)
    angles = np.append(angles, angles[:1])

    logo_vals = np.array([logo_vec[c] for c in RADAR_ORDER])
    web_vals = np.array([web_vec[c] for c in RADAR_ORDER])
    logo_vals = np.append(logo_vals, logo_vals[:1])
    web_vals = np.append(web_vals, web_vals[:1])

    sector_colors = {
        "green":  "#cfe8cf", "yellow": "#f4ecbf", "orange": "#f7d8b3",
        "red":    "#f1c2c2", "pink":   "#f3c8de", "white":  "#eeeeee",
        "grey":   "#cfcfcf", "black":  "#8a8a8a", "purple": "#dcc8e8",
        "blue":   "#c4d6ea",
    }
    width = 2 * np.pi / len(RADAR_ORDER)
    for i, c in enumerate(RADAR_ORDER):
        theta = angles[i] - width / 2
        ax.bar(theta + width / 2, 1.0, width=width, bottom=0.0,
               color=sector_colors[c], alpha=0.55, zorder=1,
               edgecolor="none", align="center")

    ax.plot(angles, logo_vals, color="black", linewidth=1.4,
            label="Logo", zorder=3)
    ax.plot(angles, web_vals, color="black", linewidth=1.0, linestyle=":",
            label="Web images", zorder=3)
    ax.fill(angles, logo_vals, color="black", alpha=0.07, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_ORDER, fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7,
                       color="#555555")
    ax.set_ylim(0, 1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.set_title(f"cosine = {cos:.2f}", fontsize=10, pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.30, 1.13),
              fontsize=8, frameon=False)


def main():
    import unicodedata as ud
    vec = pd.read_csv(DATA / "company_vectors.csv")
    sim = pd.read_csv(DATA / "company_cosine.csv")
    vec["company"] = vec["company"].map(lambda s: ud.normalize("NFC", s))
    sim["company"] = sim["company"].map(lambda s: ud.normalize("NFC", s))
    sim = sim.set_index("company")

    fig = plt.figure(figsize=(8, 11))
    gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.10,
                  height_ratios=[1, 1, 1, 1], width_ratios=[1.0, 1.8])

    for row_i, (company, kind) in enumerate(COMPANIES):
        logo_row = vec[(vec["company"] == company) & (vec["side"] == "logo")].iloc[0]
        web_row = vec[(vec["company"] == company) & (vec["side"] == "web")].iloc[0]
        cos = float(sim.loc[company, "cosine"])

        ax_name = fig.add_subplot(gs[row_i, 0])
        ax_name.axis("off")
        ax_name.text(0.0, 0.62, company,
                     fontsize=13, fontweight="bold", ha="left", va="center",
                     transform=ax_name.transAxes)
        ax_name.text(0.0, 0.40,
                     "High logo–web color similarity" if kind == "high"
                     else "Low logo–web color similarity",
                     fontsize=9, color="#666666",
                     ha="left", va="center", transform=ax_name.transAxes)

        ax_radar = fig.add_subplot(gs[row_i, 1], projection="polar")
        logo_vec_dict = {c: logo_row[c] for c in ALL}
        web_vec_dict = {c: web_row[c] for c in ALL}
        draw_radar(ax_radar, logo_vec_dict, web_vec_dict, cos)

    fig.suptitle(
        "Four-company comparison of logo-extracted vs web-image-extracted "
        "color vectors\n"
        "(top two rows: high cosine similarity; bottom two rows: low cosine similarity)",
        fontsize=11, y=0.995,
    )

    plt.savefig(OUT_DRAFT, bbox_inches="tight")
    plt.savefig(str(OUT_DRAFT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.savefig(OUT_SUBMIT, bbox_inches="tight")
    plt.savefig(str(OUT_SUBMIT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved (draft):  {OUT_DRAFT}")
    print(f"Saved (submit): {OUT_SUBMIT}")


if __name__ == "__main__":
    main()
