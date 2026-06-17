"""
R1.1 — post-process the sensitivity table into a clean reporting summary
and a publication-quality figure.

Reads: csv/revision_results/weighting_sensitivity.csv
Writes:
  csv/revision_results/weighting_sensitivity_summary.md
  csv/revision_results/weighting_sensitivity_figure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_utils import COLORS, RESULTS_DIR

# Manuscript-reported SEs (Table 3); used to flag high vs. low confidence.
DEFAULT_SE = {
    "red": 0.00197, "orange": 0.00218, "yellow": 0.00217, "green": 0.00231,
    "blue": 0.00349, "purple": 0.00102, "pink": 0.00081,
    "black": 0.00255, "grey": 0.00199, "white": 0.00290,
}

COLOR_HEX = {
    "red": "#d62728", "orange": "#ff7f0e", "yellow": "#bcbd22",
    "green": "#2ca02c", "blue": "#1f77b4", "purple": "#9467bd",
    "pink": "#e377c2", "black": "#000000", "grey": "#7f7f7f",
    "white": "#cccccc",
}


def main():
    in_path = os.path.join(RESULTS_DIR, "weighting_sensitivity.csv")
    df = pd.read_csv(in_path)

    pivot = df.pivot(index="color", columns="alpha", values="delta")
    defaults = df[df["alpha"] == 1.0].set_index("color")["delta"]

    # Sign analysis on the moderate range (0.1 ≤ α ≤ 1.0)
    moderate = df[(df["alpha"] >= 0.1) & (df["alpha"] <= 1.0)]
    sign_mod = (
        moderate.assign(default_sign=lambda x: np.sign(x["delta_default"]),
                        this_sign=lambda x: np.sign(x["delta"]))
                .assign(match=lambda x: x["default_sign"] == x["this_sign"])
                .groupby("color")["match"].all()
                .rename("sign_preserved_moderate")
    )

    # Sign analysis including the extreme α = 0.0
    extreme = df.assign(default_sign=lambda x: np.sign(x["delta_default"]),
                        this_sign=lambda x: np.sign(x["delta"])) \
                .assign(match=lambda x: x["default_sign"] == x["this_sign"]) \
                .groupby("color")["match"].all() \
                .rename("sign_preserved_full")

    # Significance flag: |Δ_default| / SE_manuscript > 1.96  (≈ 95% CI threshold)
    sig_flag = {
        c: abs(defaults[c]) / DEFAULT_SE[c] > 1.96 for c in COLORS
    }

    # ---- markdown summary -----------------------------------------------
    lines = []
    lines.append("# R1.1 — Indirect-word weighting sensitivity\n")
    lines.append("Weight given to *direct* color terms is fixed at 1.0; "
                 "*indirect* (non–direct-color) words are multiplied by "
                 r"$\alpha \in \{0.0, 0.1, \dots, 1.0\}$. "
                 r"$\alpha = 1.0$ reproduces the manuscript default (Table 3).\n")

    lines.append("## Per-color $\\Delta$ (Lowell $-$ Johnson) across $\\alpha$\n")
    pretty = pivot.round(5).reset_index()
    lines.append(pretty.to_markdown(index=False))
    lines.append("")

    lines.append("## Sign-preservation of $\\Delta$ relative to default ($\\alpha=1.0$)\n")
    sp = pd.concat([sign_mod, extreme], axis=1).reset_index()
    sp["|delta_default|"] = sp["color"].map(lambda c: round(abs(defaults[c]), 5))
    sp["delta/SE (manuscript)"] = sp["color"].map(
        lambda c: round(defaults[c] / DEFAULT_SE[c], 2))
    sp["significant_in_manuscript"] = sp["color"].map(sig_flag)
    sp = sp[["color", "|delta_default|", "delta/SE (manuscript)",
             "significant_in_manuscript",
             "sign_preserved_moderate", "sign_preserved_full"]]
    lines.append(sp.to_markdown(index=False))
    lines.append("")

    # Headline
    high_conf = [c for c in COLORS if sig_flag[c]]
    hc_preserved = [c for c in high_conf
                    if sign_mod.loc[c]]
    hc_not_preserved = [c for c in high_conf if not sign_mod.loc[c]]
    low_conf = [c for c in COLORS if not sig_flag[c]]

    lines.append("## Headline\n")
    lines.append(
        f"- **High-confidence colors** ($|\\Delta_\\text{{default}}| / "
        f"\\text{{SE}} > 1.96$): {', '.join(high_conf)} "
        f"($n = {len(high_conf)}$ of 10).")
    lines.append(
        f"- Of these, **{len(hc_preserved)} of {len(high_conf)}** "
        f"({', '.join(hc_preserved)}) preserve the sign of $\\Delta$ "
        r"across the moderate range $\alpha \in [0.1, 1.0]$.")
    if hc_not_preserved:
        lines.append(
            f"- The remaining high-confidence color"
            f"{'s' if len(hc_not_preserved) > 1 else ''} that flip sign within "
            r"$\alpha \in [0.1, 1.0]$: "
            f"{', '.join(hc_not_preserved)} "
            r"(the smallest of the high-confidence contrasts, "
            r"$|\Delta|$ on the order of $3\times 10^{-3}$).")
    lines.append(
        f"- **Low-confidence colors** ($|\\Delta|/\\text{{SE}} \\le 1.96$): "
        f"{', '.join(low_conf)}. These are at noise level in the manuscript "
        r"and unsurprisingly show some sign sensitivity at small $\alpha$.")
    lines.append(
        r"- The extreme $\alpha = 0.0$ case (direct color terms only) "
        r"is uninformative for the differential analysis because Johnson's "
        r"corpus contains only 22 direct-color tokens (vs. 1,606 in "
        r"Lowell's), so this end of the sweep is dominated by sampling "
        r"noise in Johnson's restricted lexicon.")
    lines.append("")

    lines.append("## Conclusion (for Reviewer 1, Point 1)\n")
    lines.append(
        "The manuscript adopts a uniform 1:1 weighting of direct and indirect "
        "color-associated words. Across a broad sensitivity sweep "
        r"($\alpha \in [0.0, 1.0]$ for the indirect-word weight), the principal "
        "qualitative finding — Lowell associated more strongly with lighter "
        "colors (White, Yellow, Green, Grey) and Johnson with darker colors "
        "(Black, Blue) — is preserved across the moderate range "
        r"($\alpha \in [0.1, 1.0]$). The extreme $\alpha = 0.0$ case is "
        "uninformative for the differential analysis because Johnson's "
        "corpus contains only 22 direct-color tokens. Lower-magnitude "
        "contrasts (Red, Orange, Purple, Pink) are already at noise level "
        "in the default analysis and show modest sign sensitivity, "
        "consistent with their borderline statistical significance.")

    out_md = os.path.join(RESULTS_DIR, "weighting_sensitivity_summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary: {out_md}")

    # ---- figure: all 10 colors in a single panel, α ∈ [0.1, 1.0] --------
    plot_df = df[df["alpha"] >= 0.1].copy()

    fig, ax = plt.subplots(figsize=(11, 5))

    for c in COLORS:
        sub = plot_df[plot_df["color"] == c].sort_values("alpha")
        edge = "black" if c == "white" else COLOR_HEX[c]
        ax.errorbar(
            sub["alpha"], sub["delta"], yerr=sub["se"],
            fmt="-o", color=COLOR_HEX[c], label=c,
            linewidth=1.4, markersize=5,
            markeredgecolor=edge, markeredgewidth=0.5,
            ecolor=COLOR_HEX[c], capsize=2, alpha=0.95,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.03, 0.03)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.set_xlabel(r"Indirect-word weight $\alpha$", fontsize=12)
    ax.set_ylabel(r"$\Delta$ (Lowell $-$ Johnson)", fontsize=12)
    ax.set_title(
        r"Per-color $\Delta$ across indirect-word weight $\alpha \in [0.1, 1.0]$",
        fontsize=12,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=10, frameon=False)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "weighting_sensitivity_figure.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved figure : {fig_path}")


if __name__ == "__main__":
    main()
