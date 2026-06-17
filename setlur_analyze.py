"""
Setlur analysis + S9 Fig generation.

Loads:
  - setlur_scrape_results.csv  (faithful Setlur final color per word)
  - nrc_validation_summary_table.csv  (framework's 10-d ratios per word)

Computes:
  - Full-pipeline top-1 agreement vs framework
  - Per-color one-vs-rest AUC
  - Confusion matrix

Outputs:
  - S9 Fig.pdf (two-panel: AUC bars + confusion matrix)
  - setlur_final_summary.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("/Users/qgroup/Desktop/[word-color-association]final_draft")
SETLUR_CSV = DATA_DIR / "setlur_scrape_results.csv"
NRC_TABLE = DATA_DIR / "SI File" / "csv" / "analyses" / "nrc_validation_summary_table.csv"
OUT_PDF_DRAFT = DATA_DIR / "S9 Fig.pdf"
OUT_PDF_SUBMIT = Path(
    "/Users/qgroup/Desktop/PLOS_Submission_Final/01_For_Upload/B_Figures/S9 Fig.pdf"
)
OUT_MD = DATA_DIR / "setlur_final_summary.md"

CATS = ["red", "orange", "yellow", "green", "blue", "purple",
        "pink", "black", "grey", "white"]

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


def main():
    setlur = pd.read_csv(SETLUR_CSV)
    nrc = pd.read_csv(NRC_TABLE)
    keep = ["word"] + [f"our_{c}" for c in CATS]
    merged = setlur.merge(nrc[keep], on="word", how="left")
    n_total = len(merged)
    n_lex = (merged["setlur_stage"] == "lexical").sum()
    n_img = (merged["setlur_stage"] == "image").sum()
    n_fail = (merged["setlur_stage"] == "image_fail").sum()
    # Drop image-step failures (no images returned) — report as fails
    df = merged[merged["setlur_stage"] != "image_fail"].reset_index(drop=True)
    df["match"] = df["setlur_final_bin"] == df["our_top1"]
    n = len(df)
    full_match = int(df["match"].sum())

    # Per-color AUC
    auc_per_color = {}
    for c in CATS:
        y_true = (df["setlur_final_bin"] == c).astype(int).to_numpy()
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            auc_per_color[c] = None
            continue
        y_score = df[f"our_{c}"].to_numpy()
        auc_per_color[c] = float(roc_auc_score(y_true, y_score))
    aucs = [a for a in auc_per_color.values() if a is not None]
    median_auc = float(np.median(aucs)) if aucs else float("nan")

    # Summary
    lines = [
        "# Setlur full-pipeline reproduction — final results",
        "",
        f"- Initial random sample: {n_total} English words (seed=0)",
        f"- Stage 1 (lexical, WordNet color-term lookup): {n_lex} words",
        f"- Stage 2 (image, Bing top-5 → k-means → W3C nearest): {n_img} words",
        f"- Stage 2 fails (no images returned): {n_fail} words (excluded from analysis)",
        f"- Evaluable sample: N = {n} words",
        "",
        f"- Full-pipeline top-1 agreement with framework: {full_match}/{n} "
        f"({full_match/n*100:.1f}%)",
        f"- Per-color one-vs-rest AUC: median {median_auc:.3f} "
        f"over {len(aucs)} defined colors",
        "",
        "## Per-color AUC",
        "",
        "| color | Setlur positives | AUC |",
        "|---|---|---|",
    ]
    for c in CATS:
        positives = int((df["setlur_final_bin"] == c).sum())
        auc = auc_per_color[c]
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        lines.append(f"| {c} | {positives} | {auc_str} |")
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("")
    mat = pd.crosstab(df["setlur_final_bin"], df["our_top1"]).reindex(
        index=CATS, columns=CATS, fill_value=0)
    lines.append(mat.to_markdown())
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # ---- Figure: single panel — per-color AUC (parallel form to S7 NRC) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    color_hex = {
        "red": "#d62728", "orange": "#ff7f0e", "yellow": "#bcbd22",
        "green": "#2ca02c", "blue": "#1f77b4", "purple": "#9467bd",
        "pink": "#e377c2", "black": "#000000", "grey": "#7f7f7f",
        "white": "#ffffff",
    }
    aucs = [auc_per_color[c] if auc_per_color[c] is not None else 0
            for c in CATS]
    ns = [int((df["setlur_final_bin"] == c).sum()) for c in CATS]
    has_value = [auc_per_color[c] is not None for c in CATS]
    bars = ax.bar(
        CATS, aucs,
        color=[color_hex[c] for c in CATS],
        edgecolor="black", linewidth=0.6,
    )
    # White bar: pure white fill + black border + diagonal hatching for visibility
    white_idx = CATS.index("white")
    bars[white_idx].set_hatch("///")
    bars[white_idx].set_edgecolor("black")
    bars[white_idx].set_linewidth(0.8)
    for b, nv, has in zip(bars, ns, has_value):
        if has:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"n={nv}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.02,
                    f"n={nv}", ha="center", va="bottom", fontsize=8,
                    color="#888888", style="italic")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               label="chance (AUC = 0.5)")
    ax.set_ylabel("AUC (our color-c ratio  vs.  Setlur top = c)", fontsize=11)
    ax.set_xlabel("Base color")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Per-color AUC: framework vs. Setlur full-pipeline labels "
        f"(Setlur & Stone 2016)\n"
        f"(n = {n} evaluable words; median AUC over defined colors = {median_auc:.3f})",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PDF_DRAFT, bbox_inches="tight")
    plt.savefig(str(OUT_PDF_DRAFT).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_PDF_SUBMIT, bbox_inches="tight")
    plt.savefig(str(OUT_PDF_SUBMIT).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", facecolor="white")

    print("=" * 60)
    print(f"FAITHFUL SETLUR RESULTS")
    print(f"  Initial sample: {n_total}")
    print(f"  Stage 1 lexical: {n_lex}")
    print(f"  Stage 2 image (Bing -> k-means -> W3C): {n_img}")
    print(f"  Stage 2 fails (excluded): {n_fail}")
    print(f"  Evaluable N: {n}")
    print(f"  Top-1 agreement: {full_match}/{n} ({full_match/n*100:.1f}%)")
    print(f"  Median per-color AUC: {median_auc:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
