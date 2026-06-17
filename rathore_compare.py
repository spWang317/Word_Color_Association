"""
Extended Rathore comparison: 12 fruits (UW-58 ratings) + 6 materials (BCP-37
ratings) = 18 concepts.

Both datasets are mapped from their native color set to our 10 base colors
using the same chromatic-threshold + hue/light band logic, then aggregated
per-concept and compared to our framework's output.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from revision_utils import RESULTS_DIR
from color_utils import (
    COLORS, HUE_RANGES, LIGHT_RANGES, CHROMA_THRESHOLD, SEED,
    color_vector_from_pixels, resize_to_pixel_count,
)

DATA_DIR = "external_compare/rathore2019"
SCREENSHOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Images", "rathore_raw"
)
CORPUS_CSVS = [
    "csv/processed_csv/al_df_with_ratios.csv",
    "csv/processed_csv/gdj_df_with_ratios.csv",
]


def build_corpus_lookup() -> dict[str, np.ndarray]:
    """Aggregate {word: 10-d color vector} from the manuscript-pipeline
    per-word color ratios stored in raw_ratio_content_words. Averaged across
    all per-poem occurrences (same scrape source, so mean = single value
    when consistent)."""
    accum = defaultdict(lambda: np.zeros(10))
    counts = defaultdict(int)
    for path in CORPUS_CSVS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "raw_ratio_content_words" not in df.columns:
            continue
        for raw in df["raw_ratio_content_words"].dropna():
            try:
                d = ast.literal_eval(raw)
            except Exception:
                continue
            for w, cd in d.items():
                if not isinstance(cd, dict):
                    continue
                vec = np.array([cd.get(c, 0.0) for c in COLORS])
                accum[w.lower()] += vec
                counts[w.lower()] += 1
    return {w: accum[w] / counts[w] for w in accum}


def color_set_to_base10(color_df: pd.DataFrame) -> np.ndarray:
    """Each row of color_df has L, c, h. Return idx into COLORS for each."""
    idx = np.full(len(color_df), -1, dtype=int)
    for i, row in color_df.iterrows():
        c, h, L = row["c"], row["h"], row["L"]
        if c >= CHROMA_THRESHOLD:
            hmod = h % 360
            for j, color in enumerate(COLORS[:7]):
                for lo, hi in HUE_RANGES[color]:
                    if lo <= hmod <= hi:
                        idx[i] = j
                        break
                if idx[i] >= 0:
                    break
        else:
            for j, color in enumerate(["black", "grey", "white"], start=7):
                lo, hi = LIGHT_RANGES[color]
                if lo <= L <= hi:
                    idx[i] = j
                    break
    return idx


def aggregate(ratings: pd.DataFrame, color_col: str, base10: np.ndarray,
              concepts: list[str]) -> pd.DataFrame:
    out = np.zeros((len(concepts), 10))
    for i, c in enumerate(concepts):
        sub = ratings[ratings["Concept"].str.lower() == c.lower()]
        n_colors = len(base10)
        rating_arr = np.zeros(n_colors)
        for _, r in sub.iterrows():
            ci = int(r[color_col]) - 1
            rating_arr[ci] = r["True Rating"]
        for ci in range(n_colors):
            bi = base10[ci]
            if bi >= 0:
                out[i, bi] += rating_arr[ci]
        ch, ac = out[i, :7].sum(), out[i, 7:].sum()
        if ch > 0: out[i, :7] /= ch
        if ac > 0: out[i, 7:] /= ac
    return pd.DataFrame(out, index=concepts, columns=COLORS)


def our_vec_for(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    resized = resize_to_pixel_count(img)
    arr = np.array(resized).reshape(-1, 3)
    return color_vector_from_pixels(arr, np.random.default_rng(SEED))


def main():
    # ---- Load both Rathore color sets and rating tables ----
    uw58 = pd.read_csv(os.path.join(DATA_DIR, "UW58_Colors.csv")).reset_index(drop=True)
    bcp37 = pd.read_csv(os.path.join(DATA_DIR, "BCP37_Colors.csv")).reset_index(drop=True)
    uw_base = color_set_to_base10(uw58)
    bcp_base = color_set_to_base10(bcp37)
    print(f"UW58 → 10-base mapping: {(uw_base >= 0).sum()}/58 mapped")
    print(f"BCP37 → 10-base mapping: {(bcp_base >= 0).sum()}/37 mapped")

    fruits = [
        "Avocado", "Blueberry", "Cantaloupe", "Grapefruit", "Honeydew",
        "Lemon", "Lime", "Mango", "Orange", "Raspberry", "Strawberry",
        "Watermelon",
    ]
    materials = ["Compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]
    all_concepts = fruits + materials

    fruit_ratings = pd.read_csv(os.path.join(DATA_DIR, "RatingsAllMethods.csv"))
    mat_ratings = pd.read_csv(os.path.join(DATA_DIR, "RatingsTestConcepts.csv"))

    human_fruits = aggregate(fruit_ratings, "UW-58 Colors", uw_base, fruits)
    human_mats = aggregate(mat_ratings, "BCP-37 colors", bcp_base, materials)
    human = pd.concat([human_fruits, human_mats])
    print(f"\n=== Human-aggregated 10-d distribution (18 concepts) ===")
    print(human.round(3).to_string())

    # ---- Our framework 10-d: prefer existing corpus value, fallback to screenshot ----
    corpus = build_corpus_lookup()
    print(f"\nCorpus lookup built: {len(corpus):,} unique words.")
    ours_rows = {}
    source = {}
    for c in all_concepts:
        cl = c.lower()
        if cl in corpus:
            ours_rows[c] = corpus[cl]
            source[c] = "corpus"
        else:
            path = os.path.join(SCREENSHOT_DIR, f"{c}.png")
            if not os.path.exists(path):
                print(f"  MISSING screenshot for {c}")
                continue
            ours_rows[c] = our_vec_for(path)
            source[c] = "fresh_scrape"
    ours = pd.DataFrame(ours_rows, index=COLORS).T
    print(f"\n=== Our framework 10-d (n={len(ours)}) ===")
    print(ours.round(3).to_string())
    print(f"\nSource per concept:")
    for c in all_concepts:
        if c in source:
            print(f"  {c:15s} → {source[c]}")

    # ---- Per-concept comparison ----
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    rows = []
    for c in all_concepts:
        if c not in ours.index:
            continue
        kind = "fruit" if c in fruits else "material"
        h = human.loc[c].values
        o = ours.loc[c].values
        if h.std() == 0 or o.std() == 0:
            pear, spear = np.nan, np.nan
        else:
            pear = float(pearsonr(h, o).statistic)
            spear = float(spearmanr(h, o).statistic)
        rows.append({
            "concept": c, "kind": kind, "source": source.get(c, "?"),
            "top_human": COLORS[int(np.argmax(h))],
            "top_ours": COLORS[int(np.argmax(o))],
            "top1_match": int(np.argmax(h) == np.argmax(o)),
            "cos_10d": cos(h, o),
            "cos_chrom7": cos(h[:7], o[:7]),
            "cos_achrom3": cos(h[7:], o[7:]),
            "pearson_10d": pear,
            "spearman_10d": spear,
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "rathore_compare.csv"), index=False)

    print("\n=== Per-concept results (18) ===")
    print(res[["concept", "kind", "top_human", "top_ours", "top1_match",
               "cos_10d", "cos_chrom7", "cos_achrom3",
               "pearson_10d", "spearman_10d"]].to_string(index=False))

    print("\n=== Aggregate by kind ===")
    for metric in ["cos_10d", "cos_chrom7", "cos_achrom3",
                   "pearson_10d", "spearman_10d", "top1_match"]:
        full = res[metric].mean()
        fruit_m = res[res["kind"] == "fruit"][metric].mean()
        mat_m = res[res["kind"] == "material"][metric].mean()
        print(f"  {metric:18s}  all={full:.4f}  fruits={fruit_m:.4f}  "
              f"materials={mat_m:.4f}")

    # ---- summary md ----
    lines = [
        "# Rathore et al. (2019) human ratings vs our framework — 18-concept extension",
        "",
        "**Two complementary Rathore rating tables combined**:",
        "",
        "- *RatingsAllMethods.csv* — 12 fruit concepts × 58 UW-58 colors × 54 "
        "participants → mean True Rating.",
        "- *RatingsTestConcepts.csv* — 6 material/recycling concepts (Compost, "
        "Glass, Metal, Paper, Plastic, Trash) × 37 BCP-37 colors.",
        "",
        "**Color-set → 10-base mapping**: Each row of UW-58 / BCP-37 provides "
        "(L, c, h). The same chromatic-threshold + hue/light band rule used by "
        "the manuscript's color_vector_from_pixels assigns each color to one of "
        "our 10 base colors. UW-58 covers all 7 chromatic categories + 3 "
        "achromatic (n_red=3, orange=7, yellow=9, green=9, blue=6, purple=11, "
        "pink=8, black=2, grey=2, white=1). BCP-37 covers the same 10 "
        "categories with smaller per-color counts.",
        "",
        "**Per-concept aggregation**: Human ratings on the native palette are "
        "summed into the 10 base-color bins per concept; chromatic block (7) "
        "and achromatic block (3) are normalized separately to sum 1 — matching "
        "our framework's two-block 10-d output.",
        "",
        "**Our framework's vectors**: Standard pipeline on Google Image "
        "screenshot of each concept (word2image with disambiguating modifiers "
        "where needed: 'orange fruit', 'glass material', 'metal material', "
        "'paper material', 'plastic material').",
        "",
        "## Aggregate results",
        "",
        "| Metric | All 18 | Fruits 12 | Materials 6 |",
        "|---|---:|---:|---:|",
    ]
    for col, label in [
        ("cos_10d", "10-d cosine"),
        ("cos_chrom7", "Chromatic 7-d cosine"),
        ("cos_achrom3", "Achromatic 3-d cosine"),
        ("pearson_10d", "Pearson r"),
        ("spearman_10d", "Spearman ρ"),
        ("top1_match", "Top-1 match rate"),
    ]:
        a = res[col].mean()
        f = res[res["kind"] == "fruit"][col].mean()
        m = res[res["kind"] == "material"][col].mean()
        lines.append(f"| {label} | {a:.4f} | {f:.4f} | {m:.4f} |")

    lines += [
        "",
        "## Per-concept table",
        "",
        "| Concept | kind | top (human) | top (ours) | match | cos 10d | "
        "cos chrom7 | cos achrom3 | Pearson | Spearman |",
        "|---|---|---|---|:-:|---:|---:|---:|---:|---:|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['concept']} | {r['kind']} | {r['top_human']} | "
            f"{r['top_ours']} | {'✓' if r['top1_match'] else '✗'} | "
            f"{r['cos_10d']:.3f} | {r['cos_chrom7']:.3f} | "
            f"{r['cos_achrom3']:.3f} | {r['pearson_10d']:.3f} | "
            f"{r['spearman_10d']:.3f} |"
        )
    with open(os.path.join(RESULTS_DIR, "rathore_compare_summary.md"),
              "w") as f:
        f.write("\n".join(lines))

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(res))
    w = 0.27
    ax.bar(x - w, res["cos_10d"], w, label="10-d cosine", color="#1f77b4")
    ax.bar(x, res["cos_chrom7"], w, label="Chromatic 7-d", color="#d62728")
    ax.bar(x + w, res["cos_achrom3"], w, label="Achromatic 3-d", color="#7f7f7f")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(res["concept"], rotation=45, ha="right")
    # Vertical separator between fruits and materials
    if (res["kind"] == "fruit").any() and (res["kind"] == "material").any():
        sep = (res["kind"] == "fruit").sum() - 0.5
        ax.axvline(sep, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.text(sep / 2, 1.02, "fruits (UW-58)", ha="center", fontsize=10)
        ax.text(sep + (len(res) - sep) / 2, 1.02,
                "materials (BCP-37)", ha="center", fontsize=10)
    ax.set_ylabel("Cosine similarity (human ratings ↔ our framework)")
    ax.set_ylim(-0.1, 1.08)
    ax.set_title(
        f"Rathore et al. (2019) 18 concepts (12 fruits + 6 materials) — per-concept comparison\n"
        f"means: 10-d={res['cos_10d'].mean():.3f}, "
        f"chrom={res['cos_chrom7'].mean():.3f}, "
        f"achrom={res['cos_achrom3'].mean():.3f}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "rathore_compare.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
