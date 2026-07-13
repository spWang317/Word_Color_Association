"""Polysemy sensitivity metrics and the S2 figure.

Aggregates per-query 10-d color vectors into per-word metrics that
characterize how the recovered color profile depends on query
specificity, then renders the supporting figure.

Per-word metrics
----------------
K : int
    WordNet noun sense count (`len(wn.synsets(word, pos='n'))`).
K_evaluated : int
    Number of senses with a usable disambiguator. Equals the number of
    sense-specific queries actually retrieved.
within_word_spread : float
    Maximum pairwise cosine distance among the sense-specific vectors.
bare_to_nearest_sense : float
    Smallest cosine distance from the bare-token vector to any
    sense-specific vector.
bare_to_mean_sense : float
    Cosine distance from the bare-token vector to the mean of the
    sense-specific vectors.

Outputs
-------
csv/revision_results/polysemy_metrics.csv
figures/polysemy_sensitivity.pdf
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

VECTORS_PATH = Path("csv/revision_results/polysemy_vectors.csv")
METRICS_PATH = Path("csv/revision_results/polysemy_metrics.csv")
FIG_PATH = Path("figures/polysemy_sensitivity.pdf")
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
          "black", "grey", "white"]
COLOR_HEX = {
    "red": "#d62728", "orange": "#ff7f0e", "yellow": "#ffd92f",
    "green": "#2ca02c", "blue": "#1f77b4", "purple": "#7e57c2",
    "pink": "#f4a6c0", "black": "#000000", "grey": "#9e9e9e",
    "white": "#ffffff",
}


def per_word_metrics(group):
    """Compute one row of per-word metrics from a query-grouped DataFrame.

    Returns an empty row (NaN metrics) when the bare-token query is
    missing, so the word can still appear in the output table.
    """
    bare_rows = group[group["synset_id"].fillna("") == ""]
    sense_rows = group[group["synset_id"].fillna("") != ""]
    if bare_rows.empty:
        return pd.Series({
            "K_evaluated": int(len(sense_rows)),
            "within_word_spread": float("nan"),
            "bare_to_nearest_sense": float("nan"),
            "bare_to_mean_sense": float("nan"),
        })
    bare_vec = bare_rows[COLORS].iloc[0].to_numpy(dtype=float)
    sense_mat = sense_rows[COLORS].to_numpy(dtype=float)
    k_eval = sense_mat.shape[0]

    if k_eval >= 2:
        spread = max(cosine(sense_mat[i], sense_mat[j])
                     for i, j in combinations(range(k_eval), 2))
    else:
        spread = float("nan")

    if k_eval >= 1:
        sense_dists = [cosine(bare_vec, sense_mat[i]) for i in range(k_eval)]
        nearest = float(min(sense_dists))
        mean_dist = float(cosine(bare_vec, sense_mat.mean(axis=0)))
    else:
        nearest = float("nan")
        mean_dist = float("nan")

    return pd.Series({
        "K_evaluated": k_eval,
        "within_word_spread": spread,
        "bare_to_nearest_sense": nearest,
        "bare_to_mean_sense": mean_dist,
    })


def attach_total_sense_count(vec_df):
    """Add a column `K` with the total noun sense count from WordNet."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    unique_words = vec_df["query_word"].unique()
    k_map = {w: len(wn.synsets(w, pos="n")) for w in unique_words}
    return vec_df.assign(K=vec_df["query_word"].map(k_map))


def plot_figure(metrics, vec_df, out_path):
    """Render the two-row S2 figure.

    Row 1 (panel a): scatter of within-word spread against K(w). Each
        word label is jittered with a leader line to avoid overlap.
    Row 2 (panel b): one mini-axis per representative word, stacked
        horizontal bars showing the bare-token and sense-disambiguated
        10-d color profiles, each row normalized so the segments sum to
        one and the full chromatic + achromatic distribution is visible.
    """
    valid = metrics.dropna(subset=["within_word_spread"])
    top = (valid.sort_values("within_word_spread", ascending=False)
                .head(5)["query_word"].tolist())
    n_top = max(len(top), 1)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(max(15, 3.4 * n_top), 11.5))
    gs = GridSpec(2, n_top, figure=fig,
                  height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.90,
                  top=0.92, bottom=0.08)

    ax = fig.add_subplot(gs[0, :])
    if len(valid) >= 2:
        rho, p = spearmanr(valid["K"], valid["within_word_spread"])
        title_extra = f"Spearman $\\rho={rho:.2f}$, $p={p:.3g}$, $N={len(valid)}$"
    else:
        title_extra = f"$N={len(valid)}$"
    jittered_x = _jitter_by_K(valid)
    ax.scatter(jittered_x, valid["within_word_spread"],
               s=70, alpha=0.85, edgecolor="black", linewidth=0.6,
               color="#1f77b4", zorder=3)

    from adjustText import adjust_text
    texts = [
        ax.text(x_j, r["within_word_spread"], r["query_word"],
                fontsize=9, color="#333333", ha="center", va="bottom")
        for x_j, (_, r) in zip(jittered_x, valid.iterrows())
    ]
    adjust_text(
        texts, ax=ax,
        only_move={"text": "y", "static": "y", "explode": "y"},
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4),
        expand=(1.1, 1.4),
    )

    ax.set_xlabel("WordNet noun sense count $K(w)$  "
                  "(points within the same $K$ are horizontally jittered for label readability)",
                  fontsize=11)
    ax.set_ylabel("within-word color-vector spread\n(max pairwise cosine distance)",
                  fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    if not valid.empty:
        y_max = float(valid["within_word_spread"].max())
        ax.set_ylim(-0.04, y_max * 1.18)

    for idx, word in enumerate(top):
        ax2 = fig.add_subplot(gs[1, idx])
        sub = vec_df[vec_df["query_word"] == word].copy()
        sub["query_label"] = sub.apply(
            lambda r: ("bare" if pd.isna(r["disambiguator"]) or r["disambiguator"] == ""
                       else f"+{r['disambiguator']}"),
            axis=1,
        )
        sub = sub.reset_index(drop=True)
        raw = sub[COLORS].to_numpy(dtype=float)
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = raw / row_sums

        y_positions = list(range(len(sub)))
        left = np.zeros(len(sub))
        for ci, color in enumerate(COLORS):
            widths = normalized[:, ci]
            ax2.barh(y_positions, widths, left=left,
                     color=COLOR_HEX[color],
                     edgecolor="black" if color == "white" else "none",
                     linewidth=0.35 if color == "white" else 0.0,
                     height=0.74)
            left += widths
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(sub["query_label"].tolist(), fontsize=10)
        ax2.set_xlim(0, 1)
        ax2.invert_yaxis()
        ax2.set_xticks([0, 0.5, 1.0])
        ax2.set_xticklabels(["0", "0.5", "1"], fontsize=9)
        ax2.set_title(word, fontsize=13, fontweight="bold", pad=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="x", labelsize=9)

    title_x = 0.06
    fig.text(title_x, 0.935,
             f"(a) Sensitivity scales with $K(w)$ — {title_extra}",
             fontsize=13, ha="left", color="#222222")
    fig.text(title_x, 0.455,
             "(b) Representative high-spread words: bare vs sense-disambiguated profiles "
             "(each row normalized to sum to one across the ten named base colors)",
             fontsize=12, ha="left", color="#333333")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _jitter_by_K(df):
    """Return a jittered x-coordinate list aligned with df.iterrows().

    Points sharing the same K value are spread horizontally within +-0.35
    of K, ordered by their spread value, so that overlapping labels can
    separate while the integer K position remains visible.
    """
    jitter_amount = 0.35
    result = [None] * len(df)
    for K_val, group in df.groupby("K", sort=False):
        sorted_idx = group["within_word_spread"].sort_values().index.tolist()
        n = len(sorted_idx)
        if n == 1:
            result[df.index.get_loc(sorted_idx[0])] = float(K_val)
            continue
        offsets = np.linspace(-jitter_amount, jitter_amount, n)
        for i, idx in enumerate(sorted_idx):
            result[df.index.get_loc(idx)] = float(K_val) + offsets[i]
    return result


def _annotate_without_overlap(ax, df, x_col, y_col, label_col):
    """Place word labels with leader lines to avoid label-on-label overlap.

    Sorts points by y-value and stacks labels along the right edge,
    drawing a thin leader line from each point to its label position.
    """
    if df.empty:
        return
    points = list(zip(df[x_col].tolist(), df[y_col].tolist(), df[label_col].tolist()))
    points.sort(key=lambda t: t[1])

    x_max = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else float(df[x_col].max())
    x_max = max(x_max, float(df[x_col].max())) * 1.06
    n = len(points)
    y_min = float(df[y_col].min()) - 0.01
    y_max = float(df[y_col].max()) + 0.02
    label_y_positions = np.linspace(y_min, y_max, n) if n > 1 else [points[0][1]]

    for (x, y, label), ly in zip(points, label_y_positions):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x_max, ly),
            fontsize=9, color="#333333",
            va="center", ha="left",
            arrowprops=dict(arrowstyle="-",
                            color="#888888",
                            lw=0.4,
                            connectionstyle="arc3,rad=0.0"),
        )
    ax.set_xlim(ax.get_xlim()[0], x_max * 1.10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default=str(VECTORS_PATH))
    parser.add_argument("--metrics-out", default=str(METRICS_PATH))
    parser.add_argument("--fig-out", default=str(FIG_PATH))
    args = parser.parse_args()

    vec_df = pd.read_csv(args.vectors)
    vec_df = attach_total_sense_count(vec_df)

    metrics = (vec_df.groupby("query_word", sort=False)
                     .apply(per_word_metrics)
                     .reset_index())
    metrics = metrics.merge(
        vec_df[["query_word", "K"]].drop_duplicates(), on="query_word")

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_out, index=False)
    plot_figure(metrics, vec_df, Path(args.fig_out))
    print(f"Wrote {args.metrics_out} and {args.fig_out}")


if __name__ == "__main__":
    main()
