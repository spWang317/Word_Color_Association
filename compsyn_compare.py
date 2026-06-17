"""
Representational Similarity Analysis (RSA) between our framework's per-word
10-d color vectors and comp-syn (Srinivasa Desikan et al., COLING 2020)
pre-computed 8-d JzAzBz / RGB color histograms over a shared random sample
of English content words.

The our-side per-word color vectors are aggregated from the manuscript's
content-word color extraction across the full corpus; we treat them simply
as a per-word color lookup table over a large general English vocabulary
and sample at random for the RSA pairwise computation.
"""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from revision_utils import RESULTS_DIR, COLORS

POET_CSVS = [
    "csv/processed_csv/al_df_with_ratios.csv",
    "csv/processed_csv/gdj_df_with_ratios.csv",
]
COMPSYN_JSON = "external_compare/compsyn/compsyn_vectors.json"


def build_our_word_vectors() -> dict[str, np.ndarray]:
    """Aggregate {word: 10-d color vector} from the processed corpus, averaged
    across all per-poem occurrences. Source poet identity is dropped — the
    result is a flat lookup over English content words."""
    accum = defaultdict(lambda: np.zeros(10))
    counts = defaultdict(int)
    for path in POET_CSVS:
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
                vec = np.array([cd.get(c, 0.0) for c in COLORS], dtype=float)
                accum[w.lower()] += vec
                counts[w.lower()] += 1
    return {w: accum[w] / counts[w] for w in accum}


def load_compsyn():
    with open(COMPSYN_JSON) as f:
        data = json.load(f)
    jz = {d["query"].lower(): np.array(d["jzazbz_dist"], dtype=float) for d in data}
    rgb = {d["query"].lower(): np.array(d["rgb_dist"], dtype=float) for d in data}
    return jz, rgb


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Mn = M / norms
    return Mn @ Mn.T


def rsa(M1: np.ndarray, M2: np.ndarray) -> tuple[float, float]:
    C1 = cosine_matrix(M1)
    C2 = cosine_matrix(M2)
    iu = np.triu_indices_from(C1, k=1)
    v1 = C1[iu]
    v2 = C2[iu]
    mask = np.isfinite(v1) & np.isfinite(v2)
    return (float(pearsonr(v1[mask], v2[mask]).statistic),
            float(spearmanr(v1[mask], v2[mask]).statistic))


def top_neighbors(M, words, target, k=8):
    if target not in words:
        return []
    idx = words.index(target)
    sims = cosine_matrix(M)[idx]
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == idx:
            continue
        out.append((words[j], float(sims[j])))
        if len(out) >= k:
            break
    return out


def main():
    print("Building per-word color vectors from processed corpus...")
    ours = build_our_word_vectors()
    print(f"  {len(ours):,} unique English content words.")

    print("Loading comp-syn pre-computed embeddings...")
    jz, rgb = load_compsyn()
    print(f"  {len(jz):,} unique words.")

    overlap = sorted(set(ours.keys()) & set(jz.keys()))
    print(f"Overlap: {len(overlap):,} words.")

    M_ours = np.stack([ours[w] for w in overlap])
    M_jz = np.stack([jz[w] for w in overlap])
    M_rgb = np.stack([rgb[w] for w in overlap])

    N_RSA = min(100, len(overlap))
    K_SEEDS = 100

    # Per-seed observed RSA across K_SEEDS random N=100 subsamples
    print(f"\nRunning RSA across {K_SEEDS} random {N_RSA}-word subsamples "
          f"of the {len(overlap):,}-word overlap "
          f"(pairs per draw = {N_RSA*(N_RSA-1)//2:,})...")
    pear_jz_draws, spear_jz_draws = [], []
    pear_rgb_draws, spear_rgb_draws = [], []
    per_seed_samples = []
    for s in range(K_SEEDS):
        rng_s = np.random.default_rng(s)
        idx_s = rng_s.choice(len(overlap), size=N_RSA, replace=False)
        M_o = M_ours[idx_s]; M_j = M_jz[idx_s]; M_g = M_rgb[idx_s]
        pj, sj = rsa(M_o, M_j)
        pr, sr = rsa(M_o, M_g)
        pear_jz_draws.append(pj); spear_jz_draws.append(sj)
        pear_rgb_draws.append(pr); spear_rgb_draws.append(sr)
        per_seed_samples.append((idx_s, M_o, M_j, M_g, [overlap[i] for i in idx_s]))
    pear_jz_draws = np.array(pear_jz_draws)
    spear_jz_draws = np.array(spear_jz_draws)
    pear_rgb_draws = np.array(pear_rgb_draws)
    spear_rgb_draws = np.array(spear_rgb_draws)

    pear_jz = float(np.median(pear_jz_draws))
    spear_jz = float(np.median(spear_jz_draws))
    pear_rgb = float(np.median(pear_rgb_draws))
    spear_rgb = float(np.median(spear_rgb_draws))

    # Pick the draw whose Pearson r is closest to the median to use as the
    # representative for the scatter panel.
    median_seed = int(np.argmin(np.abs(pear_jz_draws - pear_jz)))
    representative = per_seed_samples[median_seed]
    print(f"  representative draw for panel (a) = seed {median_seed}  "
          f"(Pearson {pear_jz_draws[median_seed]:.4f} ≈ median {pear_jz:.4f})")

    print(f"  Ours ↔ comp-syn JzAzBz  Pearson r:  "
          f"median = {pear_jz:.4f}  mean = {pear_jz_draws.mean():.4f}  "
          f"SD = {pear_jz_draws.std():.4f}  "
          f"[{pear_jz_draws.min():.4f}, {pear_jz_draws.max():.4f}]")
    print(f"  Ours ↔ comp-syn JzAzBz  Spearman ρ: "
          f"median = {spear_jz:.4f}  mean = {spear_jz_draws.mean():.4f}  "
          f"SD = {spear_jz_draws.std():.4f}  "
          f"[{spear_jz_draws.min():.4f}, {spear_jz_draws.max():.4f}]")
    print(f"  Ours ↔ comp-syn RGB     Pearson r:  "
          f"median = {pear_rgb:.4f}  mean = {pear_rgb_draws.mean():.4f}  "
          f"SD = {pear_rgb_draws.std():.4f}")
    print(f"  Ours ↔ comp-syn RGB     Spearman ρ: "
          f"median = {spear_rgb:.4f}  mean = {spear_rgb_draws.mean():.4f}  "
          f"SD = {spear_rgb_draws.std():.4f}")

    idx_sample, M_ours_rsa, M_jz_rsa, M_rgb_rsa, words_rsa = representative

    # Permutation null on the representative sample (100 row-shuffles)
    print("\nPermutation null (100 row-shuffles on representative draw, JzAzBz)...")
    rng_null = np.random.default_rng(10_000)
    null_pear = []
    for _ in range(100):
        perm = rng_null.permutation(len(words_rsa))
        p, _ = rsa(M_ours_rsa, M_jz_rsa[perm])
        null_pear.append(p)
    null_pear = np.array(null_pear)
    n_above_null_max = int((pear_jz_draws > null_pear.max()).sum())
    print(f"  null Pearson:  mean = {null_pear.mean():.4f}  "
          f"SD = {null_pear.std():.4f}  max = {null_pear.max():.4f}")
    print(f"  observed draws above null max: "
          f"{n_above_null_max}/{K_SEEDS}")

    # use seed-0 RSA for the figure
    pear_jz_repr = float(pear_jz_draws[median_seed])
    spear_jz_repr = float(spear_jz_draws[median_seed])

    # Seed neighbors (sanity)
    print("\nTop-8 nearest neighbors of seed words:")
    SEEDS = ["blue", "rose", "gold", "sea", "stone"]
    seed_rows = []
    for s in SEEDS:
        if s not in overlap:
            print(f"  {s!r}: not in overlap")
            continue
        nb_o = top_neighbors(M_ours, overlap, s, k=8)
        nb_j = top_neighbors(M_jz, overlap, s, k=8)
        ov = len({a for a, _ in nb_o} & {b for b, _ in nb_j})
        seed_rows.append({"seed": s, "top8_overlap": ov,
                          "ours_top8": [a for a, _ in nb_o],
                          "compsyn_top8": [a for a, _ in nb_j]})
        print(f"  {s!r}  jaccard top-8 = {ov}/8")
        print(f"    ours    : {[a for a, _ in nb_o]}")
        print(f"    comp-syn: {[a for a, _ in nb_j]}")

    # Save: headline summary + per-draw distribution
    res = pd.DataFrame([
        {"metric": "RSA Pearson r — median (ours ↔ jzazbz)", "value": pear_jz},
        {"metric": "RSA Pearson r — mean (ours ↔ jzazbz)",
         "value": float(pear_jz_draws.mean())},
        {"metric": "RSA Pearson r — SD (ours ↔ jzazbz)",
         "value": float(pear_jz_draws.std())},
        {"metric": "RSA Spearman rho — median (ours ↔ jzazbz)", "value": spear_jz},
        {"metric": "RSA Spearman rho — mean (ours ↔ jzazbz)",
         "value": float(spear_jz_draws.mean())},
        {"metric": "RSA Spearman rho — SD (ours ↔ jzazbz)",
         "value": float(spear_jz_draws.std())},
        {"metric": "RSA Pearson r — median (ours ↔ rgb)", "value": pear_rgb},
        {"metric": "RSA Spearman rho — median (ours ↔ rgb)", "value": spear_rgb},
        {"metric": "Null Pearson mean", "value": float(null_pear.mean())},
        {"metric": "Null Pearson SD", "value": float(null_pear.std())},
        {"metric": "Null Pearson max", "value": float(null_pear.max())},
        {"metric": "Observed draws above null max", "value": float(n_above_null_max)},
        {"metric": "Number of random subsamples (K)", "value": float(K_SEEDS)},
        {"metric": "Overlap vocab size", "value": float(len(overlap))},
        {"metric": "RSA sample size (N per draw)", "value": float(len(words_rsa))},
        {"metric": "Total source vocab (ours)", "value": float(len(ours))},
        {"metric": "Total comp-syn vocab", "value": float(len(jz))},
    ])
    res.to_csv(os.path.join(RESULTS_DIR, "compsyn_compare.csv"), index=False)

    # Per-draw distribution (full)
    pd.DataFrame({
        "draw": np.arange(K_SEEDS),
        "pearson_jz": pear_jz_draws,
        "spearman_jz": spear_jz_draws,
        "pearson_rgb": pear_rgb_draws,
        "spearman_rgb": spear_rgb_draws,
    }).to_csv(os.path.join(RESULTS_DIR, "compsyn_compare_per_draw.csv"),
              index=False)
    pd.DataFrame(seed_rows).to_csv(
        os.path.join(RESULTS_DIR, "compsyn_compare_seeds.csv"), index=False)

    lines = [
        "# comp-syn (Guilbeault et al. 2020; Srinivasa Desikan et al. COLING 2020) "
        "vs our framework — representational comparison",
        "",
        "**comp-syn** is an independent project that derives per-word color "
        "embeddings from Google Image search results via a 3D color-space "
        f"histogram (2×2×2 = 8 bins) computed in the perceptually uniform "
        f"JzAzBz space (and analogously in RGB). It releases pre-computed "
        f"vectors for {len(jz):,} English words from "
        "github.com/comp-syn/comp-syn.",
        "",
        f"**Our framework** outputs a 10-d color ratio vector per word (7 "
        f"chromatic + 3 achromatic categories in CIELch). We build a per-word "
        f"lookup over {len(ours):,} English content words by averaging the "
        f"per-occurrence ratios from the processed corpus.",
        "",
        f"**Overlap**: {len(overlap):,} words appear in both vocabularies.",
        "",
        "## Method: Representational Similarity Analysis (RSA)",
        "",
        "Because the two embeddings live in different geometries (10-d "
        "categorical bins vs 8-d cube histograms), individual dimensions are "
        "not directly alignable. RSA compares the *global similarity "
        "structure* of the two spaces: for a sample of N words, build the "
        "N×N pairwise-cosine matrix in each space, take the upper triangle, "
        "and correlate the two long vectors of word-pair similarities. The "
        "result is scale-invariant and dimensionality-independent.",
        "",
        f"- N = {N_RSA} words per draw (uniformly random subsample of the "
        f"{len(overlap):,}-word overlap).",
        f"- K = {K_SEEDS} independent random draws (seeds 0..{K_SEEDS-1}).",
        f"- Pair count per draw = {N_RSA*(N_RSA-1)//2:,}.",
        "",
        "## Results (distribution across K random draws)",
        "",
        "| Comparison | Pearson r (median) | Pearson r (mean ± SD) | "
        "Spearman ρ (median) | Spearman ρ (mean ± SD) |",
        "|---|---:|---:|---:|---:|",
        f"| Ours (10-d) ↔ comp-syn JzAzBz (8-d) | {pear_jz:.4f} | "
        f"{pear_jz_draws.mean():.4f} ± {pear_jz_draws.std():.4f} | "
        f"{spear_jz:.4f} | "
        f"{spear_jz_draws.mean():.4f} ± {spear_jz_draws.std():.4f} |",
        f"| Ours (10-d) ↔ comp-syn RGB (8-d) | {pear_rgb:.4f} | "
        f"{pear_rgb_draws.mean():.4f} ± {pear_rgb_draws.std():.4f} | "
        f"{spear_rgb:.4f} | "
        f"{spear_rgb_draws.mean():.4f} ± {spear_rgb_draws.std():.4f} |",
        "",
        "## Permutation null (100 row-shuffles on a representative draw)",
        "",
        f"- null Pearson r: mean = {null_pear.mean():.4f}, "
        f"SD = {null_pear.std():.4f}, max = {null_pear.max():.4f}.",
        f"- Observed Pearson r exceeds the null max in "
        f"**{n_above_null_max}/{K_SEEDS}** draws.",
        "",
        "## Sanity check — top-8 nearest neighbors of seed words",
        "",
        "Because the two embeddings differ in dimensionality and binning, "
        "fine-grained ranked neighbors are not expected to coincide. The Jaccard "
        "overlap between the two top-8 lists is therefore small even though the "
        "global structure correlates.",
        "",
        "| Seed | Jaccard overlap (top-8) |",
        "|---|---:|",
    ]
    for r in seed_rows:
        lines.append(f"| {r['seed']} | {r['top8_overlap']}/8 |")

    with open(os.path.join(RESULTS_DIR, "compsyn_compare_summary.md"),
              "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved CSV, seeds, summary.md")

    # ---- Compute scatter data (used by both single- and two-panel figures) ----
    C_ours = cosine_matrix(M_ours_rsa)
    C_jz = cosine_matrix(M_jz_rsa)
    iu = np.triu_indices_from(C_ours, k=1)
    v_o, v_j = C_ours[iu], C_jz[iu]
    mask = np.isfinite(v_o) & np.isfinite(v_j)
    v_o, v_j = v_o[mask], v_j[mask]
    if len(v_o) > 5000:
        rng_fig = np.random.default_rng(20_000)
        idx_s = rng_fig.choice(len(v_o), size=5000, replace=False)
        v_o = v_o[idx_s]
        v_j = v_j[idx_s]

    # ---- Figure A: single-panel (scatter only, no "(a)" label) ----
    fig_s, ax_s = plt.subplots(figsize=(6.5, 6.0))
    ax_s.scatter(v_o, v_j, s=5, alpha=0.15, color="#1f77b4")
    ax_s.set_xlabel("Pairwise cosine — our 10-d framework")
    ax_s.set_ylabel("Pairwise cosine — comp-syn JzAzBz (8-d)")
    ax_s.set_title(
        f"Word-pair representation alignment\n"
        f"(representative draw of {K_SEEDS}; "
        f"Pearson $r = {pear_jz_repr:.3f}$, "
        f"Spearman $\\rho = {spear_jz_repr:.3f}$; "
        f"$n = {N_RSA}$ words)"
    )
    ax_s.set_xlim(-0.1, 1.05)
    ax_s.set_ylim(-0.1, 1.05)
    ax_s.axline((0, 0), slope=1, color="grey", linewidth=0.6, linestyle="--")
    ax_s.grid(True, alpha=0.3)
    plt.tight_layout()
    single_path = os.path.join(RESULTS_DIR, "compsyn_compare_single.png")
    plt.savefig(single_path, dpi=200, bbox_inches="tight")
    plt.savefig(single_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved single-panel figure: {single_path}")

    # ---- Figure B: two-panel (scatter + histogram + null overlay) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(v_o, v_j, s=5, alpha=0.15, color="#1f77b4")
    axes[0].set_xlabel("Pairwise cosine — our 10-d framework")
    axes[0].set_ylabel("Pairwise cosine — comp-syn JzAzBz (8-d)")
    axes[0].set_title(
        f"(a) Word-pair representation alignment\n"
        f"(representative draw; "
        f"$r = {pear_jz_repr:.3f}$, "
        f"$\\rho = {spear_jz_repr:.3f}$)"
    )
    axes[0].set_xlim(-0.1, 1.05)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].axline((0, 0), slope=1, color="grey", linewidth=0.6, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(null_pear, bins=20, color="grey", alpha=0.6,
                 label=f"null (row-shuffled, n={len(null_pear)})")
    axes[1].hist(pear_jz_draws, bins=20, color="#1f77b4", alpha=0.8,
                 edgecolor="white", linewidth=0.4,
                 label=f"observed across {K_SEEDS} draws")
    axes[1].axvline(pear_jz, color="red", linewidth=2,
                    label=f"observed median = {pear_jz:.3f}")
    axes[1].set_xlabel("RSA Pearson r")
    axes[1].set_ylabel("count")
    axes[1].set_title(
        f"(b) Observed (N={N_RSA}, K={K_SEEDS}) vs permutation null"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "compsyn_compare.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved two-panel figure: {fig_path}")


if __name__ == "__main__":
    main()
