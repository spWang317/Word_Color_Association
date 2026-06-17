"""
R1.1 — Sensitivity of the Lowell-vs.-Johnson color contrast to the relative
weight given to *indirect* (non–direct-color) words.

Default in the manuscript: indirect_weight = 1.0 (1:1 BoW, no special
treatment of direct color terms).

Here we sweep indirect_weight over alpha = {0.0, 0.1, ..., 1.0} and report
- Δ(AL − GJ) per color
- bootstrap SE of Δ (B = 10,000 resamples)
- sign-preservation flag (does Δ keep the same sign as α = 1.0 default?)

Outputs:
  csv/revision_results/weighting_sensitivity.csv
  csv/revision_results/weighting_sensitivity.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_utils import (
    COLORS, DIRECT_COLOR_WORDS, RESULTS_DIR,
    color_vector, load_all,
)

# ---- bootstrap with optional indirect-word weight -------------------------

def bootstrap_delta_weighted(
    al_bow, gdj_bow, ratio_master,
    indirect_weight: float = 1.0,
    B: int = 10000,
    seed: int = 123,
):
    """Return (delta_orig, delta_se) under the given indirect weight."""
    rng_al = np.random.default_rng(seed)
    rng_gdj = np.random.default_rng(seed + 1)

    def _boots(bow, rng):
        words = np.array(list(bow.keys()))
        freqs = np.array([bow[w] for w in words], dtype=int)
        p = freqs / freqs.sum()
        N = int(freqs.sum())
        boots = np.empty((B, len(COLORS)), dtype=float)
        # Pre-compute per-word weight and per-word color row for speed
        wts = np.array(
            [1.0 if w in DIRECT_COLOR_WORDS else indirect_weight for w in words]
        )
        # Color matrix: rows = words (filtered to those in ratio_master); cols = colors
        # Words that are not in ratio_master would have been filtered out already.
        cmat = np.zeros((len(words), len(COLORS)), dtype=float)
        for i, w in enumerate(words):
            rmap = ratio_master.get(w, {})
            for j, c in enumerate(COLORS):
                cmat[i, j] = float(rmap.get(c, 0.0))
        for b in range(B):
            idx = rng.choice(len(words), size=N, replace=True, p=p)
            cnts = np.bincount(idx, minlength=len(words)).astype(float)
            eff = cnts * wts  # effective counts after weighting
            N_eff = eff.sum()
            if N_eff == 0:
                boots[b] = 0.0
            else:
                boots[b] = (eff[:, None] * cmat).sum(axis=0) / N_eff
        return boots

    boots_al = _boots(al_bow, rng_al)
    boots_gdj = _boots(gdj_bow, rng_gdj)

    delta_orig = (
        color_vector(al_bow, ratio_master, indirect_weight=indirect_weight)
        - color_vector(gdj_bow, ratio_master, indirect_weight=indirect_weight)
    )
    delta_samples = boots_al - boots_gdj
    delta_se = delta_samples.std(axis=0, ddof=1)
    return delta_orig, delta_se


# ---- main -----------------------------------------------------------------

def main(B: int = 10000, seed: int = 123):
    d = load_all()
    al_bow = d["al_bow"]
    gdj_bow = d["gdj_bow"]
    ratio_master = d["ratio_master"]

    alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)

    rows = []
    deltas_by_alpha = {}
    se_by_alpha = {}

    # baseline (default) — first for sign reference
    base_delta, base_se = bootstrap_delta_weighted(
        al_bow, gdj_bow, ratio_master,
        indirect_weight=1.0, B=B, seed=seed,
    )
    base_sign = np.sign(base_delta)
    deltas_by_alpha[1.0] = base_delta
    se_by_alpha[1.0] = base_se

    for alpha in alphas:
        if float(alpha) == 1.0:
            delta, se = base_delta, base_se
        else:
            print(f"[alpha = {alpha:.2f}] bootstrapping (B = {B})...")
            delta, se = bootstrap_delta_weighted(
                al_bow, gdj_bow, ratio_master,
                indirect_weight=float(alpha), B=B, seed=seed,
            )
            deltas_by_alpha[float(alpha)] = delta
            se_by_alpha[float(alpha)] = se

        sign_preserved = np.sign(delta) == base_sign
        for j, c in enumerate(COLORS):
            rows.append({
                "alpha": float(alpha),
                "color": c,
                "delta": float(delta[j]),
                "se": float(se[j]),
                "delta_default": float(base_delta[j]),
                "sign_preserved": bool(sign_preserved[j]),
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "weighting_sensitivity.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved table: {out_path}")

    # Summary
    print("\n=== Sign preservation across alpha (excluding alpha = 1.0 baseline) ===")
    summ = (out[out["alpha"] != 1.0]
            .groupby("color")["sign_preserved"]
            .all()
            .reset_index()
            .rename(columns={"sign_preserved": "all_alphas_preserve_sign"}))
    print(summ.to_string(index=False))

    # Plot: each color as a line over alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    color_hex = {
        "red": "#d62728", "orange": "#ff7f0e", "yellow": "#bcbd22",
        "green": "#2ca02c", "blue": "#1f77b4", "purple": "#9467bd",
        "pink": "#e377c2", "black": "#000000", "grey": "#7f7f7f",
        "white": "#bbbbbb",
    }
    for c in COLORS:
        ys = [deltas_by_alpha[a][COLORS.index(c)] for a in sorted(deltas_by_alpha)]
        xs = sorted(deltas_by_alpha)
        ax.plot(xs, ys, "-o", color=color_hex[c], label=c, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Indirect-word weight $\alpha$ "
                  r"(1.0 = manuscript default, 0.0 = direct color terms only)",
                  fontsize=12)
    ax.set_ylabel(r"$\Delta$ (Lowell $-$ Johnson) per color", fontsize=12)
    ax.set_title("R1.1 — sensitivity of per-color contrast to indirect-word weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "weighting_sensitivity.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    return out, summ


if __name__ == "__main__":
    out, summ = main(B=10000, seed=123)
