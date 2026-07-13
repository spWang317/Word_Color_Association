"""
Shared utilities for the literary corpus analyses.

- Loads the already-processed AL / GDJ CSVs.
- Builds bag-of-words and merged ratio masters.
- Defines the set of *direct* color terms (Table 1 of the manuscript).
"""

from __future__ import annotations

import ast
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----- paths ---------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(ROOT, "csv")
PROCESSED_DIR = os.path.join(CSV_DIR, "processed_csv")
AL_FILE = os.path.join(PROCESSED_DIR, "al_df_with_ratios.csv")
GDJ_FILE = os.path.join(PROCESSED_DIR, "gdj_df_with_ratios.csv")

# Output directory for revision artifacts
RESULTS_DIR = os.path.join(CSV_DIR, "revision_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----- color schema --------------------------------------------------------

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
          "black", "grey", "white"]

# Direct color terms — Table 1 of the manuscript (lowercased, no spaces,
# matching the form stored in the lemmatized BoW).
COLOR_SYNONYMS = {
    "red":    ["red", "scarlet", "vermilion", "ruby", "carmine"],
    "orange": ["orange", "tangerine", "marmalade", "orangish", "apricot"],
    "yellow": ["yellow", "yellowish", "yellowy", "gold", "golden"],
    "green":  ["green", "greenish", "verdant", "leafy", "greenery"],
    "blue":   ["blue", "azure", "cobalt", "cerulean", "ultramarine"],
    "purple": ["purple", "violet", "purply", "purplish", "amethyst"],
    "pink":   ["pink", "rosy", "blushing", "shellpink", "rose"],
    "black":  ["black", "pitchblack", "pitchdark", "jetblack", "blackish"],
    "grey":   ["grey", "silver", "slategray", "smokegray", "silvery"],
    "white":  ["white", "snowywhite", "milkwhite", "milkywhite", "chalkwhite"],
}

DIRECT_COLOR_WORDS = {w for ws in COLOR_SYNONYMS.values() for w in ws}

# ----- I/O helpers ---------------------------------------------------------

def safe_eval(x):
    if isinstance(x, (list, dict)):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def merge_ratio_dicts(ratio_series: pd.Series) -> dict:
    """word -> {color: mean_ratio_over_rows} over a series of per-row dicts."""
    acc = defaultdict(lambda: defaultdict(list))
    for rmap in ratio_series:
        if not isinstance(rmap, dict):
            continue
        for w, cmap in rmap.items():
            if not isinstance(cmap, dict):
                continue
            for c, r in cmap.items():
                acc[w][c].append(float(r))
    return {w: {c: float(np.mean(v)) for c, v in cmap.items()}
            for w, cmap in acc.items()}


def build_bow(df: pd.DataFrame) -> Counter:
    counts = Counter()
    for lst in df["content_words"]:
        if isinstance(lst, list):
            counts.update(lst)
    return counts


def load_poet(file_path: str):
    """Return (df, bow_counter, ratio_master) for one poet CSV."""
    df = pd.read_csv(file_path)
    df["content_words"] = df["content_words"].apply(safe_eval)
    df["raw_ratio_content_words"] = df["raw_ratio_content_words"].apply(safe_eval)
    bow = build_bow(df)
    ratio_master = merge_ratio_dicts(df["raw_ratio_content_words"])
    return df, bow, ratio_master


def union_ratio_masters(*masters: dict) -> dict:
    """Combine multiple word->{color:ratio} masters; average if a word is
    present in more than one source."""
    keys = set().union(*masters)
    out = {}
    for w in keys:
        cmap = defaultdict(list)
        for m in masters:
            if w in m:
                for c, r in m[w].items():
                    cmap[c].append(r)
        out[w] = {c: float(np.mean(v)) for c, v in cmap.items()}
    return out


def filter_bow(bow: Counter, ratio_master: dict) -> Counter:
    """Keep only words for which a color ratio is known."""
    return Counter({w: c for w, c in bow.items() if w in ratio_master})


# ----- weighted color-score computation ------------------------------------

def color_vector(
    bow: Counter,
    ratio_master: dict,
    indirect_weight: float = 1.0,
) -> np.ndarray:
    """Per-token fraction of each base color, with optional down-weighting
    of *indirect* (non–direct-color) words.

    indirect_weight = 1.0 reproduces the manuscript's default 1:1 BoW.
    indirect_weight = 0.0 keeps only the direct-color terms.
    """
    S = np.zeros(len(COLORS), dtype=float)
    N_eff = 0.0
    for w, cnt in bow.items():
        rmap = ratio_master.get(w)
        if not rmap:
            continue
        wt = 1.0 if w in DIRECT_COLOR_WORDS else float(indirect_weight)
        if wt == 0.0:
            continue
        eff = cnt * wt
        N_eff += eff
        for j, c in enumerate(COLORS):
            S[j] += eff * float(rmap.get(c, 0.0))
    if N_eff == 0:
        return np.zeros(len(COLORS), dtype=float)
    return S / N_eff


# ----- convenience entry point --------------------------------------------

def load_all():
    """Load both poets and return everything analyses need."""
    al_df, al_bow, al_ratios = load_poet(AL_FILE)
    gdj_df, gdj_bow, gdj_ratios = load_poet(GDJ_FILE)
    ratio_master = union_ratio_masters(al_ratios, gdj_ratios)
    al_bow_f = filter_bow(al_bow, ratio_master)
    gdj_bow_f = filter_bow(gdj_bow, ratio_master)
    return {
        "al_df": al_df, "gdj_df": gdj_df,
        "al_bow": al_bow_f, "gdj_bow": gdj_bow_f,
        "ratio_master": ratio_master,
    }


if __name__ == "__main__":
    d = load_all()
    print(f"AL  filtered BoW: {len(d['al_bow'])} unique, "
          f"{sum(d['al_bow'].values())} tokens")
    print(f"GDJ filtered BoW: {len(d['gdj_bow'])} unique, "
          f"{sum(d['gdj_bow'].values())} tokens")
    print(f"Ratio master   : {len(d['ratio_master'])} unique words")
    n_direct_al = sum(c for w, c in d["al_bow"].items() if w in DIRECT_COLOR_WORDS)
    n_direct_gdj = sum(c for w, c in d["gdj_bow"].items() if w in DIRECT_COLOR_WORDS)
    print(f"Direct-color tokens in AL : {n_direct_al}")
    print(f"Direct-color tokens in GDJ: {n_direct_gdj}")
    v_al = color_vector(d["al_bow"], d["ratio_master"], indirect_weight=1.0)
    v_gdj = color_vector(d["gdj_bow"], d["ratio_master"], indirect_weight=1.0)
    print("AL  fractions:", dict(zip(COLORS, np.round(v_al, 5))))
    print("GDJ fractions:", dict(zip(COLORS, np.round(v_gdj, 5))))
