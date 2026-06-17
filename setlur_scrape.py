"""
Setlur & Stone (2016) pipeline applied to 100 sample words.

Stage 1 (lexical): WordNet color-term lookup over COLOR_LEXICON.
Stage 2 (image): Bing Image Search top-N -> pooled RGB pixels -> k-means
(k=5) -> largest cluster centroid -> nearest entry in COLOR_LEXICON.

Outputs:
  setlur_scrape_results.csv  per-word (word, our_top1, lex_name, lex_bin,
                              img_name, img_bin, img_R, img_G, img_B,
                              setlur_final_name, setlur_final_bin,
                              setlur_stage)
  setlur_scrape_progress.log
"""

from __future__ import annotations

import glob
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from bing_image_downloader import downloader as bing_dl

# Reuse helpers
sys.path.insert(0, str(Path(__file__).parent))
from setlur_full_pipeline import (
    COLOR_LEXICON, BASE_RGB, rgb_to_lch, rgb_to_our_bin, lch_distance,
    nearest_w3c_name, lexical_lookup, NRC_TABLE, SEED, N_SAMPLE, CATS,
)

OUT_CSV = Path("/Users/qgroup/Desktop/[word-color-association]final_draft/setlur_scrape_results.csv")
LOG = Path("/Users/qgroup/Desktop/[word-color-association]final_draft/setlur_scrape_progress.log")
TMP_IMG_DIR = Path("/tmp/setlur_scrape_imgs")
TMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

K = 5             # k-means clusters (Setlur-typical small k)
N_IMAGES = 5      # images per word
TIMEOUT = 8       # per-image download timeout


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def download_images_bing(query: str, n: int = N_IMAGES) -> list[Image.Image]:
    """Download n images for query via bing_image_downloader."""
    safe_query = query.replace("/", "_").replace("\\", "_")
    out_root = TMP_IMG_DIR
    word_dir = out_root / safe_query
    shutil.rmtree(word_dir, ignore_errors=True)
    try:
        bing_dl.download(
            safe_query, limit=n, output_dir=str(out_root),
            adult_filter_off=True, force_replace=False, timeout=20,
            verbose=False,
        )
    except Exception as e:
        log(f"  [{query}] bing_dl failed: {e}")
        return []
    images = []
    for path in sorted(glob.glob(str(word_dir / "*"))):
        try:
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            images.append(img.copy())
            img.close()
        except Exception:
            continue
    shutil.rmtree(word_dir, ignore_errors=True)
    return images


def pool_pixels(images: list[Image.Image], max_per_image: int = 2000) -> np.ndarray:
    """Pool RGB pixels across images, downsampled per image."""
    pool = []
    for img in images:
        # downsample to keep size manageable
        w, h = img.size
        scale = (max_per_image / max(1, w * h)) ** 0.5
        if scale < 1:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.asarray(img.convert("RGB"))
        arr = arr.reshape(-1, 3)
        # filter pure white/black artifacts
        bright = arr.sum(axis=1) > 250 * 3
        dark = arr.sum(axis=1) < 5 * 3
        arr = arr[~(bright | dark)]
        if len(arr) > max_per_image:
            idx = np.random.default_rng(0).choice(len(arr), size=max_per_image, replace=False)
            arr = arr[idx]
        pool.append(arr)
    if not pool:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pool, axis=0)


def setlur_dominant_w3c(pixels: np.ndarray) -> tuple[str | None, tuple[float, float, float] | None]:
    """Setlur image step: k-means on pooled pixels, take centroid of LARGEST
    cluster, snap to nearest named color in COLOR_LEXICON."""
    if len(pixels) < K:
        return None, None
    km = KMeans(n_clusters=K, n_init=4, random_state=0)
    labels = km.fit_predict(pixels.astype(np.float64))
    sizes = np.bincount(labels, minlength=K)
    dom = int(np.argmax(sizes))
    centroid = tuple(km.cluster_centers_[dom])
    name = nearest_w3c_name(centroid)
    return name, centroid


def main():
    df = pd.read_csv(NRC_TABLE)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=N_SAMPLE, replace=False)
    sample = df.iloc[idx].reset_index(drop=True)

    # Resume support: skip words already processed
    done = set()
    if OUT_CSV.exists():
        try:
            done = set(pd.read_csv(OUT_CSV)["word"].tolist())
            log(f"resume: {len(done)} words already done")
        except Exception:
            done = set()

    cols = ["word", "our_top1", "lex_name", "lex_bin",
            "img_name", "img_bin", "img_R", "img_G", "img_B",
            "setlur_final_name", "setlur_final_bin", "setlur_stage"]
    if not OUT_CSV.exists():
        pd.DataFrame(columns=cols).to_csv(OUT_CSV, index=False)

    log(f"START: N={len(sample)} words; already done {len(done)}")
    for i, row in sample.iterrows():
        word = row["word"]
        our_top = row["our_top1"]
        if word in done:
            continue

        # Stage 1: lexical
        lex_name, lex_rgb = lexical_lookup(word)
        if lex_name is not None:
            lex_bin = rgb_to_our_bin(lex_rgb)
            final_name = lex_name
            final_bin = lex_bin
            stage = "lexical"
            img_name = img_bin = ""
            img_rgb = (None, None, None)
        else:
            # Stage 2: image step via Bing.
            images = download_images_bing(word, n=N_IMAGES)
            pixels = pool_pixels(images) if images else np.zeros((0, 3))
            name, centroid = setlur_dominant_w3c(pixels)
            if name is None:
                log(f"  [{i+1}/{len(sample)}] {word}: IMAGE/KMEANS FAIL — fallback grey")
                final_name = "grey"; final_bin = "grey"
                stage = "image_fail"
                img_name = "grey"; img_bin = "grey"; img_rgb = (128, 128, 128)
            else:
                img_name = name
                img_bin = rgb_to_our_bin(COLOR_LEXICON[name])
                img_rgb = centroid
                final_name = name; final_bin = img_bin
                stage = "image"
            lex_bin = ""

        # append row
        rec = {
            "word": word, "our_top1": our_top,
            "lex_name": lex_name or "", "lex_bin": lex_bin,
            "img_name": img_name, "img_bin": img_bin,
            "img_R": img_rgb[0], "img_G": img_rgb[1], "img_B": img_rgb[2],
            "setlur_final_name": final_name, "setlur_final_bin": final_bin,
            "setlur_stage": stage,
        }
        pd.DataFrame([rec]).to_csv(OUT_CSV, mode="a", header=False, index=False)
        log(f"  [{i+1}/{len(sample)}] {word}: stage={stage}, final={final_name} → bin={final_bin} (ours_top1={our_top})")

    log("DONE")


if __name__ == "__main__":
    main()
