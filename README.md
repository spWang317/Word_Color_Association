# Word–Color Association

Code accompanying the manuscript *Word–Color Association from Large-Scale
Online Data: Methodology and Applications* by Sungpil Wang and Juyong Park.

The framework retrieves word–color associations from publicly accessible
Google Image Search results: it scrapes the top-results screenshot for a
query word, resizes it to 10,000 pixels, converts each pixel to the CIELch
color space, and represents the word as a 10-dimensional additive ratio
vector over named base colors (red, orange, yellow, green, blue, purple,
pink, black, grey, white).

---

## Installation

You can set up the required environment using either `requirements.txt`
(pip) or `environment.yml` (conda).

### Option 1 — pip
```bash
brew install python@3.9
/opt/homebrew/opt/python@3.9/bin/python3.9 -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Option 2 — conda
```bash
conda env create -f environment.yaml
conda activate word-color-association
```

Chrome and chromedriver are required for `word2image.py` (Google Image
Search screenshot via Selenium).

---

## Quick start

Extract the color vector for a single word:
```bash
python main.py --term "moss" --option ratio
```

Visualize the color information of a target word on a local webpage:
```bash
cd website
python app.py
```

---

## Core pipeline

- `word2image.py` — Google Image Search screenshot via Selenium
  (incognito, regional/language presets, light-mode forced via CDP)
- `calculation.py` — pixel-to-CIELch conversion, base-color binning, per-word
  10-d vector computation
- `color_utils.py` — shared color-space utilities (`rgb_to_lch`,
  `color_vector_from_pixels`, `resize_to_pixel_count`, base-color ranges,
  chroma threshold)
- `visualization.py` — per-word color distribution plots
- `main.py` — command-line entry point

### Notebooks
- `Text2Color.ipynb` — extract color vectors per word in long sentences and
  render as text + color-bar visualizations
- `LogoAndScreenshotOfBrands.ipynb` — top-100 publicly traded companies'
  logo vs. web-image color comparison
- `ThresholdOfChroma.ipynb` — chroma distribution of 15 achromatic terms
  and the chroma threshold derivation
- `WhiteGridRatio.ipynb` — white-divider trimming ratio derived from a
  100-screenshot sample (Kolmogorov–Smirnov normality check)
- `ComparisonPoets.ipynb` — Lowell vs. Johnson corpus-level color
  comparison

---

## External validation

Each script produces CSV / PDF / PNG / summary-MD outputs under
`csv/revision_results/`.

- `nrc_validation.py` — NRC Word–Colour Association Lexicon
  (Mohammad 2013): per-color one-vs-rest AUC + confusion matrix
- `rathore_scrape.py`, `rathore_compare.py` — graded slider ratings
  (Rathore et al. 2019, IEEE TVCG): per-concept cosine on the UW-58 /
  BCP-37 palettes
- `compsyn_compare.py` — Representational Similarity Analysis (RSA) vs.
  comp-syn 8-bin JzAzBz histograms (Guilbeault et al. 2020)
- `setlur_scrape.py`, `setlur_full_pipeline.py`, `setlur_analyze.py` —
  Setlur & Stone (2016) full-pipeline reproduction: WordNet color-term
  lookup with Bing-Image-Search dominant-color extraction
  (`k`-means + W3C-nearest)

## Analyses

- `permutation_null.py` — Lowell vs. Johnson Δ permutation null test
  (B = 10,000 size-matched partitions)
- `weighting_sensitivity.py`, `alpha_robustness_report.py` — α-sweep
  robustness of the Lowell–Johnson contrast over indirect-word weights
- `polysemy_scrape.py`, `polysemy_demo.py` — bare-token vs.
  full-registered-name color recovery for polysemous queries
- `scroll_depth_scrape.py`, `page_compare_scrape.py`,
  `page_compare_analyze.py` — page-1 vs. page-10 stability of the
  recovered color vector
- `sample_words.py` — frequency-random English-word sampling pool
- `date_filter_test.py` — Google Images date-range parameter
  (`tbs=cdr:1,cd_min=…,cd_max=…`) sanity check

## Figure regeneration

- `fig3_brand_similarity_transposed.py` — Fig 3 (logo vs. web-image
  cosine, sorted leaderboard with full registered names)
- `fig4_radar_only.py` — Fig 4 (per-company 10-d radar comparison)
- `regen_fig6.py` — Fig 6 (Lowell − Johnson Δ with permutation-null SD
  error bars)
- `make_serp_grids.py` — moss / lilac schematic SERP grids used as
  components of Fig 1 and Fig 2
- `extract_company_vectors.py` — re-extracts the per-company 10-d color
  vectors in `company_data/` from locally re-scraped imagery. The chromatic
  and the achromatic blocks are each normalised to sum to one, as in the
  manuscript. Because the logo and web imagery cannot be redistributed and
  is re-scraped at run time, the values it produces are close to, but not
  identical with, the published ones in `brand materials/`.

## Data

- `company_data/` — re-extracted 10-d color vectors of the top-100 publicly
  traded companies (by market capitalization, TradingView, February 2025)
- `brand materials/sorted_cosine_values.csv`,
  `brand_similarity_all_metrics.csv`, `cosine_summary_by_sector.csv` — the
  logo-vs-web-image cosine similarities and per-sector means reported in the
  manuscript (Fig 3, Fig 5 and S4 Table)
- `csv/revision_results/` — per-analysis outputs
- `brand materials/` — the brand list and numerical outputs for the top-100
  companies. Google screenshots and corporate logo images are not
  redistributed (see `brand materials/README.txt`); they are re-scraped by
  `LogoAndScreenshotOfBrands.ipynb`
- `Images/` — not redistributed (third-party Google Image content; see
  `Images/README.txt`); recreated locally by the scraping scripts
- `external_compare/` — public datasets from prior approaches (Rathore
  UW-58 / BCP-37; comp-syn JzAzBz embeddings) used by validation scripts

## Citation

Wang, S., & Park, J. *Word–Color Association from Large-Scale Online
Data: Methodology and Applications*. PLOS ONE (under revision).
