Image directories (Images/original_images, Images/resize_images,
Images/r26b_pages, Images/temp) are NOT included in this distribution.

They are third-party Google Image search results / screenshots and cannot be
redistributed under PLOS ONE's CC BY 4.0 license.

To reproduce: run word2image.py / main.py (single-word queries) or
page_compare_scrape.py (depth-page screenshots) — these scripts recreate the
directories locally from your own Google Image session. The downstream
notebooks (calculation.ipynb, ComparisonPoets.ipynb) and the analysis scripts
then consume the locally re-scraped images.

The processed per-word color vectors used in the manuscript's figures and
tables (csv/processed_csv/al_df_with_ratios.csv, gdj_df_with_ratios.csv) and
the per-analysis outputs (csv/revision_results/) are provided directly, so
downstream analysis reproduces without re-scraping.
