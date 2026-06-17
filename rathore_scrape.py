"""Scrape 6 Rathore test concepts (BCP-37 ratings: materials/recycling)."""

import os
import time
import word2image as w2i

OUT_DIR = "Images/rathore_raw"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_CONCEPTS = [
    ("Compost", "compost"),
    ("Glass", "glass material"),
    ("Metal", "metal material"),
    ("Paper", "paper material"),
    ("Plastic", "plastic material"),
    ("Trash", "trash"),
]


def main():
    for i, (label, query) in enumerate(TEST_CONCEPTS, 1):
        path = os.path.join(OUT_DIR, f"{label}.png")
        if os.path.exists(path):
            print(f"[{i}/6] {label}: already exists", flush=True)
            continue
        try:
            print(f"[{i}/6] {label} (query: {query!r}): capturing...", flush=True)
            w2i.google_image_search_screenshot(query, OUT_DIR)
            scraped = os.path.join(OUT_DIR, f"{query}.png")
            if os.path.exists(scraped) and scraped != path:
                os.rename(scraped, path)
        except Exception as e:
            print(f"  ERROR {label}: {e!r}", flush=True)
        time.sleep(2)
    print("\nDONE.")


if __name__ == "__main__":
    main()
