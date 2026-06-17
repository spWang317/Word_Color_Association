"""
R2.6 v2 — Capture Google Image results-page screenshots for the relaxed
crop robustness word set.
"""

import os
import time

import word2image as w2i

OUT_DIR = "Images/r26_raw_v2"
os.makedirs(OUT_DIR, exist_ok=True)

WORDS = [
    "activity", "aside", "blog", "boat", "boyfriend", "business",
    "character", "college", "corporation", "customer", "daughter",
    "device", "door", "equivalent", "escape", "form", "government",
    "jail", "morning", "nothing", "peace", "position", "situation",
    "squad", "study", "success", "surprise", "television", "vice", "week",
]

# words from v1 we can copy over if already scraped (saves Google API calls)
V1_DIR = "Images/r26_raw"


def main():
    import shutil
    done, failed = [], []
    for i, w in enumerate(WORDS, 1):
        path = os.path.join(OUT_DIR, f"{w}.png")
        if os.path.exists(path):
            print(f"[{i}/{len(WORDS)}] {w}: already exists, skip")
            done.append(w)
            continue
        # Reuse v1 screenshot if available
        v1_path = os.path.join(V1_DIR, f"{w}.png")
        if os.path.exists(v1_path):
            shutil.copy(v1_path, path)
            print(f"[{i}/{len(WORDS)}] {w}: copied from v1")
            done.append(w)
            continue
        try:
            print(f"[{i}/{len(WORDS)}] {w}: capturing...")
            w2i.google_image_search_screenshot(w, OUT_DIR)
            if os.path.exists(path):
                done.append(w)
            else:
                failed.append(w)
        except Exception as e:
            print(f"  ERROR on {w}: {e!r}")
            failed.append(w)
        time.sleep(2)

    print(f"\nDONE. captured {len(done)}/{len(WORDS)}.")
    if failed:
        print(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
