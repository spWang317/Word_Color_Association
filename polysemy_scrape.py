"""
R2.3 polysemy demo — capture screenshots for polysemous base words and
their sense-cued variants (using the manuscript's own screenshot method).
"""

import os
import time

import word2image as w2i

OUT_DIR = "Images/r23_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# (query, safe filename stem)
QUERIES = [
    # brand polysemy (reviewer's own examples): bare token vs full registered name
    ("apple", "apple"),
    ("Apple Inc.", "apple_inc"),
    ("amazon", "amazon"),
    ("Amazon.com Inc.", "amazon_inc"),
    ("visa", "visa"),
    ("Visa Inc.", "visa_inc"),
    ("shell", "shell"),
    ("Shell plc", "shell_plc"),
    # general object polysemy: base + two sense-cue words
    ("nail", "nail"),
    ("metal nail", "nail_metal"),
    ("finger nail", "nail_finger"),
    ("chip", "chip"),
    ("potato chip", "chip_potato"),
    ("computer chip", "chip_computer"),
]


def main():
    done, failed = [], []
    for i, (q, stem) in enumerate(QUERIES, 1):
        path = os.path.join(OUT_DIR, f"{stem}.png")
        if os.path.exists(path):
            print(f"[{i}/{len(QUERIES)}] {q!r}: exists, skip")
            done.append(stem)
            continue
        try:
            print(f"[{i}/{len(QUERIES)}] {q!r} -> {stem}.png")
            w2i.google_image_search_screenshot(q, OUT_DIR)
            # the function saves as "{query}.png"; rename to safe stem
            raw = os.path.join(OUT_DIR, f"{q}.png")
            if os.path.exists(raw) and raw != path:
                os.replace(raw, path)
            if os.path.exists(path):
                done.append(stem)
            else:
                failed.append(stem)
        except Exception as e:
            print(f"  ERROR on {q!r}: {e!r}")
            failed.append(stem)
        time.sleep(2)

    print(f"\nDONE. {len(done)}/{len(QUERIES)} captured.")
    if failed:
        print("FAILED:", failed)


if __name__ == "__main__":
    main()
