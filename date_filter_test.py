#!/usr/bin/env python
# coding: utf-8
"""
Test whether Google Images respects the date-range URL parameter
(tbs=cdr:1,cd_min:M/D/YYYY,cd_max:M/D/YYYY). If yes, we can in principle
restrict the framework to images indexed within a chosen historical window.

Captures 2 screenshots for the same query across two time periods and saves
them side by side for visual inspection.
"""

import os
import time
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image

OUT_DIR = "Images/r25_date_test"
os.makedirs(OUT_DIR, exist_ok=True)

QUERY = "wedding dress"

# Two periods we expect to differ visually if the filter works
PERIODS = [
    ("1995_2000", "1/1/1995", "12/31/2000"),
    ("2020_2024", "1/1/2020", "12/31/2024"),
]


def make_url(query, cd_min, cd_max):
    q = quote_plus(query)
    return (f"https://www.google.com/search?q={q}&tbm=isch"
            f"&tbs=cdr:1,cd_min:{cd_min},cd_max:{cd_max}&hl=en&gl=us")


def setup_driver():
    o = webdriver.ChromeOptions()
    o.add_argument("--headless")
    o.add_argument("--no-sandbox")
    o.add_argument("--disable-dev-shm-usage")
    o.add_argument("--start-maximized")
    o.add_argument("--guest")
    o.add_argument("--disable-extensions")
    o.add_argument("--force-color-profile=srgb")
    o.add_argument("--force-light-mode")
    o.add_argument("--disable-features=DarkMode,WebUIDarkMode")
    o.add_argument("--lang=en-US")
    o.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/121.0.0.0 Safari/537.36")
    d = webdriver.Chrome(options=o)
    d.execute_cdp_cmd("Emulation.setEmulatedMedia",
                      {"features": [{"name": "prefers-color-scheme",
                                     "value": "light"}]})
    return d


def main():
    from selenium.webdriver.common.keys import Keys
    d = setup_driver()
    try:
        # 1) Do a normal search first (passes Google's bot check the way our
        #    other scrapes do), then navigate to the date-filtered URL.
        d.get("https://www.google.com/imghp?hl=en&gl=us")
        time.sleep(6)
        box = d.find_element(By.NAME, "q")
        box.send_keys(QUERY)
        box.send_keys(Keys.RETURN)
        time.sleep(8)
        d.set_window_size(1920, 1080)
        time.sleep(2)
        # save baseline (no date filter) for visual comparison
        base = os.path.join(OUT_DIR, f"{QUERY.replace(' ', '_')}_NOFILTER.png")
        d.save_screenshot(base)
        print(f"baseline saved: {base}")
        print(f"  URL: {d.current_url[:140]}")

        for tag, cd_min, cd_max in PERIODS:
            url = make_url(QUERY, cd_min, cd_max)
            print(f"\n[{tag}] navigating: {url}")
            d.get(url)
            time.sleep(10)
            d.execute_script("window.scrollTo(0, document.body.scrollHeight*0.05);")
            time.sleep(5)
            path = os.path.join(OUT_DIR, f"{QUERY.replace(' ', '_')}_{tag}.png")
            d.save_screenshot(path)
            with Image.open(path) as im:
                w, h = im.size
                im.crop((0, int(h * 0.05), w, h)).save(path)
            print(f"  saved: {path}")
            print(f"  final URL: {d.current_url[:140]}")
    finally:
        d.quit()
    print("\nDONE. Compare PNGs in", OUT_DIR)


if __name__ == "__main__":
    main()
