#!/usr/bin/env python
# coding: utf-8
# %%


import os
import time
import shutil
import re
import requests
import sympy
import pytesseract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bing_image_downloader.downloader import download
from PIL import Image


# %%



def google_image_search_screenshot(keyword,screenshot_dir):
    # Setup the webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # GUI off
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")  # browser maximize
    # Incognito session: no history, regional/language presets, or personalization.
    # Light-mode rendering of the search page itself is forced below via the CDP
    # Emulation.setEmulatedMedia call (prefers-color-scheme=light), which makes
    # the result independent of the host OS theme. This is what guarantees the
    # white-divider trimming pipeline (S1 Fig) works correctly under incognito.
    options.add_argument("--incognito")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--force-color-profile=srgb")
    options.add_argument("--force-light-mode")
    options.add_argument("--disable-features=DarkMode,WebUIDarkMode")
    options.add_argument("disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    # CRITICAL: Chrome's --force-light-mode only affects Chrome's own UI, not
    # the rendered website. Google Images honors the CSS prefers-color-scheme
    # media query (driven by the OS theme), so on a dark-mode OS the page
    # would render with a DARK background, which corrupts color extraction
    # (the white-divider trimming in S1 Fig assumes white/bright dividers).
    # We force the page to light mode via CDP emulation, independent of OS theme.
    driver.execute_cdp_cmd("Emulation.setEmulatedMedia", {
        "features": [{"name": "prefers-color-scheme", "value": "light"}]
    })

    try:
        # Navigate to Google Images
        driver.get("https://www.google.com/imghp")
        
        # Perform the search
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for the results to load
        time.sleep(10)
        
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.05);")
        
        # Wait for scroll and additional loading
        time.sleep(10)
        
        # Ensure all images are loaded
        images_loaded = False
        while not images_loaded:
            time.sleep(2)
            images = driver.find_elements(By.TAG_NAME, "img")
            images_loaded = all(img.get_attribute('complete') for img in images)
        
        # Set browser to full screen mode
        driver.set_window_size(1920, 1080)
        driver.maximize_window()
        
        # Wait for scroll and additional loading
        time.sleep(10)
        
        # Take a screenshot
        screenshot = driver.get_screenshot_as_png()
        screenshot_path = os.path.join(screenshot_dir,f"{keyword}.png")
        
        with open(screenshot_path, "wb") as file:
            file.write(screenshot)
        
        #Crop the top 5% of the image
        with Image.open(screenshot_path) as img:
           width, height = img.size
           crop_area = (0, int(height * 0.05), width, height)
           cropped_img = img.crop(crop_area)
           cropped_img.save(screenshot_path)
        # with Image.open(screenshot_path) as img:
        #     img.save(screenshot_path) 

        print(f"Screenshot screenshot saved to {screenshot_path}")
        
    finally:
        driver.quit()


# %%



"""
Determine if there are enough images related to the keyword. If not, assume no relevant images exist for the keyword.
"""
def check_images_without_download(keyword):
    url = f"https://www.bing.com/images/search?q={keyword}&count=100"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')
        image_count = len(images)
        if image_count >= 30:
            print(f"There are more than 30 images related to '{keyword}'. (Total {image_count} images)")
            return True
        else:
            print(f"There are less than 30 images related to '{keyword}'. (Total {image_count} images)")
            return False
    else:
        print(f"Failed to search '{keyword}': HTTP {response.status_code}")
        return False

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

"""
Determine if there are enough images related to the keyword and if so, proceed to download them.
If more than 3 images from the initial 10 contain text, assume no relevant images exist for the keyword.
"""

def check_images(keyword, temp_dir, check_text):
    print(f"check_text value: {check_text}, type: {type(check_text)}")
    
    if not check_images_without_download(keyword):
        print("Not enough images, no images related to the keyword.")
        return False
    
    
    # Skip text detection if check_text is False
    if check_text is False:
        print("Skipping text detection. Assuming images are valid.")
        return True

    # Text detection if check_text is True
    elif check_text is True:

        # Perform the download
        download(keyword, limit=10, output_dir=temp_dir, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)

        output_dir = os.path.join(temp_dir, keyword)

        # Remove .DS_Store file if it exists
        ds_store_path = os.path.join(output_dir, '.DS_Store')
        if os.path.exists(ds_store_path):
            os.remove(ds_store_path)

        # Check files in the directory
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        files.sort(key=natural_sort_key)

        text_image_count = 0
        for file_name in files[:10]:
            print(file_name)
            file_path = os.path.join(output_dir, file_name)
            try:
                with Image.open(file_path) as img:
                    text = pytesseract.image_to_string(img)
                if text.strip():
                    text_image_count += 1
                if text_image_count > 3:
                    shutil.rmtree(output_dir)  # Delete the folder
                    print("Text found, no images related to the keyword.")
                    return False
            except Exception as e:
                print(f"Image processing error {file_path}: {e}")
                continue

    print("Images are valid. Proceeding without issues.")
    return True






# %%



def resize_and_save_screenshot(screenshot_path, keyword, pixel_count, save_dir):    
    # Construct the specific image path
    image_file = f"{keyword}.png"
    screenshot_file_path = os.path.join(screenshot_path, image_file)
    
    if not os.path.exists(screenshot_file_path):
        print(f"{screenshot_file_path} does not exist.")
        return
    
    # Determine the new size based on the pixel count
    with Image.open(screenshot_file_path) as img:
        width, height = img.size
        ratio = max(width, height) / min(width, height)
        x = sympy.symbols("x")
        f = sympy.Eq(x * (x / ratio), pixel_count)
        new_size = sympy.solve(f)[1]
        new_width, new_height = (int(new_size / ratio), int(new_size)) if width < height else (int(new_size), int(new_size / ratio))
        
        # Resize the image
        resized_img = img.resize((new_width, new_height))
        
        # Construct the new file path
        resized_screenshot_path = os.path.join(save_dir, image_file)
        
        # Save the resized image
        resized_img.save(resized_screenshot_path)
        
        print(f"Resized screenshot saved to {resized_screenshot_path}")
        
    return


# %%


"""
If there are images related to the keyword, search for the keyword, take a screenshot of the image page, and resize it.
"""

def word2image(keyword, pixel_count, original_dir="./Images/original_images", temp_dir="./Images/temp", resize_dir="./Images/resize_images", check_text=True):
    resize_path = os.path.join(resize_dir, f"{keyword}.png")
    # Check if the file exists in the resize directory
    if os.path.isfile(resize_path):
        print(f"{keyword}.png file already exists in {resize_dir}. Skipping operation.")
    else:
        # Check if the file exists in the original directory
        original_path = os.path.join(original_dir, f"{keyword}.png")
        if os.path.isfile(original_path):
            resize_and_save_screenshot(original_dir, keyword, pixel_count, resize_dir)
            print(f"{keyword}.png file retrieved from {original_dir}, resized, and saved to {resize_dir}.")
        else:
            # Check if there are images in the temp directory, then save and resize the screenshot
            if check_images(keyword, temp_dir, check_text=check_text):
                google_image_search_screenshot(keyword, original_dir)
                resize_and_save_screenshot(original_dir, keyword, pixel_count, resize_dir)
                print(f"{keyword}.png file created via Google Image search, resized, and saved to {resize_dir}.")
            else:
                print(f"No related images for {keyword}.")


