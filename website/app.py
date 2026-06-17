#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import shutil
import re
import math
import requests
import sympy
import numpy as np
import pytesseract
import colorsys
import base64
import networkx as nx
import scipy.stats as stats
from io import BytesIO
from collections import Counter
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bing_image_downloader.downloader import download
from PIL import Image, ImageDraw
import plotly.graph_objects as go


# In[4]:


"""
All variables can be reproduced based on the results from calculation.py.
"""

# Define ranges for chromatic colors
color_ranges = {
    "red": [(19.326959847036328, 45.67667984189723)],
    "orange": [(45.67667984189723, 80.90534979423869)],
    "yellow": [(80.90534979423869, 120.1811320754717)],
    "green": [(120.1811320754717, 203.28)],
    "blue": [(203.28, 293.86897590361446)],
    "purple": [(293.86897590361446, 331.2451923076923)],
    "pink": [(331.2451923076923, 360.00), (0.00, 19.326959847036328)]
}

# Define ranges for achromatic colors
a_color_ranges = {
    "black": [(0, 41.65384615384615)],
    'grey': [(41.65384615384615, 82.89746682750301)],
    'white': [(82.89746682750301, 100)]
}
# Define the minimum chroma to determine which pixels to hue
chroma_threshold = 15.930582082686795

# Define the proportion of white lattice in the screenshot
white_lattice = 31.41835271415166


# In[ ]:


def google_image_search_screenshot(keyword,screenshot_dir):
    # Setup the webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # GUI off
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")  # browser maximize
    options.add_argument("--incognito")  
    options.add_argument("disable-blink-features=AutomationControlled")  
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

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
        
        # Crop the top 5% of the image
        with Image.open(screenshot_path) as img:
            width, height = img.size
            crop_area = (0, int(height * 0.05), width, height)
            cropped_img = img.crop(crop_area)
            cropped_img.save(screenshot_path)
        #with Image.open(screenshot_path) as img:
        #    img.save(screenshot_path) 

        print(f"Screenshot screenshot saved to {screenshot_path}")
        
    finally:
        driver.quit()

# In[ ]:


import os
import shutil
from bing_image_downloader.downloader import download
from PIL import Image
import pytesseract
import re
import requests
from bs4 import BeautifulSoup


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



# In[7]:



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


# In[9]:


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

"""
These functions are used to extract the first, second, or third element from each sublist 
in a list of lists, where each sublist contains three elements.
"""

def leave_last_element(List):
    lst=[]    
    for x in List:
            lst.append(x[2])
    return lst

def leave_middle_element(List):
    lst=[]
    for x in List:
        lst.append(x[1])
    return lst

def leave_first_element(List):
    lst=[]
    for x in List:
        lst.append(x[0])
    return lst


# In[ ]:


import numpy as np
from PIL import Image
import os
"""
Extract colors from an image and return three values:
1. The original Lab values.
2. Updated Hue values considering issues with very low chroma pixels.
3. Updated Lightness values considering the white lattice in the screenshot.
The key process is restoring as much as was removed by considering the distribution of the remaining values.
"""

def extractColors(base_dir, search_term=""):
    filename = search_term + ".png"
    resize_image_dir = base_dir
    img_url = os.path.join(resize_image_dir, filename)

    # Open the image file
    image = Image.open(img_url)

    # Get the size of the image in pixels
    width, height = image.size

    # Create an empty list to store the RGB values for each pixel
    rgb_list = []
    lab_list = []

    # Loop over each pixel in the image and extract the RGB values
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if isinstance(pixel, int):
                r, g, b = pixel, pixel, pixel
            else:
                r, g, b = pixel[:3]
            rgb_list.append((r, g, b))

    # Convert RGB to Lab
    for (r, g, b) in rgb_list:
        if isinstance(r, int) and isinstance(g, int) and isinstance(b, int) and 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            # Convert RGB to XYZ
            r_norm = r / 255
            g_norm = g / 255
            b_norm = b / 255

            r_lin = r_norm if r_norm <= 0.04045 else ((r_norm + 0.055) / 1.055) ** 2.4
            g_lin = g_norm if g_norm <= 0.04045 else ((g_norm + 0.055) / 1.055) ** 2.4
            b_lin = b_norm if b_norm <= 0.04045 else ((b_norm + 0.055) / 1.055) ** 2.4

            x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
            y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
            z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

            # Convert XYZ to Lab
            xn, yn, zn = 0.95047, 1.00000, 1.08883

            f_x = x ** (1/3) if x > 0.008856 else 7.787 * x + 16 / 116
            f_y = y ** (1/3) if y > 0.008856 else 7.787 * y + 16 / 116
            f_z = z ** (1/3) if z > 0.008856 else 7.787 * z + 16 / 116

            L = 116 * f_y - 16
            a = 500 * (f_x - f_y)
            b = 200 * (f_y - f_z)

            lab_list.append((L, a, b))
        else:
            print("Invalid RGB value:", (r, g, b))

    if not lab_list:
        return None

    # Convert Lab values to LCH values
    lch_list = []
    lch_real_list = []
    chroma_threshold = 15.930582082686795  # Chroma threshold value
    white_lattice = 31.41835271415166  # White lattice proportion

    for lab in lab_list:
        L, a, b = lab
        C = np.sqrt(a**2 + b**2)
        if C < chroma_threshold:
            H = -1  # Assume -1 indicates an invalid or undefined Hue due to low chroma
        else:
            H = np.degrees(np.arctan2(b, a))
            if H < 0:
                H += 360
        lch_list.append((L, C, H))

    for lab in lab_list:
        L, a, b = lab
        C = np.sqrt(a**2 + b**2)
        H = np.degrees(np.arctan2(b, a))
        if H < 0:
            H += 360
        lch_real_list.append((L, C, H))

    # Extract H values and handle low chroma pixels
    h_values = leave_last_element(lch_list)
    h_minus_one_count = h_values.count(-1)

    bins = np.linspace(0, 360, 361)
    h_distribution, _ = np.histogram([h for h in h_values if h != -1], bins=bins)

    total_valid_h = sum(h_distribution)
    h_distribution_ratios = h_distribution / total_valid_h if total_valid_h > 0 else np.zeros(len(h_distribution))

    distributed_counts = np.round(h_distribution_ratios * h_minus_one_count).astype(int)

    new_h_values = [h for h in h_values if h != -1]
    for i, count in enumerate(distributed_counts):
        new_h_values.extend([i + 0.5] * count)

    # Extract L values and handle white lattice
    l_values = [lab[0] for lab in lab_list]
    threshold = np.percentile(l_values, 100 - white_lattice)
    l_values_cut = [l for l in l_values if l <= threshold]

    l_hist, bins = np.histogram(l_values_cut, bins=np.arange(0, 101, 1))
    num_cut = len(l_values) - len(l_values_cut)
    l_distribution = l_hist / l_hist.sum()
    additional_values = np.random.choice(bins[:-1], size=num_cut, p=l_distribution)

    new_l_values = np.sort(np.concatenate((l_values_cut, additional_values)))

    return lch_real_list, new_h_values, new_l_values



# In[2]:


"""
To create a word-color association that maximally reflects human cognitive abilities, the LCH system has been adopted.
However, for the visualization process, conversion to RGB is necessary.
"""
# Convert from LCH to XYZ color space.
def lch_to_xyz(l, c, h):
    # Convert h to radians
    h_rad = math.radians(h)
    
    # Calculate the a and b values
    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)
    
    # Calculate the intermediate values
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    
    # Calculate the XYZ values
    x_n = 0.95047   # D65 reference white
    y_n = 1.0
    z_n = 1.08883
    x = x_n * (fx ** 3 if fx ** 3 > 0.008856 else (fx - (16.0/116.0)) / 7.787) * 100
    y = y_n * (fy ** 3 if fy ** 3 > 0.008856 else (fy - (16.0/116.0)) / 7.787) * 100
    z = z_n * (fz ** 3 if fz ** 3 > 0.008856 else (fz - (16.0/116.0)) / 7.787) * 100
    
    return x, y, z

# Convert from XYZ to RGB color space.
def xyz_to_rgb(xyz):
    # Transformation matrix
    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570]]
    
    rgb = []
    
    for x, y, z in xyz:
        # Normalize
        x /= 100
        y /= 100
        z /= 100
        
        # Convert XYZ to RGB
        r = M[0][0] * x + M[0][1] * y + M[0][2] * z
        g = M[1][0] * x + M[1][1] * y + M[1][2] * z
        b = M[2][0] * x + M[2][1] * y + M[2][2] * z
        
        # Apply gamma correction
        if r > 0.0031308:
            r = 1.055 * (r ** (1 / 2.4)) - 0.055
        else:
            r *= 12.92
        
        if g > 0.0031308:
            g = 1.055 * (g ** (1 / 2.4)) - 0.055
        else:
            g *= 12.92
        
        if b > 0.0031308:
            b = 1.055 * (b ** (1 / 2.4)) - 0.055
        else:
            b *= 12.92
        
        # Clip to the range 0 to 1
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        rgb.append((r, g, b))
    
    return rgb


# In[ ]:


"""
Each word is represented by the color ratio based on pre-segmented color ranges.
The chromatic ratio of the word is expressed based on the h values of the pixels,
and the achromatic ratio of the word is expressed based on the l values of the pixels.
"""

def color_ratio_for_each_term(word, base_dir):
    # Return hue values for determining the chromatic ratio
    values = list(extractColors(base_dir, word)[1])
    
    # Initialize color counts
    color_counts = {color: 0 for color in color_ranges}

    # Determine which color range each value belongs to and count
    for value in values:
        for color, ranges in color_ranges.items():
            for start, end in ranges:
                if start <= value % 360 <= end:
                    color_counts[color] += 1
                    break
    
    # Total number of values
    total_values = len(values)

    # Calculate and print the ratio of each color
    color_ratios = {color: count / total_values for color, count in color_counts.items()}

    # Return lightness values for determining the achromatic ratio
    a_values = list(extractColors(base_dir, word)[2])

    # Initialize achromatic color counts
    a_color_counts = {color: 0 for color in a_color_ranges}

    # Determine which achromatic color range each value belongs to and count
    for value in a_values:
        for color, ranges in a_color_ranges.items():
            for start, end in ranges:
                if start <= value % 100 <= end:
                    a_color_counts[color] += 1
                    break
    
    # Total number of values
    a_total_values = len(a_values)

    # Calculate and print the ratio of each achromatic color
    color_ratios_a = {color: count / a_total_values for color, count in a_color_counts.items()}
    
    # Combine chromatic and achromatic color ratios
    color_dict_final = dict(color_ratios, **color_ratios_a)
    
    return color_dict_final



# In[ ]:



""" 
Visualize the colors using polar coordinates where chroma is the radius and hue is the angle.
"""

def colorWordVisualization(lch_list, lightness):
    # Copy the filtered LCH list
    filtered_lch_list = lch_list.copy()

    # Convert LCH to the given lightness
    lch = [(lightness, c, h) for _, c, h in filtered_lch_list]
    xyz = [lch_to_xyz(l, c, h) for l, c, h in lch]
    rgb = xyz_to_rgb(xyz)

    # Calculate the frequency of points
    point_counts = Counter((round(h, 2), round(c, 2)) for _, c, h in filtered_lch_list)

    # Generate colors for the ring based on the H values in LCH
    theta_full_range = np.linspace(0, 2 * np.pi, 360)
    lch_colors_ring = [(70, 100, h) for h in np.linspace(0, 360, 360)]
    xyz_colors_ring = [lch_to_xyz(l, c, h) for l, c, h in lch_colors_ring]
    colors_ring = ['rgba({}, {}, {}, 1)'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in xyz_to_rgb(xyz_colors_ring)]

    # Draw the inner coloring ring with varying colors based on H values
    ring_trace = go.Scatterpolar(
        r=[1.12] * len(theta_full_range),
        theta=np.degrees(theta_full_range),
        mode='markers',
        marker=dict(
            color=colors_ring,
            size=15,
            opacity=1
        ),
        showlegend=False,
        hoverinfo='none'
    )

    # Draw cluster points
    base_alpha = 0.2  # Set base alpha value
    alpha_increment = 0.05  # Set alpha increment rate
    cluster_traces = []
    for (_, c, h), color in zip(filtered_lch_list, rgb):
        count = point_counts[(round(h, 2), round(c, 2))]
        alpha = min(base_alpha + count * alpha_increment, 1.0)  # Adjust transparency based on overlap
        rgba_color = 'rgba({}, {}, {}, {})'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), alpha)
        cluster_traces.append(go.Scatterpolar(
            r=[c / 120 * 1.00],
            theta=[h],
            mode='markers',
            marker=dict(
                color=rgba_color,
                size=8,
                line=dict(color='lightgray', width=0.5)
            ),
            showlegend=False,
            hoverinfo='text',
            text=[f'c: {c}, h: {h:.2f}°']  # Set hover text
        ))

    # Draw lines and labels at specific H values
    hues_to_mark = [19.33, 45.68, 80.91, 120.18, 203.28, 293.87, 331.25]
    line_traces = []
    text_traces = []
    for hue in hues_to_mark:
        line_traces.append(go.Scatterpolar(
            r=[0, 1.2],
            theta=[hue, hue],
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False,
            hoverinfo='none'
        ))
        text_traces.append(go.Scatterpolar(
            r=[1.35],  # Position the label further outside
            theta=[hue],
            mode='text',
            text=[f'{hue:.2f}°'],
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hoverinfo='none'
        ))

    # Set layout
    layout = go.Layout(
        width=graph_width,  # Set width to 300 pixels
        height=300,  # Set height to 300 pixels
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor='#e0f7fa'  # Set background color
        ),
        showlegend=False,
        margin=dict(l=40, r=40, b=40, t=40),  # Set margins
    )

    # Combine data
    data = [ring_trace] + cluster_traces + line_traces + text_traces

    # Draw the graph
    fig = go.Figure(data=data, layout=layout)

    # Convert the graph to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html





# In[ ]:


"""
Displays the distribution of pixel hue values in a histogram.
"""
def analyze_number_distribution_hue(h_list):
    
    # Calculate the count of h_list in each bin
    counts, _ = np.histogram(h_list, bins=np.arange(361))

    # Calculate colors based on H values in LCH
    lch_colors = [(50, 50, h) for h in range(360)]
    rgb_colors = [lch_to_rgb(l, c, h) for l, c, h in lch_colors]
    
    # Set bar colors
    bar_colors = ['rgba({}, {}, {}, 1)'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in rgb_colors]

    # Visualize as a bar graph
    bars = go.Bar(
        x=np.arange(360),
        y=counts,
        marker=dict(color=bar_colors),
        hoverinfo='text',
        hovertemplate='Hue: %{x} <br>Count: %{y}<extra></extra>',  # Hover text
        name=''  # Set trace name as an empty string
    )

    fig = go.Figure(data=[bars])

    fig.update_layout(
        xaxis_title='Hue',
        yaxis_title='Count',
        xaxis=dict(range=[0, 360]),
        yaxis=dict(range=[0, max(counts)]),
        template='plotly_white',
        width=graph_width,  # Set width to 300 pixels
        height=graph_height,  # Set height to 300 pixels
        margin=graph_margin,  # Set graph margins
        plot_bgcolor='#e0f7fa',
    )

    # Add hover effect to bars (remove outline)
    fig.update_traces(marker=dict(line=dict(width=0)))

    # Convert the graph to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html

def lch_to_rgb(l, c, h):
    # Convert LCH to RGB (approximate conversion)
    h = h / 360.0
    r, g, b = colorsys.hls_to_rgb(h, l / 100.0, c / 100.0)
    return r, g, b


# In[ ]:


"""
Displays the distribution of pixel lightness values in a histogram.
"""

def analyze_number_distribution_lightness(l_list):
    # Calculate the number of l_list in each bin
    counts, _ = np.histogram(l_list, bins=np.arange(101))

    # Create a colormap from white to black
    colors = ['rgba({}, {}, {}, 1)'.format(int(i * 2.55), int(i * 2.55), int(i * 2.55)) for i in range(101)]

    # Plot the bar graph using Plotly
    bars = go.Bar(
        x=np.arange(100),
        y=counts,
        marker=dict(color=colors),
        hoverinfo='text',
        hovertemplate='Lightness: %{x} <br>Count: %{y}<extra></extra>',  # Hover text
        name=''  # Set trace name as an empty string
    )

    fig = go.Figure(data=[bars])

    fig.update_layout(
        xaxis_title='Lightness',
        yaxis_title='Count',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, max(counts)]),
        template='plotly_white',
        width=graph_width,  # Set width to 300 pixels
        height=graph_height,  # Set height to 300 pixels
        margin=graph_margin,  # Set graph margins 
        plot_bgcolor='#e0f7fa',  # Set background color
        
    )

    # Add hover effect to bars
    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))

    # Convert the graph to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# In[ ]:


"""
Displays the pixels on a scatter plot with chroma on the x-axis and lightness on the y-axis.
"""

def lightnessWordVisualization(lch_list):
    # Copy the filtered LCH list
    filtered_lch_list = lch_list.copy()
    
    # Calculate colors by setting H value to 0 and C value to 0 in LCH
    lch_for_color = [(l, 0, 0) for l, c, h in filtered_lch_list]
    xyz_for_color = [lch_to_xyz(l, c, h) for l, c, h in lch_for_color]
    rgb_for_color = xyz_to_rgb(xyz_for_color)
    
    # Calculate the frequency of points
    point_counts = Counter((round(l, 2), round(c, 2)) for l, c, h in filtered_lch_list)
    
    # Generate data
    c_vals = [c for _, c, _ in filtered_lch_list]
    l_vals = [l for l, _, _ in filtered_lch_list]
    
    # Draw cluster points
    base_alpha = 0.2  # Set base alpha value
    alpha_increment = 5.0  # Set alpha increment rate
    cluster_traces = []
    for l, c, color in zip(l_vals, c_vals, rgb_for_color):
        count = point_counts[(round(l, 2), round(c, 2))]
        alpha = min(base_alpha + count * alpha_increment, 1.0)  # Adjust transparency based on overlap
        rgba_color = 'rgba({}, {}, {}, {})'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), alpha)
        cluster_traces.append(go.Scatter(
            x=[c],
            y=[l],
            mode='markers',
            marker=dict(
                color=rgba_color,
                size=8,
                line=dict(color='lightblue', width=0.5)
            ),
            showlegend=False,
            hoverinfo='text',
            text=[f'L: {l}, C: {c}']  # Set hover text
        ))

    # Add grayscale color bar
    lightness_vals = np.linspace(0, 100, 100).reshape(100, 1)
    grayscale_colors = np.hstack([lightness_vals / 100, lightness_vals / 100, lightness_vals / 100])
    
    color_bar_trace = go.Heatmap(
        z=lightness_vals,
        colorscale=[[0, 'black'], [1, 'white']],  # Set color scale based on lightness values
        showscale=False,
        hoverinfo='none',
        xaxis='x2',
        yaxis='y2',
        opacity=1.0  # Set the background color of the color bar to be opaque
    )

    # Set graph size
    title_standoff_value = graph_width * 0.01  # Set title standoff to 1% of the graph width

    layout = go.Layout(
        xaxis=dict(domain=[0.26, 1], range=[0, 100], title='Chroma (C)', showgrid=False, gridcolor='white'),
        yaxis=dict(domain=[0, 1], range=[0, 100], showgrid=True, zeroline=False, showticklabels=False, ticks='', gridcolor='white'),
        xaxis2=dict(domain=[0.18, 0.25], anchor='y2', showgrid=False, zeroline=False, ticks='', showticklabels=False),
        yaxis2=dict(domain=[0, 1], anchor='x2', showgrid=True, zeroline=False, ticks='', range=[0, 100], showticklabels=True, title='Lightness (L)', title_standoff=title_standoff_value, gridcolor='white'),
        plot_bgcolor='#e0f7fa',
        width=graph_width,  # Set width to 300 pixels
        height=graph_height,  # Set height to 300 pixels
        margin=graph_margin,  # Set graph margins
    )

    fig = go.Figure(data=cluster_traces + [color_bar_trace], layout=layout)
    
    plot_html = fig.to_html(full_html=False)
    return plot_html


# In[ ]:


"""
Visualizes the composition ratio of chromatic colors for a word based on predefined color ranges.
"""

def colorWordRatio_hue(word, base_dir):
    # Define the hue ranges and corresponding colors
    hues_to_mark = [19.33, 45.68, 80.91, 120.18, 203.28, 293.87, 331.25, 360 + 19.33]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    
    color_ratio = color_ratio_for_each_term(word, base_dir)
    
    # Get the color ratios in the specified order
    ratios = [color_ratio[color] for color in colors]

    # Create the figure
    fig = go.Figure()

    max_radius = 1.0
    total_area = np.pi * max_radius**2

    result_areas = {}

    for i, (color, ratio) in enumerate(zip(colors, ratios)):
        start_angle = np.radians(hues_to_mark[i])
        end_angle = np.radians(hues_to_mark[i + 1])
        
        theta = np.linspace(np.degrees(start_angle), np.degrees(end_angle), 100)
        
        # Calculate the area for the color ratio and derive the radius
        sector_area = ratio * total_area
        sector_angle = end_angle - start_angle
        r = np.sqrt(sector_area / (np.pi * (sector_angle / (2 * np.pi))))
        
        # Create the radius array
        radius_arr = np.full_like(theta, r)
        
        # Add the filled sector to the figure
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate(([0], radius_arr, [0])),  # Start and end at 0
            theta=np.concatenate(([theta[0]], theta, [theta[-1]])),
            fill='toself',
            fillcolor=color,
            line=dict(color='black'),
            hoverinfo='text',  # Show hover info only on the sector
            text=f'{color}: {ratio * 100:.2f}%',  # Display the color ratio as percentage
            showlegend=False
        ))

        # Add to the result areas dictionary
        result_areas[color] = sector_area / total_area

    # Draw the inner dotted circle
    fig.add_trace(go.Scatterpolar(
        r=[max_radius] * 100,
        theta=np.linspace(0, 360, 100),
        mode='lines',
        line=dict(color='black', dash='dot'),
        hoverinfo='skip',  # Remove hover info from the inner circle
        showlegend=False
    ))

    # Add the dividing lines and labels for each segment
    extended_radius = max_radius * 1.5  # Extend the lines outward
    for hue in hues_to_mark[:-1]:
        hue_deg = hue % 360
        fig.add_trace(go.Scatterpolar(
            r=[0, extended_radius],
            theta=[hue_deg, hue_deg],
            mode='lines',
            line=dict(color='black'),
            hoverinfo='skip',  # Remove hover info from the lines
            showlegend=False
        ))
        # Add angle labels next to the dividing lines
        fig.add_trace(go.Scatterpolar(
            r=[extended_radius + 0.3],  # Position the angle labels further out
            theta=[hue_deg],
            mode='text',
            text=[f'{hue:.2f}°'],
            textposition='top center',
            hoverinfo='skip',  # Remove hover info from the labels
            showlegend=False
        ))

    fig.update_layout(
        width=graph_width,  # Set width to 300 pixels
        height=graph_height,  # Set height to 300 pixels
        margin=graph_margin,  # Set graph margins
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(showticklabels=False),
            bgcolor='#e0f7fa'
        ),
        showlegend=False,  # Remove legend
    )

    plot_html = fig.to_html(full_html=False)
    return plot_html



# In[ ]:


"""
Visualizes the composition ratio of achromatic colors for a word based on predefined color ranges.
"""

def colorWordRatio_lightness(word, base_dir):
    color_ratio = color_ratio_for_each_term(word, base_dir)

    # Get the achromatic color ratios
    black_ratio = color_ratio.get('black', 0)
    grey_ratio = color_ratio.get('grey', 0)
    white_ratio = color_ratio.get('white', 0)

    # Line positions for dividing sections
    line_positions = [0.4165384615384615, 0.8289746682750301]
    y_ranges = [0, 0.4165384615384615, 0.8289746682750301, 1.0]

    fig = go.Figure()

    # Set background color
    fig.update_layout(
        plot_bgcolor='#e0f7fa',
        width=300,  # Set width
        height=300  # Set height
    )

    # Draw bar graphs for each achromatic color
    colors_rect = ['black', 'grey', 'white']
    color_names = ['Black', 'Grey', 'White']
    ratios = [black_ratio, grey_ratio, white_ratio]

    for i, (color, color_name, ratio) in enumerate(zip(colors_rect, color_names, ratios)):
        fig.add_trace(go.Bar(
            x=[ratio],
            y=[(y_ranges[i] + y_ranges[i + 1]) / 2],
            width=[y_ranges[i + 1] - y_ranges[i]],
            marker_color=color,
            orientation='h',
            showlegend=False,
            hovertemplate=f'{color_name}: {ratio * 100:.1f}%<extra></extra>'
        ))

    # Draw dividing lines
    for pos in line_positions:
        fig.add_shape(type="line",
                      x0=0, y0=pos, x1=1, y1=pos,
                      line=dict(color="black", width=1.5))

    # Add axis labels
    fig.update_layout(
        height=graph_height,  # Set height to 300 pixels
        margin=graph_margin,  # Set graph margins  # Set height to 800 pixels
        xaxis=dict(showticklabels=False),
        yaxis=dict(
            tickmode='array',
            tickvals=[0] + line_positions + [1],
            ticktext=[f'{pos:.2f}' for pos in [0] + line_positions + [1]],
            tickfont=dict(size=10),
            range=[0, 1]
        ),
    
    )

    plot_html = fig.to_html(full_html=False)
    
    return plot_html


# In[ ]:



# In[ ]:


"""
Allows viewing all visualizations that can be created using color information associated with a word on a single webpage.
"""

app = Flask(__name__)

graph_width = 300  
graph_height = 300  
graph_margin=dict(l=10, r=10, t=10, b=10)

@app.route('/', methods=['GET', 'POST'])
def index():

    # Get the path of the currently running script.
    current_script_path = os.path.abspath(__file__)
    # Get the path of the folder where the current script is located.
    script_folder_path = os.path.dirname(current_script_path)
    # Get the full folder path.
    original_folder_path = os.path.dirname(script_folder_path)
    # Get the path of the images folder.
    original_dir = os.path.join(original_folder_path, 'Images','original_images')
    temp_dir = os.path.join(original_folder_path,'Images', 'temp')
    resize_dir = os.path.join(original_folder_path,'Images', 'resize_images')
    # Create folders if they do not exist.
    for dir_path in [original_dir, temp_dir, resize_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")

    plot_html_1 = None
    plot_html_2 = None    
    plot_html_3 = None    
    plot_html_4 = None    
    plot_html_5 = None    
    plot_html_6 = None

    
#    if request.method == 'POST':
#        word = request.form['word']
#
#
#        check_text = request.form.get('check_text', 'True') == 'True'  # 라디오 버튼에서 값을 가져옴
#
#        word2image(word,10000,original_dir,temp_dir,resize_dir, check_text=check_text)
#        
#        color_dict_final = color_ratio_for_each_term(word,resize_dir)
#        
#        color_list = list(extractColors(resize_dir, word))
#        
#        plot_html_1 = colorWordVisualization(color_list[0], 70)
#        plot_html_2 = analyze_number_distribution_hue(color_list[1])
#        plot_html_3 = lightnessWordVisualization(color_list[0])
#        plot_html_4 = analyze_number_distribution_lightness(color_list[2])
#        plot_html_5 = colorWordRatio_hue(word,resize_dir)
#        plot_html_6 = colorWordRatio_lightness(word,resize_dir)
#        
    if request.method == 'POST':
        word = request.form['word']

        check_text = request.form.get('check_text', 'True') == 'True'  # 라디오 버튼에서 값을 가져옴

        word2image(word, 10000, original_dir, temp_dir, resize_dir, check_text=check_text)
        
        color_dict_final = color_ratio_for_each_term(word, resize_dir)
        
        color_list = list(extractColors(resize_dir, word))
        lch_real_list = color_list[0]   # (L, C, H) 리스트

        # -----------------------------
        # 1) C 값 분포 기반으로 저채도 비율 계산
        # -----------------------------
        chroma_threshold = 15.930582082686795  # 이미 위에서 쓰는 값과 동일하게
        c_values = [c for (L, c, H) in lch_real_list]

        if len(c_values) > 0:
            low_chroma_ratio = sum(1 for c in c_values if c <= chroma_threshold) / len(c_values)
        else:
            low_chroma_ratio = 1.0  # 혹시나 비어 있으면 '거의 무채색'으로 취급

        print(f"Low-chroma ratio: {low_chroma_ratio:.4f}")

        # -----------------------------
        # 2) 공통으로 그리는 1~4번 그래프
        # -----------------------------
        plot_html_1 = colorWordVisualization(color_list[0], 70)
        plot_html_2 = analyze_number_distribution_hue(color_list[1])
        plot_html_3 = lightnessWordVisualization(color_list[0])
        plot_html_4 = analyze_number_distribution_lightness(color_list[2])

        # -----------------------------
        # 3) 조건에 따라 5번/6번 중 하나만 선택
        #    - low_chroma_ratio >= 0.9 이면 lightness 비율만
        #    - 그렇지 않으면 hue 비율만
        # -----------------------------
        if low_chroma_ratio >= 0.9:
            plot_html_5 = None
            plot_html_6 = colorWordRatio_lightness(word, resize_dir)
        else:
            plot_html_5 = colorWordRatio_hue(word, resize_dir)
            plot_html_6 = None
        
    return render_template('index.html', plot_html_1=plot_html_1, plot_html_2=plot_html_2, plot_html_3=plot_html_3, plot_html_4=plot_html_4, plot_html_5=plot_html_5, plot_html_6=plot_html_6)

if __name__ == '__main__':
    app.run(debug=True, port=5002)





