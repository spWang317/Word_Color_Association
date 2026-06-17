#!/usr/bin/env python
# coding: utf-8
# %%

import os
import math
import nltk
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import word2image as w2i
#import scipy.stats as stats
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn


# %%
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


# %%


"""
To create a word-color association that maximally reflects human cognitive abilities, the LCH system has been adopted.
However, for the visualization process, conversion to RGB is necessary.
"""

def xyz_to_rgb(xyz):
    """
    Converts XYZ color space values to RGB color space.

    Args:
        xyz (list of tuples): List of tuples where each tuple contains X, Y, and Z values.

    Returns:
        list of tuples: List of tuples where each tuple contains R, G, and B values.
    """
    # Transformation matrix
    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570]]
    
    rgb = []
    
    for x, y, z in xyz:
        # Normalize the values
        x /= 100
        y /= 100
        z /= 100
        
        # Convert XYZ to RGB
        r = M[0][0] * x + M[0][1] * y + M[0][2] * z
        g = M[1][0] * x + M[1][1] * y + M[1][2] * z
        b = M[2][0] * x + M[2][1] * y + M[2][2] * z
        
        # Apply gamma correction
        r = 1.055 * (r ** (1 / 2.4)) - 0.055 if r > 0.0031308 else r * 12.92
        g = 1.055 * (g ** (1 / 2.4)) - 0.055 if g > 0.0031308 else g * 12.92
        b = 1.055 * (b ** (1 / 2.4)) - 0.055 if b > 0.0031308 else b * 12.92
        
        # Clip values between 0 and 1
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        rgb.append((r, g, b))
    
    return rgb

def lch_to_xyz(l, c, h):
    """
    Converts LCH color space values to XYZ color space.

    Args:
        l (float): Lightness value between 0 and 100.
        c (float): Chroma value between 0 and 100.
        h (float): Hue value between 0 and 360.

    Returns:
        tuple: Tuple of floats (X, Y, Z) representing the XYZ values.
    """
    # Convert hue to radians
    h_rad = math.radians(h)
    
    # Calculate the a and b values
    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)
    
    # Calculate the intermediate values
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    
    # Calculate the XYZ values
    x_n = 0.95047  # D65 reference white
    y_n = 1.0
    z_n = 1.08883
    x = x_n * (fx ** 3 if fx ** 3 > 0.008856 else (fx - 16.0/116.0) / 7.787) * 100
    y = y_n * (fy ** 3 if fy ** 3 > 0.008856 else (fy - 16.0/116.0) / 7.787) * 100
    z = z_n * (fz ** 3 if fz ** 3 > 0.008856 else (fz - 16.0/116.0) / 7.787) * 100
    
    return x, y, z


# %%


def extractColors_before_update(resize_images_dir, search_term=""):
    """
    Extracts color information from an image file and converts it into various color spaces.

    Args:
        resize_images_dir (str): Base directory where the resized image files are located.
        search_term (str): Search term or filename (without extension) of the image.

    Returns:
        list: A list containing lists of LCH values.
    """
    filename = search_term + ".png"
    img_url = os.path.join(resize_images_dir, filename)
    
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
            xn = 0.95047
            yn = 1.00000
            zn = 1.08883

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

    # Convert CIELAB values to CIELCH values
    lch_list_before_update = []

    for lab in lab_list:
        L, a, b = lab
        C = np.sqrt(a**2 + b**2)
        H = np.degrees(np.arctan2(b, a))
        if H < 0:
            H += 360
        lch_list_before_update.append((L, C, H))

    return lch_list_before_update


# # Wild type
# 
# ## Using 100 random words, we perform two tasks.
# 
# ### 1. White lattice
# #### Calculate the mean proportion and standard deviation of white lattice (lightness 100.0) within 100 images. For each word, remove the top lightness by the mean value of lightness and update the distribution of the remaining lightness accordingly.
# 
# ### 2. Mean and Std for Z-Score
# #### The second task uses 100 random words as a reference group to calculate the mean and standard deviation for each color. These statistics are used to express the intensity of the colors of the target word as a z-score.

# ## 1. White lattice

# %%
#Selection of 100 Random Words

wild_list = [
    'hemerocallis', 'archiannelid', 'ponce', 'malvales', 'wegener', 'seraphic', 'heroics', 
    'cordaites', 'taxaceae', 'coerebidae', 'seeder', 'leptotyphlopidae', 'entolomataceae', 
    'disclosure', 'psittacosis', 'spectroscope', 'explode', 'klyuchevskaya', 'scaled', 'transfiguration',
    'dingo', 'megadeath', 'commissar', 'monotone', 'cart', 'bugbear', 'parsonage', 'bathtub', 'earmuff', 
    'niqaabi', 'snowshoe', 'cuculiformes', 'chelicerae', 'aardvark', 'belemnitic', 'toby', 'enantiomorph', 
    'videocassette', 'psychologist', 'keratin', 'amphineura', 'orchid', 'baccarat', 'czarina', 'mitterrand',
    'mariner', 'kosteletzya', 'kutuzov', 'file', 'caring', 'unilateralism', 'latrodectus', 'hathaway', 
    'babylonian', 'hasdrubal', 'chorally', 'polygonum', 'festival', 'nectarine', 'clavichord', 'geophyte',
    'prowl', 'mint', 'competitive', 'mergus', 'cryptogramma', 'kilroy', 'pinkify', 'burner', 'watchtower', 
    'hurrah', 'tambour', 'palaemonidae', 'girder', 'excited', 'tumble', 'noisy', 'conima', 'aerophile', 
    'haley', 'tread', 'immigration', 'pontifex', 'timbrel', 'studbook', 'erring', 'cathedral', 'porringer', 
    'helodermatidae', 'hakham', 'proserpina', 'asteroid', 'declutch', 'isohel', 'carcinoma', 'boy', 
    'saprophytic', 'lupinus', 'malacostraca', 'tank'
]


# %%
#Proportion of lightness 100.0 within 100 random words
all_ratios= [0.3095693139698364, 0.3193071476697712, 0.3250299844343412, 0.31615169986263736, 0.2861118219037871, 0.29769897016528324, 0.30050751549020327, 0.3084683631408852, 0.31674616007391687, 0.3239374497881975, 0.322304875218436, 0.31387126720895414, 0.31625028617216117, 0.29757254464285715, 0.3194288958546571, 0.2960310923563386, 0.31740463751643294, 0.3176420024010371, 0.3093709529585583, 0.31695413701149006, 0.3179173820970696, 0.32160206146307446, 0.32355059067338593, 0.29573291358679743, 0.304515182223196, 0.3100291260477405, 0.3123533440976288, 0.33947705713554616, 0.32100918788382804, 0.32647605020491804, 0.29597705421179743, 0.33623467117059425, 0.28889186216003804, 0.3168609404731411, 0.3084981763986269, 0.29680974352061706, 0.31736241395522935, 0.302174912814052, 0.3006686463647959, 0.3204690013159819, 0.3035841501847922, 0.3219514800226641, 0.3248556695972101, 0.31690702565250767, 0.29885303730867346, 0.3140965070844289, 0.31622627065701153, 0.3111789533628513, 0.32298744448201494, 0.32976301740980135, 0.3157802447360725, 0.31202855111561495, 0.31965136761587953, 0.2983419834574934, 0.31965596996823825, 0.29541394026904516, 0.31575754062180145, 0.2958410933391762, 0.32069939146176857, 0.340453672744072, 0.2999038272805288, 0.33654951153311885, 0.31758751118353784, 0.3221408087295501, 0.3116198228907735, 0.327155032988915, 0.3126635708245546, 0.3194382167246489, 0.320323242473966, 0.31967957452344437, 0.32049433186195825, 0.3241643772893773, 0.3331976244436743, 0.3262532433332117, 0.3011550313275208, 0.30635204978271985, 0.31759407704552106, 0.3088280680575802, 0.3139636250183527, 0.3026968971162265, 0.3137324478633572, 0.3168882419130562, 0.30258222262579076, 0.31059666895604393, 0.32991978664548643, 0.3213354228290243, 0.3164485037309328, 0.31229935794542535, 0.3156048947214505, 0.32481905249305454, 0.3170634254948145, 0.30776126581005997, 0.3225402298196064, 0.33382273312984356, 0.31654476039840784, 0.3249257093558282, 0.31394658491272276, 0.2936221715528586, 0.28757240155677655, 0.3135696412389818]
#Now, when calculating the lightness_values for all random words, we trim the top 0.34329116918453534 (wild_mean) of lightness to remove the white lattice.
wild_mean = 0.3141835271415166
 # wild_std = 0.011389262932592533


# %%
# # Extracting Hue
# ## The average of the minimum chroma values containing 90% of each achromatic word is set as the threshold.

chroma_threshold = 15.930582082686795

def extracthue(resize_images_dir, search_term=""):
    """
    Extracts color information from an image file and converts it into various color spaces.

    Args:
        resize_images_dir (str): Base directory where the image files are located.
        search_term (str): Search term or filename (without extension) of the image.

    Returns:
        Returns:
        Updated Hue values considering the threshold.
        The key process is restoring as much as was removed by considering the distribution of the remaining values.
    """
    filename = search_term + ".png"
    img_url = os.path.join(resize_images_dir, filename)
    
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
            xn = 0.95047
            yn = 1.00000
            zn = 1.08883

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

    # Convert CIELAB values to CIELCH values
    lch_list = []

    for lab in lab_list:
        L, a, b = lab
        C = np.sqrt(a**2 + b**2)
        if C < chroma_threshold:
            H = -1  # Set H to -1 if chroma is too small to accurately determine hue
        else:
            H = np.degrees(np.arctan2(b, a))
            if H < 0:
                H += 360
        lch_list.append((L, C, H))

    # Extract H values
    h_values = leave_last_element(lch_list)
    h_minus_one_count = h_values.count(-1)

    # Calculate distribution of H values
    bins = np.linspace(0, 360, 361)
    h_distribution, _ = np.histogram([h for h in h_values if h != -1], bins=bins)

    # Calculate the ratio for each H value
    total_valid_h = sum(h_distribution)
    h_distribution_ratios = h_distribution / total_valid_h if total_valid_h > 0 else np.zeros(len(h_distribution))

    # Distribute the -1 H values
    distributed_counts = np.round(h_distribution_ratios * h_minus_one_count).astype(int)

    # Create new H value list
    new_h_values = [h for h in h_values if h != -1]
    for i, count in enumerate(distributed_counts):
        new_h_values.extend([i + 0.5] * count)

    return new_h_values



# %%
# # Extracting lightness
# ## Screenshot can be extracted except for the grid of white lattice.

def extractlightness(resize_images_dir, search_term=""):
    """
    Extracts color information from an image file and converts it into various color spaces.

    Args:
        resize_images_dir (str): Base directory where the resized image files are located.
        search_term (str): Search term or filename (without extension) of the image.

    Returns:
        Updated Lightness values considering the white lattice in the screenshot.
        The key process is restoring as much as was removed by considering the distribution of the remaining values.
    """
    filename = search_term + ".png"
    img_url = os.path.join(resize_images_dir, filename)
    
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
            xn = 0.95047
            yn = 1.00000
            zn = 1.08883

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


    # Extract L values and compute the threshold to cut off the top 34.329116918453534%
    l_values = [lab[0] for lab in lab_list]
    threshold = np.percentile(l_values, 100 - (wild_mean*100.0))
    l_values_cut = [l for l in l_values if l <= threshold]

    # Create a histogram of the remaining L values
    l_hist, bins = np.histogram(l_values_cut, bins=np.arange(0, 101, 1))

    # Determine how many values were cut
    num_cut = len(l_values) - len(l_values_cut)

    # Distribute the cut values according to the histogram distribution
    l_distribution = l_hist / l_hist.sum()
    additional_values = np.random.choice(bins[:-1], size=num_cut, p=l_distribution)

    # Combine and sort the new L values
    new_l_values = np.sort(np.concatenate((l_values_cut, additional_values)))

    return new_l_values


# # Color Range
# ## Create color ranges using synonyms.
# 
# ### First, create achromatic color ranges using lightness, and then create chromatic color ranges using hue.
# 
# ### Reference papers indicate that low chroma can make hue inaccurate, so we won't use ranges with too low chroma when creating chromatic color ranges. The threshold for chroma will be determined using achromatic words. 
# 
# ### There will be a chroma where achromatic words are predominantly clustered, and we will use this as the baseline.

# %%
def create_color_folders(base_path='./Visualizations/basic_color_visualization'):
    # List of colors
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'white', 'grey']
    
    # Create visualization folder if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create folders for each color
    for color in colors:
        color_path = os.path.join(base_path, color)
        if not os.path.exists(color_path):
            os.makedirs(color_path)
            print(f"Created folder: {color_path}")
        else:
            print(f"Folder already exists: {color_path}")


# %%
# Create a folder in advance to store visualizations for each color
create_color_folders()


# %%
red_word_list=["red","scarlet","vermilion","ruby","carmine"]
orange_word_list= ["orange","tangerine","marmalade","orangish","apricot"]
yellow_word_list = ["yellow","yellowish","yellowy","gold","golden"]
green_word_list = ["green","greenish","verdant","leafy","greenery"]
blue_word_list = ["blue","azure","cobalt","cerulean","ultramarine"]
purple_word_list = ["purple","violet","purply","purplish","amethyst"]
pink_word_list = ["pink","rosy","blushing","shellpink","rose"]
black_word_list = ["black","pitchblack","pitchdark","jetblack","blackish"]
grey_word_list = ["grey","silver","slategray","smokegray","silvery"]
white_word_list=["white","snowywhite","milkwhite","milkywhite","chalkwhite"]


# %%
color_nouns_list = [
    red_word_list,
    blue_word_list,
    green_word_list,
    yellow_word_list,
    purple_word_list,
    orange_word_list,
    black_word_list,
    white_word_list,
    grey_word_list,
    pink_word_list
]

color_nouns_list_1d=np.concatenate(color_nouns_list).tolist()


# # Setting and visualizing achromatic sections
# ## hue update function declaration

# %%
def calculate_total_frequency_in_range(range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h):
    # Generate bins for the specified range
    bins = np.linspace(range_start, range_end, num=(range_end - range_start) + 1)
    
    # Calculate frequency for each color
    first_counts, _ = np.histogram(first_h, bins=bins)
    second_counts, _ = np.histogram(second_h, bins=bins)
    third_counts, _ = np.histogram(third_h, bins=bins)
    fourth_counts, _ = np.histogram(fourth_h, bins=bins)
    fifth_counts, _ = np.histogram(fifth_h, bins=bins)    
    
    # Calculate the sum of frequencies for all lists
    total_counts = first_counts + second_counts + third_counts + fourth_counts + fifth_counts
    
    # Return all but the last element of bins (bins[:-1] represents the start of each interval)
    return bins[:-1], total_counts


# %%
from matplotlib.colors import to_rgb


def lch_to_xyz(l, c, h):
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    x_n, y_n, z_n = 0.95047, 1.0, 1.08883
    x = x_n * ((fx ** 3) if fx ** 3 > 0.008856 else (fx - 16/116) / 7.787) * 100
    y = y_n * ((fy ** 3) if fy ** 3 > 0.008856 else (fy - 16/116) / 7.787) * 100
    z = z_n * ((fz ** 3) if fz ** 3 > 0.008856 else (fz - 16/116) / 7.787) * 100
    return x, y, z


def xyz_to_rgb(x, y, z):
    x /= 100
    y /= 100
    z /= 100
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570
    r = 1.055 * (r ** (1/2.4)) - 0.055 if r > 0.0031308 else r * 12.92
    g = 1.055 * (g ** (1/2.4)) - 0.055 if g > 0.0031308 else g * 12.92
    b = 1.055 * (b ** (1/2.4)) - 0.055 if b > 0.0031308 else b * 12.92
    return max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))


# %% [markdown]
# ## Making Histogram for base color words

# %%
def analyze_number_distribution_hue_save(h_values, word, save_dir):
    counts, _ = np.histogram(h_values, bins=np.arange(361))  # 0–360 range for hue

    # Generate corresponding RGB colors from hue (fixed Lightness=60, Chroma=50)
    lch_colors = [lch_to_xyz(60, 50, h) for h in range(360)]
    rgb_colors = [xyz_to_rgb(*xyz) for xyz in lch_colors]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(range(360), counts, color=rgb_colors)
    ax.set_facecolor('#e0f7fa')
    ax.set_ylim(0, 1500)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_yticks([0, 500, 1000, 1500])
    ax.set_xlabel('Hue (0–360°)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_title(f'{word.capitalize()} Histogram', fontsize=20)
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{word}_hue.png")
    plt.savefig(save_path) 
    plt.close(fig)



# %%
def analyze_number_distribution_lightness_save(l_values, word, save_dir):
    counts, _ = np.histogram(l_values, bins=np.arange(101))

    gray_colors = [(i/100, i/100, i/100) for i in range(100)]

    fig, ax = plt.subplots(figsize=(9,6))
    ax.bar(range(100), counts, color=gray_colors)
    ax.set_facecolor('#e0f7fa')
    ax.set_ylim(0, 1500)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Lightness (0–100)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_title(f'{word.capitalize()} Histogram', fontsize=20)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{word}_lightness.png")
    plt.savefig(save_path)
    plt.close(fig)



# %%
resize_images_dir="./Images/resize_images"
base_save_dir = "./Visualizations/basic_color_visualization"

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# Define word lists for each basic color category
color_word_dict = {
    "red": ["red", "scarlet", "vermilion", "ruby", "carmine"],
    "orange": ["orange", "tangerine", "marmalade", "orangish", "apricot"],
    "yellow": ["yellow", "yellowish", "yellowy", "gold", "golden"],
    "green": ["green", "greenish", "verdant", "leafy", "greenery"],
    "blue": ["blue", "azure", "cobalt", "cerulean", "ultramarine"],
    "purple": ["purple", "violet", "purply", "purplish", "amethyst"],
    "pink": ["pink", "rosy", "blushing", "shellpink", "rose"],
    "black": ["black", "pitchblack", "pitchdark", "jetblack", "blackish"],
    "grey": ["grey", "silver", "slategray", "smokegray", "silvery"],
    "white": ["white", "snowywhite", "milkwhite", "milkywhite", "chalkwhite"]
}

# Categorize chromatic and achromatic colors
achromatic_colors = ["black", "grey", "white"]  # Use lightness histogram
chromatic_colors = [c for c in color_word_dict if c not in achromatic_colors]  # Use hue histogram

# Run analysis and save visualizations
for color, word_list in color_word_dict.items():
    save_dir = os.path.join(base_save_dir, color)
    os.makedirs(save_dir, exist_ok=True)
    
    for word in word_list:
        if color in achromatic_colors:
            l_values = extractlightness(resize_images_dir, search_term=word)
            if l_values is not None:
                analyze_number_distribution_lightness_save(l_values, word, save_dir)
            else:
                print(f"[{word}] lightness error")
        else:
            h_values = extracthue(resize_images_dir, search_term=word)
            if h_values is not None:
                analyze_number_distribution_hue_save(h_values, word, save_dir)
            else:
                print(f"[{word}] hue error")


# %%
def plot_union_frequency_bar(color, range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h):
    bins, union_counts = calculate_total_frequency_in_range(range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h)
    
    fig, ax = plt.subplots(figsize=(15, 12))  
    
    ax.bar(bins, union_counts,  width=1.0, align='edge',  color=color)
    
    # Remove the box (frame)
    ax.spines['top'].set_visible(False)    # Remove the top border
    ax.spines['right'].set_visible(False)  # Remove the right border
    ax.spines['bottom'].set_visible(True)  # Keep the bottom border
    ax.spines['left'].set_visible(True)    # Keep the left border
    ax.set_facecolor('#e0f7fa')
    
    if color == "black" or color == "white" or color == "grey":
        plt.xlabel('Lightness', fontsize=20)
    else:
        plt.xlabel('Hue', fontsize=20)
    plt.ylabel('Frequency', fontsize=20, labelpad=20)
    
    color_capitalized = color.capitalize()
    if color == "black" or color == "white" or color == "grey":
        plt.title(f'Common Lightness Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=14)
    else:
        plt.title(f'Common Hue Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=14)
 
    # Set the entire x-axis range
    plt.xlim(range_start, range_end)
    
    # Set the entire y-axis range
    plt.ylim(0, max(union_counts) * 1.1)
    
    # Adjust x-axis ticks
    plt.xticks(ticks=np.arange(range_start, range_end + 1, step=20)) 
    
    # Directory path to save
    directory = f"./Visualizations/basic_color_visualization/{color}"
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(f"./Visualizations/basic_color_visualization/{color}/save_intersection_plot_{color}.png")
    #plt.show()
    plt.close(fig) 
    return union_counts



def plot_union_frequency_smoothing_line(color, range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h):
    # Calculate the total frequency within the specified range
    bins, union_counts = calculate_total_frequency_in_range(range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h)
    
    # Create a DataFrame for bins and union counts
    df = pd.DataFrame({'bins': bins, 'union_counts': union_counts})
    # Apply a rolling mean to smooth the union counts
    df['smoothed'] = df['union_counts'].rolling(window=5, min_periods=1, center=True).mean()
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(15, 12))  
    # Plot the smoothed data
    ax.plot(df['bins'].to_numpy(), df['smoothed'].to_numpy(), color=color, linewidth=2)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set the background color of the plot area
    ax.set_facecolor('#e0f7fa')
    
    # Set the x-axis label based on the color
    if color == "black" or color == "white" or color == "grey":
        plt.xlabel('Lightness', fontsize=20)
    else:
        plt.xlabel('Hue', fontsize=20)
        
    # Set the y-axis label
    plt.ylabel('Frequency', fontsize=20, labelpad=20)
    
    # Capitalize the first letter of the color for the title
    color_capitalized = color.capitalize()
    
    # Set the plot title based on the color
    if color == "black" or color == "white" or color == "grey":
        plt.title(f'Smoothed Common Lightness Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=25)
    else:
        plt.title(f'Smoothed Common Hue Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=25)
    
    # Set the limits for the x-axis
    plt.xlim(range_start, range_end)
    # Set the limits for the y-axis
    plt.ylim(0, max(union_counts) * 1.1)
    
    # Set the x-axis ticks
    plt.xticks(ticks=np.arange(range_start, range_end + 1, step=20))

    # Save the plot as a PNG file
    plt.savefig(f"./Visualizations/basic_color_visualization/{color}/save_intersection_smoothing_line_{color}.png")
    # Close the plot to free up memory
    plt.close(fig)
    
    
    
def plot_union_frequency_bar_with_smoothing_line(color, range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h):
    # Calculate the total frequency within the specified range
    bins, union_counts = calculate_total_frequency_in_range(range_start, range_end, first_h, second_h, third_h, fourth_h, fifth_h)
    
    # Create a DataFrame for bins and union counts
    df = pd.DataFrame({'bins': bins, 'union_counts': union_counts})
    # Apply a rolling mean to smooth the union counts
    df['smoothed'] = df['union_counts'].rolling(window=5, min_periods=1, center=True).mean()
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(15, 12))  
    # Set the background color of the plot area
    ax.set_facecolor('#e0f7fa')
    
    # Plot the bar chart
    ax.bar(bins, union_counts, width=1.0, align='edge', color=color)
    # Plot the smoothed line
    ax.plot(df['bins'].to_numpy(), df['smoothed'].to_numpy(), color=color, linewidth=2)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set the x-axis label based on the color
    if color == "black" or color == "white" or color == "grey":
        plt.xlabel('Lightness', fontsize=20)
    else:
        plt.xlabel('Hue', fontsize=20)
        
    # Set the y-axis label
    plt.ylabel('Frequency', fontsize=20, labelpad=20)
    
    # Capitalize the first letter of the color for the title
    color_capitalized = color.capitalize()
    # Set the plot title based on the color
    if color == "black" or color == "white" or color == "grey":
        plt.title(f'Common Lightness Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=20)
    else:
        plt.title(f'Common Hue Frequency \nwithin Representative {color_capitalized} Ranges', fontsize=20)
    
    # Set the limits for the x-axis
    plt.xlim(range_start, range_end)
    # Set the limits for the y-axis
    plt.ylim(0, max(union_counts) * 1.1)
    
    # Set the x-axis ticks
    plt.xticks(ticks=np.arange(range_start, range_end + 1, step=20), fontsize=20)
    # Set the y-axis ticks
    plt.yticks(fontsize=20)

    # Save the plot as a PNG file
    plt.savefig(f"./Visualizations/basic_color_visualization/{color}/save_intersection_bar_with_smoothing_line_{color}.png")
    # Close the plot to free up memory
    plt.close(fig)



# %%
color_word_dict = {
    "red": ["red", "scarlet", "vermilion", "ruby", "carmine"],
    "orange": ["orange", "tangerine", "marmalade", "orangish", "apricot"],
    "yellow": ["yellow", "yellowish", "yellowy", "gold", "golden"],
    "green": ["green", "greenish", "verdant", "leafy", "greenery"],
    "blue": ["blue", "azure", "cobalt", "cerulean", "ultramarine"],
    "purple": ["purple", "violet", "purply", "purplish", "amethyst"],
    "pink": ["pink", "rosy", "blushing", "shellpink", "rose"],
    "black": ["black", "pitchblack", "pitchdark", "jetblack", "blackish"],
    "grey": ["grey", "silver", "slategray", "smokegray", "silvery"],
    "white": ["white", "snowywhite", "milkwhite", "milkywhite", "chalkwhite"]
}

achromatic_colors = ["black", "grey", "white"]  # Use lightness
chromatic_colors = [c for c in color_word_dict if c not in achromatic_colors]

# Dictionaries for visualization and unified data
achromatic_data = {}
chromatic_data = {}

# Iterate over all colors
for color, word_list in color_word_dict.items():
    save_dir = os.path.join(base_save_dir, color)
    os.makedirs(save_dir, exist_ok=True)

    # Choose extraction function and value range based on color type
    extractor = extractlightness if color in achromatic_colors else extracthue
    range_start, range_end = (0, 100) if color in achromatic_colors else (0, 360)

    # Extract values for each word
    value_lists = []
    for word in word_list:
        values = extractor(resize_images_dir, search_term=word)
        if values is None:
            print(f"[{word}] value extraction error")
        else:
            value_lists.append(values)

    # Skip visualization if fewer than 5 valid value sets
    if len(value_lists) < 5:
        print(f"[{color}] Less than 5 valid words → skipping visualization")
        continue

    # Use the first 5 value sets
    first, second, third, fourth, fifth = value_lists[:5]

    # Perform individual visualizations
    plot_union_frequency_bar(color, range_start, range_end, first, second, third, fourth, fifth)
    plot_union_frequency_smoothing_line(color, range_start, range_end, first, second, third, fourth, fifth)
    plot_union_frequency_bar_with_smoothing_line(color, range_start, range_end, first, second, third, fourth, fifth)

    print(f"[{color}] visualization complete.")

    # Save data for integrated visualization
    if color in achromatic_colors:
        achromatic_data[color] = value_lists[:5]
    else:
        chromatic_data[color] = value_lists[:5]


# %%
def plot_combined_total_frequency_with_smoothing_acrhoma(range_starts, range_ends, datasets, colors, titles, alpha=1.0):
    # Create a plot with a specific size
    fig, ax = plt.subplots(figsize=(16, 10))
    all_min_counts = []  # List to store min_counts values from all datasets

    # Loop through each dataset and plot the data
    for i, dataset in enumerate(datasets):
        range_start, range_end = range_starts[i], range_ends[i]
        # Calculate the total frequency within the specified range
        bins, union_counts = calculate_total_frequency_in_range(range_start, range_end, *dataset)
        # Plot the bar chart
        ax.bar(bins, union_counts, color=colors[i], width=1.0, align='edge', alpha=alpha, label=titles[i])
        
        # Create a DataFrame for bins and union counts
        df = pd.DataFrame({'bins': bins, 'union_counts': union_counts})
        # Apply a rolling mean to smooth the union counts
        df['smoothed'] = df['union_counts'].rolling(window=5, min_periods=1, center=True).mean()
        
        # Plot the smoothed line
        ax.plot(df['bins'].to_numpy(), df['smoothed'].to_numpy(), color=colors[i], linewidth=2)
        # Add the current dataset's min_counts values to the list
        all_min_counts.extend(union_counts)
    
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Ensure bottom and left spines are visible
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set the background color of the plot area
    ax.set_facecolor('#e0f7fa')   
    
    # Set the x-axis and y-axis labels
    plt.xlabel('Lightness', fontsize=20, labelpad=20)
    plt.ylabel('Frequency', fontsize=20, labelpad=20)
    # Set the plot title
    plt.title('Color Spectrum Frequency Comparison', fontsize=20, pad=25)
    
    # Set the limits for the x-axis and y-axis
    plt.xlim(0, 100)
    plt.ylim(0, max(all_min_counts) + 50)  # Add 50 to the maximum value of all_min_counts for better spacing
    
    # Set the x-axis ticks
    plt.xticks(ticks=np.arange(0, 101, step=10))
    
    # Adjust tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=14)
        
    # Add a legend to the plot
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig("./Visualizations/basic_color_visualization/save_combined_plot_achroma.png")  # Change the save path as needed
    # Close the plot to free up memory
    plt.close(fig)




# %%
# achromatic integrated visualization
if all(c in achromatic_data for c in ["black", "grey", "white"]):
    range_starts = [0, 0, 0]
    range_ends = [100, 100, 100]
    datasets = [
        achromatic_data["black"],
        achromatic_data["grey"],
        achromatic_data["white"]
    ]
    colors = ["black", "grey", "white"]
    titles = ["Black", "Grey", "White"]

    plot_combined_total_frequency_with_smoothing_acrhoma(
        range_starts, range_ends, datasets, colors, titles
    )


# %%

# Define color ranges (derived from primary colors and their synonyms)
achroma_ranges = {
        "black": [(0, 41.65384615384615)],
        'grey': [(41.65384615384615, 82.89746682750301)],
        'white': [(82.89746682750301, 100)]
    }


# %%
"""
Display the final graphs of the above three colors together on a single plane.
This will help identify the dominant regions for each color.
"""

def plot_combined_total_frequency_with_smoothing_chroma(range_starts, range_ends, datasets, colors, titles):
    fig, ax = plt.subplots(figsize=(16, 10))
    all_min_counts = []
    for i, dataset in enumerate(datasets):
        range_start, range_end = range_starts[i], range_ends[i]
        bins, union_counts = calculate_total_frequency_in_range(range_start, range_end, *dataset)
        ax.bar(bins, union_counts, color=colors[i], width=1.0, align='edge', alpha=0.5, label=titles[i])
        df = pd.DataFrame({'bins': bins, 'union_counts': union_counts})
        df['smoothed'] = df['union_counts'].rolling(window=5, min_periods=1, center=True).mean()

        ax.plot(df['bins'].to_numpy(), df['smoothed'].to_numpy(), color=colors[i], linewidth=2)
        all_min_counts.extend(union_counts)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    plt.xlabel('Hue', fontsize=20, labelpad=20)
    plt.ylabel('Frequency', fontsize=20, labelpad=20)
    plt.title('Color Spectrum Frequency Comparison', fontsize=20, pad=25)

    plt.xlim(0, 360)
    plt.ylim(0, max(all_min_counts) + 50)  

    plt.xticks(ticks=np.arange(0, 361, step=50))

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend()
    plt.savefig("./Visualizations/basic_color_visualization/save_combined_plot_chroma_bar_and_smoothing.png")
    #plt.show()
    plt.close(fig)


# %%
# List all 7 chromatic color keys
selected_colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

# Check if all required chromatic color data is available
if not all(c in chromatic_data for c in selected_colors):
    print("Some chromatic color groups are missing. Skipping chromatic combined plot.")
else:
    range_starts = [0] * len(selected_colors)
    range_ends = [360] * len(selected_colors)
    datasets = [chromatic_data[c] for c in selected_colors]
    colors = ["red", "orange", "gold", "green", "blue", "purple", "pink"]  # Use visually distinct actual colors
    titles = [c.capitalize() for c in selected_colors]

    plot_combined_total_frequency_with_smoothing_chroma(
        range_starts,
        range_ends,
        datasets,
        colors,
        titles
    )

    print("Full chromatic spectrum visualization complete.")


# %%
# Intervals determined based on the regions where each color is dominant
chroma_ranges = {
    "red": [(19.326959847036328, 45.67667984189723)],
    "orange": [(45.67667984189723, 80.90534979423869)],
    "yellow": [(80.90534979423869, 120.1811320754717)],
    "green": [(120.1811320754717, 203.28)],
    "blue": [(203.28, 293.86897590361446)],
    "purple": [(293.86897590361446, 331.2451923076923)],
    "pink": [(331.2451923076923, 360.00), (0.00, 19.326959847036328)]
}


# %%
def color_ratio_for_each_term(h_values, l_values):
    """
    Calculate the ratio of each chromatic and achromatic color for given hue and lightness values.

    Args:
        h_values (list): List of hue values.
        l_values (list): List of lightness values.

    Returns:
        tuple: A tuple containing dictionaries for combined color ratios, chromatic color ratios, and achromatic color ratios.
    """
    # Initialize counts for chromatic colors
    chroma_color_counts = {color: 0 for color in chroma_ranges}
    
    # Count the number of values falling within each chromatic color range
    for value in h_values:
        for color, ranges in chroma_ranges.items():
            for start, end in ranges:
                # Use modulo to ensure the value is within the 0-360 range and correctly handle circular intervals
                hue = value % 360
                if start <= hue <= end or (start > end and (hue >= start or hue <= end)):
                    chroma_color_counts[color] += 1
                    break

    # Calculate the ratio for each chromatic color
    total_h_values = len(h_values)
    chroma_color_ratio = {color: count / total_h_values for color, count in chroma_color_counts.items() if total_h_values > 0}
    
    
    # Initialize counts for achromatic colors
    achroma_color_counts = {color: 0 for color in achroma_ranges}

    # Count the number of values falling within each achromatic color range
    for value in l_values:
        for color, ranges in achroma_ranges.items():
            for start, end in ranges:
                if start <= value % 100 <= end:
                    achroma_color_counts[color] += 1
                    break

    # Calculate the ratio for each achromatic color
    achroma_color_ratio = {color: count / len(l_values) for color, count in achroma_color_counts.items()}

    # Merge chromatic and achromatic color ratios
    color_ratio_dict = {**chroma_color_ratio, **achroma_color_ratio}

    return color_ratio_dict, chroma_color_ratio, achroma_color_ratio

