#!/usr/bin/env python
# coding: utf-8
# %%


import os
import argparse
import webbrowser
import word2image as w2i
import calculation as cal
import visualization as viz
from IPython.display import IFrame, display


# %%


def create_directories():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Define the root directories and their respective subdirectories
    root_dirs = {
        'Images': ['original_images', 'resize_images', 'temp'],
        'Visualizations': ['basic_color_visualization', 'plot_visualization']
    }
    
    # Create each directory if it doesn't exist
    for root, subdirs in root_dirs.items():
        root_path = os.path.join(current_dir, root)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
            print(f"Root directory '{root}' created at {root_path}")
        else:
            print(f"Root directory '{root}' already exists at {root_path}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(root_path, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Subdirectory '{subdir}' created at {subdir_path}")
            else:
                print(f"Subdirectory '{subdir}' already exists at {subdir_path}")

create_directories()


# %%


def open_html_file(filepath):
    """Automatically opens an HTML file in the default web browser."""
    webbrowser.open('file://' + os.path.realpath(filepath))

def save_and_open_html(plot_html, term, plot_type):
    """Saves the plot HTML to a file and opens it in the browser."""
    if not os.path.exists('./Visualizations/plot_visualization'):
        os.makedirs('./Visualizations/plot_visualization')
    
    # HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Color Word Visualization</title>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0; 
            }}
        </style>
    </head>
    <body>
        <div class="plot-container">
            {plot_html}
        </div>
    </body>
    </html>
    """
    
    filepath = os.path.join('./Visualizations/plot_visualization', f'{term}_{plot_type}.html')
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"{plot_type.replace('_', ' ').title()} saved to {filepath}.")
    open_html_file(filepath)



# %%


graph_width = 500  
graph_height = 500  
graph_margin=dict(l=10, r=10, t=10, b=10)

def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up argument parser for command line interface
    parser = argparse.ArgumentParser(description="Calculate ratio, or visualize color for a term.")
    parser.add_argument('--term', type=str, required=True, help="The term to calculate or visualize.")
    parser.add_argument('--pixel', type=int, default=10000, help="The pixel of image")
    parser.add_argument('--option', type=str, choices=[
        'hue_scatter_plot', 'lightness_scatter_plot', 'hue_histogram', 'lightness_histogram',
        'color_ratio_hue', 'color_ratio_lightness', 'ratio'
    ], required=True, help="The calculation or visualization option.")
    parser.add_argument('--base_dir', type=str, default=current_file_dir, help="Base directory for images (for network visualization).")
    parser.add_argument('--check_text', dest='check_text', action='store_true', help="Enable text checking in images.")
    parser.add_argument('--no_check_text', dest='check_text', action='store_false', help="Disable text checking in images.")
    parser.set_defaults(check_text=True)

    args = parser.parse_args()
    
    term = args.term
    pixel = args.pixel
    option = args.option
    base_dir = args.base_dir
    threshold = args.threshold
    check_text = args.check_text

    # Specify the directory for generating image files
    original_images_dir = os.path.join(base_dir, 'Images', 'original_images')
    resize_images_dir = os.path.join(base_dir, 'Images', 'resize_images')
    temp_images_dir = os.path.join(base_dir, 'Images', 'temp')

    # Set global variables in the modules
    viz.graph_width = graph_width
    viz.graph_height = graph_height
    viz.graph_margin = graph_margin

    # Link the word to the image and save the corresponding image.
    w2i.word2image(term, pixel, original_images_dir, temp_images_dir, resize_images_dir, check_text=check_text)
    
    # Define color ranges (derived from primary colors and their synonyms)
    chroma_ranges = cal.chroma_ranges
    achroma_ranges = cal.achroma_ranges
    
    # Calculate required statistics using the calculator module
    lch_list = cal.extractColors_before_update(resize_images_dir, term)
    h_values = cal.extracthue(resize_images_dir, term)
    l_values = cal.extractlightness(resize_images_dir, term)
    
    color_ratio_dict, chroma_color_ratio, achroma_color_ratio = cal.color_ratio_for_each_term(h_values, l_values)
    
    print("Calculation completed.")
    
    # Create './Visualizations/plot_visualization' folder if it does not exist
    if not os.path.exists('./Visualizations/plot_visualization'):
        os.makedirs('./Visualizations/plot_visualization')
    
    # Process the option and generate the appropriate visualization or calculation
    if option == 'hue_scatter_plot':
        plot_html = viz.colorWordVisualization(lch_list, chroma_ranges, lightness=70)
        save_and_open_html(plot_html, term, 'hue_scatter_plot')
    elif option == 'lightness_scatter_plot':
        plot_html = viz.lightnessWordVisualization(lch_list, achroma_ranges)
        save_and_open_html(plot_html, term, 'lightness_scatter_plot')
    elif option == 'hue_histogram':
        plot_html = viz.analyze_number_distribution_hue(h_values)
        save_and_open_html(plot_html, term, 'hue_histogram')
    elif option == 'lightness_histogram':
        plot_html = viz.analyze_number_distribution_lightness(l_values)
        save_and_open_html(plot_html, term, 'lightness_histogram')
    elif option == 'color_ratio_hue':
        plot_html = viz.colorWordRatio_hue(color_ratio_dict, chroma_ranges)
        save_and_open_html(plot_html, term, 'color_ratio_hue')
    elif option == 'color_ratio_lightness':
        plot_html = viz.colorWordRatio_lightness(color_ratio_dict, achroma_ranges)
        save_and_open_html(plot_html, term, 'color_ratio_lightness')
    elif option == 'ratio':
        print(f"The color ratio for the term '{term}' is {color_ratio_dict}.")


if __name__ == "__main__":
    main()


# %%




