#!/usr/bin/env python
# coding: utf-8
# %%


import os
import base64
import numpy as np
import networkx as nx
from io import BytesIO
from collections import Counter 
from PIL import Image, ImageDraw
import plotly.graph_objects as go


# %%
graph_width = 800
graph_height = 800
graph_margin = dict(l=50, r=50, t=50, b=50)

# Define color ranges (derived from primary colors and their synonyms) through calculation.py
chroma_ranges = {
    "red": [(19.326959847036328, 45.67667984189723)],
    "orange": [(45.67667984189723, 80.90534979423869)],
    "yellow": [(80.90534979423869, 120.1811320754717)],
    "green": [(120.1811320754717, 203.28)],
    "blue": [(203.28, 293.86897590361446)],
    "purple": [(293.86897590361446, 331.2451923076923)],
    "pink": [(331.2451923076923, 360.00), (0.00, 19.326959847036328)]
}
achroma_ranges = {
        "black": [(0, 41.65384615384615)],
        'grey': [(41.65384615384615, 82.89746682750301)],
        'white': [(82.89746682750301, 100)]
    }

def lch_to_xyz(l, c, h):
    """단일 LCH 값을 XYZ 튜플로 변환"""
    h_rad = math.radians(h)
    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)
    
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    x_n, y_n, z_n = 0.95047, 1.0, 1.08883
    x = x_n * (fx ** 3 if fx ** 3 > 0.008856 else (fx - 16.0/116.0) / 7.787) * 100
    y = y_n * (fy ** 3 if fy ** 3 > 0.008856 else (fy - 16.0/116.0) / 7.787) * 100
    z = z_n * (fz ** 3 if fz ** 3 > 0.008856 else (fz - 16.0/116.0) / 7.787) * 100
    return x, y, z

def xyz_to_rgb(xyz_list):
    """XYZ 리스트를 통째로 받아 RGB 리스트로 변환 (리스트용 버전)"""
    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570]]
    
    rgb_list = []
    for x, y, z in xyz_list:
        x_norm, y_norm, z_norm = x / 100, y / 100, z / 100
        
        r = M[0][0] * x_norm + M[0][1] * y_norm + M[0][2] * z_norm
        g = M[1][0] * x_norm + M[1][1] * y_norm + M[1][2] * z_norm
        b = M[2][0] * x_norm + M[2][1] * y_norm + M[2][2] * z_norm
        
        # gamma correction
        r = 1.055 * (max(0, r) ** (1 / 2.4)) - 0.055 if r > 0.0031308 else r * 12.92
        g = 1.055 * (max(0, g) ** (1 / 2.4)) - 0.055 if g > 0.0031308 else g * 12.92
        b = 1.055 * (max(0, b) ** (1 / 2.4)) - 0.055 if b > 0.0031308 else b * 12.92
        
        rgb_list.append((max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))))
    return rgb_list

# %%
"""
Change data type of color ranges for visualization
"""
def convert_chroma_ranges_to_hue_categorizations(chroma_ranges):
    boundaries = []
    for ranges in chroma_ranges.values():
        for start, end in ranges:
            boundaries.append(start)
            boundaries.append(end)
    
    # Remove 0.0 values
    boundaries = [b for b in boundaries if b != 0.0]
    
    # Add the last two values together
    if len(boundaries) > 1:
        last_two_sum = boundaries[-1] + boundaries[-2]
        boundaries = boundaries[:-2] + [last_two_sum]
    
    # Remove duplicates and sort
    unique_boundaries = sorted(set(boundaries))
    
    return unique_boundaries

def convert_achroma_ranges_to_lightness_categorizations(achroma_ranges):
    lightness_categorizations = [0]  # Add the starting value
    for ranges in achroma_ranges.values():
        for range_tuple in ranges:
            end_value = range_tuple[1] / 100  # Convert to ratio by dividing by 100
            lightness_categorizations.append(end_value)
    # Set the last boundary value to 1.0
    lightness_categorizations[-1] = 1.0
    return lightness_categorizations



# %%


def colorWordVisualization(lch_list, hue_range=chroma_ranges,lightness=70):
    """Visualize color words using LCH color space.

    Args:
        lch_list (list): List of LCH color values.
        lightness (float): Lightness value to set for all colors.

    Returns:
        str: HTML representation of the plot.
    """
    
    hue_categorizations = convert_chroma_ranges_to_hue_categorizations(hue_range)
    
    # Convert LCH to the given lightness
    lch = [(lightness, c, h) for _, c, h in lch_list]
    xyz = [lch_to_xyz(l, c, h) for l, c, h in lch]
    rgb = xyz_to_rgb(xyz)

    # Calculate the frequency of points
    point_counts = Counter((round(h, 2), round(c, 2)) for _, c, h in lch_list)

    # Generate colors for the ring (varying by the H value in LCH)
    theta_full_range = np.linspace(0, 2 * np.pi, 360)
    lch_colors_ring = [(70, 100, h) for h in np.linspace(0, 360, 360)]
    xyz_colors_ring = [lch_to_xyz(l, c, h) for l, c, h in lch_colors_ring]
    colors_ring = ['rgba({}, {}, {}, 1)'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in xyz_to_rgb(xyz_colors_ring)]

    # Draw the inner coloring (ring) with varying colors based on H value in LCH
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
    for (_, c, h), color in zip(lch_list, rgb):
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
            text=[f'c: {c}, h: {h:.2f}°']  # Set text to display on hover
        ))

    # Draw lines and labels for specific H values
    hues_to_mark = hue_categorizations
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
            r=[1.35],  # Position value further outward
            theta=[hue],
            mode='text',
            text=[f'{hue:.2f}°'],
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hoverinfo='none'
        ))

    # Set layout
    layout = go.Layout(
        width=graph_width,  
        height=graph_height,
        margin=graph_margin,
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor='#e0f7fa'  # Set inner background color
        ),
        showlegend=False
    )

    # Combine data
    data = [ring_trace] + cluster_traces + line_traces + text_traces

    # Create the plot
    fig = go.Figure(data=data, layout=layout)

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# %%


def analyze_number_distribution_hue(h_values):
    """
    Analyzes the distribution of hue values in a given list.

    Args:
        numbers (list): List of hue values to analyze.

    Returns:
        str: HTML representation of the bar plot showing the distribution.
    """
    
    # Calculate the number of values in each hue bin
    counts, _ = np.histogram(h_values, bins=np.arange(361))

    # Calculate RGB colors based on H values in LCH color space
    lch_colors = [(50, 50, h) for h in range(360)]
    xyz_colors = [lch_to_xyz(l, c, h) for l, c, h in lch_colors]
    rgb_colors = xyz_to_rgb(xyz_colors)
    
    # Set colors for the bar plot
    bar_colors = ['rgba({}, {}, {}, 1)'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in rgb_colors]

    # Visualize the distribution as a bar plot
    bars = go.Bar(
        x=np.arange(360),
        y=counts,
        marker=dict(color=bar_colors),
        hoverinfo='text',
        hovertemplate='Hue: %{x} <br>Count: %{y}<extra></extra>',  # Text to display on hover
        name=''  # Set trace name to an empty string
    )

    fig = go.Figure(data=[bars])

    fig.update_layout(
        xaxis_title='Hue (H)',
        yaxis_title='Count',
        xaxis=dict(range=[0, 360]),
        yaxis=dict(range=[0, max(counts)]),
        template='plotly_white',
        width=graph_width,  
        height=graph_height,
        margin=graph_margin,
        plot_bgcolor='#e0f7fa',
    )

    # Add hover effects (remove outlines from bars)
    fig.update_traces(marker=dict(line=dict(width=0)))

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# %%


def lightnessWordVisualization(lch_list,lightness_ranges):
    """
    Visualizes lightness values from a list of LCH color values.

    Args:
        lch_list (list): List of LCH color values.

    Returns:
        str: HTML representation of the plot.
    """
    # Fix H values to 0 and set C values to 0 to calculate colors
    lch_for_color = [(l, 0, 0) for l, c, h in lch_list]
    xyz_for_color = [lch_to_xyz(l, c, h) for l, c, h in lch_for_color]
    rgb_for_color = xyz_to_rgb(xyz_for_color)
    
    lightness_categorizations=convert_achroma_ranges_to_lightness_categorizations(lightness_ranges)
    
    # Calculate the frequency of points
    point_counts = Counter((round(l, 2), round(c, 2)) for l, c, h in lch_list)
    
    # Generate data
    c_vals = [c for _, c, _ in lch_list]
    l_vals = [l for l, _, _ in lch_list]
    
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
            text=[f'L: {l}, C: {c}']  # Set text to display on hover
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
        opacity=1.0  # Set color bar background to opaque
    )

    # Set graph size and layout
    title_standoff_value = graph_width * 0.01  # Dynamically set title standoff based on graph width

    layout = go.Layout(
        xaxis=dict(domain=[0.26, 1], range=[0, 100], title='Chroma (C)', showgrid=False, gridcolor='white'),
        yaxis=dict(domain=[0, 1], range=[0, 100], showgrid=True, zeroline=False, showticklabels=False, ticks='', gridcolor='white'),
        xaxis2=dict(domain=[0.18, 0.25], anchor='y2', showgrid=False, zeroline=False, ticks='', showticklabels=False),
        yaxis2=dict(domain=[0, 1], anchor='x2', showgrid=True, zeroline=False, ticks='', range=[0, 100], showticklabels=True, title='Lightness (L)', title_standoff=title_standoff_value, gridcolor='white'),
        plot_bgcolor='#e0f7fa',
        width=graph_width,  
        height=graph_height,
        margin=graph_margin
    )

    # Add horizontal blue lines at y=0.4165 and y=0.8290
    line_traces = [
        go.Scatter(
            x=[0, 100],
            y=[lightness_categorizations[1] * 100, lightness_categorizations[1] * 100],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='none',
            xaxis='x',
            yaxis='y'
        ),
        go.Scatter(
            x=[0, 100],
            y=[lightness_categorizations[2] * 100, lightness_categorizations[2] * 100],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='none',
            xaxis='x',
            yaxis='y'
        ),
        go.Scatter(
            x=[0, 100],
            y=[lightness_categorizations[1] * 100, lightness_categorizations[1] * 100],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='none',
            xaxis='x2',
            yaxis='y2'
        ),
        go.Scatter(
            x=[0, 100],
            y=[lightness_categorizations[2] * 100, lightness_categorizations[2] * 100],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='none',
            xaxis='x2',
            yaxis='y2'
        )
    ]

    fig = go.Figure(data=cluster_traces + [color_bar_trace] + line_traces, layout=layout)

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# %%


def analyze_number_distribution_lightness(l_values):
    """
    Analyzes the distribution of lightness values in a given list.

    Args:
        numbers (list): List of lightness values to analyze.

    Returns:
        str: HTML representation of the bar plot showing the distribution.
    """
    # Calculate the number of values in each bin
    counts, _ = np.histogram(l_values, bins=np.arange(101))

    # Create a colormap from white to black
    colors = ['rgba({}, {}, {}, 1)'.format(int(i * 2.55), int(i * 2.55), int(i * 2.55)) for i in range(101)]

    # Plot the bar graph using Plotly
    bars = go.Bar(
        x=np.arange(100),
        y=counts,
        marker=dict(color=colors),
        hoverinfo='text',
        hovertemplate='Lightness: %{x} <br>Count: %{y}<extra></extra>',  # Text to display on hover
        name=''  # Set trace name to an empty string
    )

    fig = go.Figure(data=[bars])

    fig.update_layout(
        xaxis_title='Lightness',
        yaxis_title='Count',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, max(counts)]),
        template='plotly_white',
        width=graph_width,  
        height=graph_height,
        margin=graph_margin,
        plot_bgcolor='#e0f7fa',  # Set background color
    )

    # Add hover effects (outline bars)
    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# %%


def colorWordRatio_hue(color_ratio, hue_range=chroma_ranges):
    """
    Visualizes the color word ratio based on hue ranges.

    Args:
        color_ratio (dict): Dictionary with color ratios for each term.
        hue_range (list): List of hue ranges for different colors.

    Returns:
        str: HTML representation of the plot.
    """
    # Define the hues and corresponding colors
    
    hue_categorizations = convert_chroma_ranges_to_hue_categorizations(hue_range)
    
    hues_to_mark = hue_categorizations
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    
    # Extract ratios for each color
    ratios = [color_ratio[color] for color in colors]

    # Initialize the figure
    fig = go.Figure()

    max_radius = 1.0
    total_area = np.pi * max_radius**2

    result_areas = {}

    # Create the filled sectors for each color
    for i, (color, ratio) in enumerate(zip(colors, ratios)):
        start_angle = np.radians(hues_to_mark[i])
        end_angle = np.radians(hues_to_mark[i + 1])
        
        theta = np.linspace(np.degrees(start_angle), np.degrees(end_angle), 100)
        
        # Calculate the area and radius for each sector based on the ratio
        sector_area = ratio * total_area
        sector_angle = end_angle - start_angle
        r = np.sqrt(sector_area / (np.pi * (sector_angle / (2 * np.pi))))  # Calculate radius within the sector
        
        # Create radius array
        radius_arr = np.full_like(theta, r)
        
        # Fill the sector
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate(([0], radius_arr, [0])),  # Start and end at 0
            theta=np.concatenate(([theta[0]], theta, [theta[-1]])),
            fill='toself',
            fillcolor=color,
            line=dict(color='black'),
            hoverinfo='text',  # Show hover info only for the filled area
            text=f'{color}: {ratio * 100:.2f}%',  # Set text to display ratio as a percentage
            showlegend=False
        ))

        # Add the result to the dictionary
        result_areas[color] = sector_area / total_area

    # Draw the inner circle with dashed lines
    fig.add_trace(go.Scatterpolar(
        r=[max_radius] * 100,
        theta=np.linspace(0, 360, 100),
        mode='lines',
        line=dict(color='black', dash='dot'),
        hoverinfo='skip',  # Remove hover info for the inner circle
        showlegend=False
    ))

    # Draw the dividing lines and angle labels
    extended_radius = max_radius * 1.5  # Extend the trend lines
    for hue in hues_to_mark[:-1]:
        hue_deg = hue % 360
        fig.add_trace(go.Scatterpolar(
            r=[0, extended_radius],
            theta=[hue_deg, hue_deg],
            mode='lines',
            line=dict(color='black'),
            hoverinfo='skip',  # Remove hover info for the lines
            showlegend=False
        ))
        # Add angle labels
        fig.add_trace(go.Scatterpolar(
            r=[extended_radius + 0.3],  # Move angle labels outward
            theta=[hue_deg],
            mode='text',
            text=[f'{hue:.2f}°'],
            textposition='top center',
            hoverinfo='skip',  # Remove hover info for the text
            showlegend=False
        ))

    # Update the layout
    fig.update_layout(
        width=graph_width,  
        height=graph_height,
        margin=graph_margin,       
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(showticklabels=False),
            bgcolor='#e0f7fa'
        ),
        showlegend=False,  # Remove legend

    )

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html


# %%

def colorWordRatio_lightness(color_ratio, lightness_ranges=achroma_ranges):
    """
    Visualizes the color word ratio based on lightness ranges.

    Args:
        color_ratio (dict): Dictionary with color ratios for each term.
        lightness_ranges (list): List of lightness ranges for different colors.

    Returns:
        str: HTML representation of the plot.
    """
    # Get the ratios for achromatic colors
    black_ratio = color_ratio.get('black', 0)
    grey_ratio = color_ratio.get('grey', 0)
    white_ratio = color_ratio.get('white', 0)

    lightness_categorizations=convert_achroma_ranges_to_lightness_categorizations(lightness_ranges)
    
    # Define the positions of the dividing lines
    y_ranges = lightness_categorizations
    line_positions = lightness_categorizations[1:3]

    fig = go.Figure()

    # Set background color
    fig.update_layout(
        plot_bgcolor='#e0f7fa',
        width=graph_width,  
        height=graph_height,
    )

    # Draw bar graphs for each achromatic color
    colors_rect = ['black', 'grey', 'white']
    color_names = ['Black', 'Grey', 'White']
    ratios = [black_ratio, grey_ratio, white_ratio]

    for i, (color, color_name, ratio) in enumerate(zip(colors_rect, color_names, ratios)):
        fig.add_trace(go.Bar(
            x=[ratio],
            y=[(y_ranges[i] + y_ranges[i+1]) / 2],
            width=[y_ranges[i+1] - y_ranges[i]],
            marker_color=color,
            orientation='h',
            showlegend=False,
            hovertemplate=f'{color_name}: {ratio*100:.1f}%<extra></extra>'
        ))

    # Draw dividing lines
    for pos in line_positions:
        fig.add_shape(type="line",
                      x0=0, y0=pos, x1=1, y1=pos,
                      line=dict(color="black", width=1.5))

    # Add axis labels
    fig.update_layout(
        width=graph_width,  
        height=graph_height,
        margin=graph_margin,  
        xaxis=dict(showticklabels=False),
        yaxis=dict(
            tickmode='array',
            tickvals=[0] + line_positions + [1],
            ticktext=[f'{pos:.2f}' for pos in [0] + line_positions + [1]],
            tickfont=dict(size=10),
            range=[0, 1]
        ),

    )

    # Convert the plot to HTML and return
    plot_html = fig.to_html(full_html=False)
    return plot_html

