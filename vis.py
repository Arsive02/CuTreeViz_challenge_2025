import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.collections import PatchCollection

# Read and clean data
df = pd.read_csv('data/Data_Viz_Challenge_2025-UCB_Trees.csv')
df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
df['Canopy Spread'] = pd.to_numeric(df['Canopy Spread'], errors='coerce')

# CU Boulder colors
CU_GOLD = '#CFB87C'
CU_BLACK = '#000000'

# Create directory
import os
if not os.path.exists('tree_viz'):
    os.makedirs('tree_viz')

def create_deciduous_tree(x, y, width, height):
    """Create a realistic deciduous tree shape with trunk and crown"""
    # Trunk dimensions
    trunk_width = width * 0.15
    trunk_height = height * 0.25
    
    # Crown dimensions
    crown_width = width
    crown_height = height * 0.75
    
    # Create trunk vertices
    trunk = [
        (x - trunk_width/2, y),  # Bottom left
        (x - trunk_width/2, y + trunk_height),  # Top left
        (x + trunk_width/2, y + trunk_height),  # Top right
        (x + trunk_width/2, y),  # Bottom right
    ]
    
    # Create crown using multiple curved sections
    crown_base_y = y + trunk_height
    crown_center_x = x
    crown_center_y = y + height
    
    # Create a more natural looking crown with multiple curved sections
    theta = np.linspace(0, 2*np.pi, 100)
    r = crown_width/2
    
    # Create slightly random variations for more natural look
    variations = 0.15 * np.sin(5*theta)
    
    crown_x = crown_center_x + (r + variations) * np.cos(theta)
    crown_y = crown_base_y + crown_height * (0.5 + 0.5 * np.sin(theta - np.pi/2))
    
    # Combine vertices
    verts = trunk + list(zip(crown_x, crown_y)) + [trunk[0]]
    
    return Path(verts)

def create_coniferous_tree(x, y, width, height):
    """Create a realistic coniferous (pine) tree shape"""
    # Trunk dimensions
    trunk_width = width * 0.1
    trunk_height = height * 0.2
    
    # Create trunk vertices
    trunk = [
        (x - trunk_width/2, y),
        (x - trunk_width/2, y + trunk_height),
        (x + trunk_width/2, y + trunk_height),
        (x + trunk_width/2, y),
    ]
    
    # Create multiple triangle sections for the pine shape
    sections = 4
    section_height = (height - trunk_height) / sections
    
    vertices = trunk
    
    for i in range(sections):
        section_width = width * (1 - i * 0.2)
        section_y = y + trunk_height + i * section_height
        
        # Add slight random variation to make it look more natural
        left_variation = np.random.uniform(-0.05, 0.05) * section_width
        right_variation = np.random.uniform(-0.05, 0.05) * section_width
        
        vertices.extend([
            (x - section_width/2 + left_variation, section_y),
            (x, section_y + section_height),
            (x + section_width/2 + right_variation, section_y),
        ])
    
    return Path(vertices)

def create_tree_icon(x, y, width, height, tree_type):
    """Create a custom tree shape based on tree type"""
    if tree_type == 'CONIFEROUS':
        # Triangle shape for coniferous
        points = [
            [x - width/2, y],
            [x, y + height],
            [x + width/2, y],
            [x - width/2, y]
        ]
    else:
        # Round shape for deciduous
        theta = np.linspace(0, 2*np.pi, 30)
        points = [[x + width/2 * np.cos(t), y + height/2 + height/2 * np.sin(t)] for t in theta]
        points.append(points[0])  # Close the shape
    
    return np.array(points)

def create_pine_tree(x, y, width, height, style='pine'):
    """Create a pine tree shape for visualization"""
    if style == 'pine':
        # Create triangular sections for pine tree
        sections = 3
        section_height = height * 0.25
        trunk_width = width * 0.15
        trunk_height = height * 0.2
        
        verts = [(x - trunk_width/2, y),  # Start at trunk base
                 (x - trunk_width/2, y + trunk_height),  # Trunk left
                 (x + trunk_width/2, y + trunk_height),  # Trunk right
                 (x + trunk_width/2, y)]  # Back to base
        
        # Add triangular sections
        for i in range(sections):
            section_width = width * (1 - i * 0.2)
            section_y = y + trunk_height + i * section_height
            verts.extend([
                (x - section_width/2, section_y),
                (x, section_y + section_height),
                (x + section_width/2, section_y)
            ])
            
    return Path(verts)

def create_tree_histogram(data, bins, ax, color):
    """Create a histogram in the shape of a tree"""
    hist, bin_edges = np.histogram(data, bins=bins)
    max_height = hist.max()
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Create tree shapes for each bin
    for i, count in enumerate(hist):
        if count > 0:
            height = count / max_height
            x = bin_edges[i] + bin_width/2
            tree = create_pine_tree(x, 0, bin_width*0.8, height)
            patch = patches.PathPatch(tree, facecolor=color, alpha=0.6)
            ax.add_patch(patch)
    
    return hist, bin_edges

# 1. Tree-shaped Height Distribution
plt.figure(figsize=(15, 8))
ax = plt.gca()

# Create separate histograms for each tree type
colors = {'DECIDUOUS': '#2D5A27', 'CONIFEROUS': '#1B4332'}
for tree_type, color in colors.items():
    data = df[df['Tree Type'] == tree_type]['Height'].dropna()
    hist, bins = create_tree_histogram(data, bins=15, ax=ax, color=color)

plt.title('Height Distribution by Tree Type\n(Each bar is a tiny tree!)', pad=20)
plt.xlabel('Height (feet)')
plt.ylabel('Count')
custom_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
plt.legend(custom_handles, colors.keys(), title='Tree Type')
plt.savefig('tree_viz/1_creative_height_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Tree Species Diversity Analysis
species_by_type = df.groupby(['Tree Type', 'Species']).size().reset_index(name='count')
species_diversity = species_by_type.groupby('Tree Type').agg({
    'Species': 'count',
    'count': ['sum', 'mean', 'std']
}).round(2)

# Create a sunburst chart for species composition
fig = px.sunburst(species_by_type, 
                  path=['Tree Type', 'Species'], 
                  values='count',
                  color='Tree Type',
                  color_discrete_map={'DECIDUOUS': '#2D5A27', 'CONIFEROUS': '#1B4332'})
fig.update_layout(title='Tree Species Composition')
fig.write_html('tree_viz/2_species_sunburst.html')

# 3. Growth Pattern Analysis

# CU Boulder colors
CU_GOLD = '#CFB87C'
CU_BLACK = '#000000'
TREE_COLORS = {'DECIDUOUS': '#2D5A27', 'CONIFEROUS': '#1B4332'}

plt.figure(figsize=(12, 8))

# Create separate scatter plots and trend lines for each tree type
for tree_type, color in TREE_COLORS.items():
    # Filter data for current tree type and drop NaN values
    type_data = df[df['Tree Type'] == tree_type].dropna(subset=['Height', 'Canopy Spread'])
    
    # Create scatter plot
    plt.scatter(type_data['Height'], type_data['Canopy Spread'], 
               c=color, alpha=0.6, s=100, label=f'{tree_type} Trees')
    
    # Calculate and plot trend line
    if len(type_data) > 1:  # Only fit if we have at least 2 points
        z = np.polyfit(type_data['Height'], type_data['Canopy Spread'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(type_data['Height'].min(), type_data['Height'].max(), 100)
        plt.plot(x_range, p(x_range), '--', color=CU_GOLD, linewidth=2)

plt.title('Tree Growth Pattern: Height vs Canopy Spread', fontsize=14)
plt.xlabel('Height (feet)', fontsize=12)
plt.ylabel('Canopy Spread (feet)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('tree_viz/1_growth_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# Create tree type distribution visualization
plt.figure(figsize=(15, 8))
type_counts = df['Tree Type'].value_counts()

for i, (tree_type, count) in enumerate(type_counts.items()):
    # Create tree shape
    tree = create_tree_icon(i, 0, 0.8, count/type_counts.max() * 5, tree_type)
    plt.fill(tree[:,0], tree[:,1], color=TREE_COLORS[tree_type], alpha=0.7,
             label=f'{tree_type} ({count} trees)')
    
    # Add count text
    plt.text(i, -0.5, str(count), ha='center', va='top', fontsize=12)

plt.title('Distribution of Tree Types\n(Shape represents tree type)', fontsize=14, pad=20)
plt.legend()
plt.axis('equal')
plt.axis('off')
plt.savefig('tree_viz/2_tree_type_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Height Distribution by Species (Top 10)
plt.figure(figsize=(15, 8))
species_height = df.groupby('Species')['Height'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)

for i, (species, data) in enumerate(species_height.iterrows()):
    # Create custom tree shape
    height = data['mean'] / species_height['mean'].max() * 5
    tree = create_tree_icon(i, 0, 0.8, height, 'DECIDUOUS' if height > 2.5 else 'CONIFEROUS')
    plt.fill(tree[:,0], tree[:,1], 
             color=TREE_COLORS['DECIDUOUS'] if height > 2.5 else TREE_COLORS['CONIFEROUS'],
             alpha=0.7)
    
    # Add labels
    plt.text(i, -0.2, species.split()[-1], ha='center', va='top', rotation=45)
    plt.text(i, height/2, f"{data['mean']:.1f} ft", ha='center', va='center', color='white')

plt.title('Average Height of Top 10 Most Common Species\n(Tree size represents height)', fontsize=14, pad=20)
plt.axis('equal')
plt.axis('off')
plt.savefig('tree_viz/3_species_height.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Spatial Density Analysis
from scipy.stats import gaussian_kde

# Create density map
plt.figure(figsize=(12, 8))
x = df['Longitude'].values
y = df['Latitude'].values
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x, y, c=z, s=50, alpha=0.5, cmap='YlOrBr')
plt.colorbar(label='Tree Density')
plt.title('Tree Density Heat Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('tree_viz/4_density_map.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Height Distribution Over Space
fig = go.Figure()

# Add scatter plot with height-based coloring
fig.add_trace(go.Scattermapbox(
    lat=df['Latitude'],
    lon=df['Longitude'],
    mode='markers',
    marker=dict(
        size=10,
        color=df['Height'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Height (ft)')
    ),
    text=df.apply(lambda x: f"Height: {x['Height']}ft<br>Type: {x['Tree Type']}<br>Species: {x['Species']}", axis=1),
    hoverinfo='text'
))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(lat=df['Latitude'].mean(), lon=df['Longitude'].mean()),
        zoom=14
    ),
    title='Tree Height Distribution Across Campus'
)
fig.write_html('tree_viz/5_height_map.html')

# 4. Interactive Species Distribution Map
df_clean = df.dropna(subset=['Latitude', 'Longitude', 'Height'])
fig = px.scatter_map(df_clean,
                     lat='Latitude',
                     lon='Longitude',
                     color='Tree Type',
                     size='Height',
                     hover_data=['Species', 'Height', 'Canopy Spread'],
                     color_discrete_map=TREE_COLORS,
                     title='Tree Distribution on Campus')

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        zoom=14.5,
        center=dict(lat=df_clean['Latitude'].mean(),
                   lon=df_clean['Longitude'].mean())
    ),
    showlegend=True,
    title_x=0.5
)
fig.write_html('tree_viz/4_campus_map.html')


# Create an informative HTML report
html_content = f"""
<html>
<head>
    <title>CU Boulder's Living Forest: A Data Story</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: {CU_BLACK};
            background-color: #f8f9fa;
        }}
        .header {{
            background-color: {CU_BLACK};
            color: {CU_GOLD};
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .story-section {{
            margin: 40px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 2px solid {CU_GOLD};
        }}
        .insight-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid {CU_GOLD};
            margin: 10px 0;
        }}
        .key-stat {{
            font-size: 24px;
            color: {CU_BLACK};
            text-align: center;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CU Boulder's Living Forest</h1>
        <p>A Data-Driven Journey Through Our Campus Trees</p>
    </div>

    <div class="story-section">
        <h2>Growth Patterns</h2>
        <img src="1_growth_pattern.png" width="100%">
        <div class="insight-box">
            <h3>Key Insights:</h3>
            <ul>
                <li>Deciduous trees generally show a wider spread relative to height</li>
                <li>Coniferous trees maintain a more columnar growth pattern</li>
                <li>Average height-to-spread ratio varies significantly between species</li>
            </ul>
        </div>
    </div>

    <div class="story-section">
        <h2>Tree Population</h2>
        <img src="2_tree_type_dist.png" width="100%">
        <div class="insight-box">
            <h3>Distribution Highlights:</h3>
            <ul>
                <li>Total trees on campus: {len(df)}</li>
                <li>Predominant type: {df['Tree Type'].mode()[0].title()}</li>
                <li>Number of species: {df['Species'].nunique()}</li>
            </ul>
        </div>
    </div>

    <div class="story-section">
        <h2>Species Height Comparison</h2>
        <img src="3_species_height.png" width="100%">
        <div class="insight-box">
            <h3>Height Analysis:</h3>
            <ul>
                <li>Tallest species average: {df.groupby('Species')['Height'].mean().max():.1f} feet</li>
                <li>Most common species: {df['Species'].mode()[0]}</li>
            </ul>
        </div>
    </div>

    <div class="story-section">
        <h2>Campus Distribution</h2>
        <iframe src="4_campus_map.html" width="100%" height="600px" frameborder="0"></iframe>
        <div class="insight-box">
            <h3>Spatial Patterns:</h3>
            <ul>
                <li>Trees are distributed across {df['Latitude'].nunique()} distinct latitude points</li>
                <li>Clear clustering patterns around major campus areas</li>
                <li>Mixed species distribution throughout campus</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

with open('tree_viz/tree_story.html', 'w') as f:
    f.write(html_content)

print("Story complete! Check the 'tree_viz' directory for the visual narrative.")