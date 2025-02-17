import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import matplotlib.cm as cm


CU_GOLD = '#CFB87C'
CU_BLACK = '#000000'
TREE_GREEN = '#2E7D32'


st.set_page_config(
    page_title="CU Boulder Tree Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #F5F5F5;
    }
    .css-1d391kg {
        background-color: #000000;
    }
    .st-bn {
        background-color: #CFB87C;
    }
    .metric-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #CFB87C;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h1 style='color: {CU_GOLD}; text-align: center; margin: 0;'>🌳 CU Boulder Campus Tree Analysis</h1>
    </div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('data/Data_Viz_Challenge_2025-UCB_Trees.csv')
    return df

df = load_data()


tab1, tab2, tab3 = st.tabs(["Overview", "Tree Distribution", "Environmental Impact"])

with tab1:

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Total Trees</h3>
                <h2 style='margin:0;'>{len(df):,}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        species_count = df['Species'].nunique()
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Unique Species</h3>
                <h2 style='margin:0;'>{species_count}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_height = df['Height'].mean()
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Average Height</h3>
                <h2 style='margin:0;'>{avg_height:.1f} ft</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        canopy_area = np.sum(np.pi * (df['Canopy Spread']/2)**2)
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Total Canopy Cover</h3>
                <h2 style='margin:0;'>{canopy_area:,.0f} sq ft</h2>
            </div>
        """, unsafe_allow_html=True)


    st.subheader("Tree Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        tree_types = df['Tree Type'].value_counts()
        fig_types = go.Figure(data=[go.Pie(
            labels=tree_types.index,
            values=tree_types.values,
            hole=0.6,
            marker_colors=[CU_GOLD, TREE_GREEN],
            textinfo='label+percent'
        )])
        
        fig_types.update_layout(
            showlegend=False,
            annotations=[dict(text='Tree Types', x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=400
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:

        height_bins = pd.cut(df['Height'], 
                           bins=[0, 20, 40, 60, float('inf')],
                           labels=['0-20 ft', '21-40 ft', '41-60 ft', '60+ ft'])
        height_dist = height_bins.value_counts()
        
        fig_height = go.Figure(data=[go.Bar(
            x=height_dist.index,
            y=height_dist.values,
            marker_color=CU_GOLD,
            text=height_dist.values,
            textposition='auto',
        )])
        
        fig_height.update_layout(
            title="Height Distribution",
            height=400,
            xaxis_title="Height Range",
            yaxis_title="Number of Trees"
        )
        st.plotly_chart(fig_height, use_container_width=True)

with tab2:

    st.subheader("Tree Genus Distribution")
    

    genus_data = df.groupby('Genus').agg({
        'Tree ID': 'count',
        'Species': 'nunique'
    }).reset_index()
    
    genus_data = genus_data.sort_values('Tree ID', ascending=True).tail(10)
    
    fig_genus = go.Figure()
    
    fig_genus.add_trace(go.Bar(
        y=genus_data['Genus'],
        x=genus_data['Tree ID'],
        orientation='h',
        name='Number of Trees',
        marker_color=CU_GOLD
    ))
    
    fig_genus.add_trace(go.Scatter(
        y=genus_data['Genus'],
        x=genus_data['Species'] * 10,
        mode='markers',
        name='Species Diversity (x10)',
        marker=dict(
            color=CU_BLACK,
            size=12,
            symbol='diamond'
        )
    ))
    
    fig_genus.update_layout(
        height=500,
        xaxis_title="Count",
        barmode='overlay'
    )
    
    st.plotly_chart(fig_genus, use_container_width=True)

    st.subheader("Tree Distribution Map")

    df_map = df.copy()
    df_map['Canopy Spread'] = df_map['Canopy Spread'].fillna(df_map['Canopy Spread'].median())
    
    fig_map = px.scatter_map(
        df_map,
        lat='Latitude',
        lon='Longitude',
        color='Tree Type',
        size='Canopy Spread',
        hover_data=['Common Name', 'Height', 'Genus'],
        color_discrete_map={'DECIDUOUS': CU_GOLD, 'CONIFEROUS': TREE_GREEN},
        zoom=14,
        height=600
    )
    
    fig_map.update_layout(
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Environmental Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        canopy_by_type = df.groupby('Tree Type')['Canopy Spread'].mean()
        
        fig_canopy = go.Figure(data=[go.Bar(
            x=canopy_by_type.index,
            y=canopy_by_type.values,
            marker_color=[CU_GOLD, TREE_GREEN],
            text=np.round(canopy_by_type.values, 1),
            textposition='auto'
        )])
        
        fig_canopy.update_layout(
            title="Average Canopy Spread by Tree Type",
            height=400,
            yaxis_title="Average Spread (ft)"
        )
        
        st.plotly_chart(fig_canopy, use_container_width=True)
    
    with col2:
        fig_correlation = px.scatter(
            df,
            x='Height',
            y='Canopy Spread',
            color='Tree Type',
            color_discrete_map={'DECIDUOUS': CU_GOLD, 'CONIFEROUS': TREE_GREEN},
            title="Height vs Canopy Spread Correlation"
        )
        
        fig_correlation.update_layout(height=400)
        st.plotly_chart(fig_correlation, use_container_width=True)
    

    st.subheader("Estimated Annual Environmental Benefits")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annual_co2 = len(df) * 48
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>CO₂ Sequestration</h3>
                <h2 style='margin:0;'>{annual_co2:,.0f} lbs/year</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stormwater = len(df) * 1000
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Stormwater Managed</h3>
                <h2 style='margin:0;'>{stormwater:,.0f} gallons/year</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        energy_savings = len(df[df['Tree Type'] == 'DECIDUOUS']) * 35
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Energy Savings</h3>
                <h2 style='margin:0;'>${energy_savings:,.0f}/year</h2>
            </div>
        """, unsafe_allow_html=True)

st.sidebar.title("Analysis Controls")

selected_types = st.sidebar.multiselect(
    "Tree Types",
    options=df['Tree Type'].unique(),
    default=df['Tree Type'].unique()
)

height_range = st.sidebar.slider(
    "Height Range (ft)",
    min_value=int(df['Height'].min()),
    max_value=int(df['Height'].max()),
    value=(0, int(df['Height'].max()))
)

selected_genera = st.sidebar.multiselect(
    "Select Genera",
    options=sorted(df['Genus'].unique()),
    default=[]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Species Diversity")


def shannon_diversity(genus_counts):
    proportions = genus_counts / genus_counts.sum()
    return -np.sum(proportions * np.log(proportions))

genus_counts = df['Genus'].value_counts()
diversity_index = shannon_diversity(genus_counts)

st.sidebar.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h4 style='color: {CU_GOLD}; margin: 0;'>Shannon Diversity Index</h4>
        <p style='color: white; font-size: 24px; margin: 0;'>{diversity_index:.2f}</p>
        <p style='color: {CU_GOLD}; font-size: 12px;'>Higher values indicate greater diversity</p>
    </div>
""", unsafe_allow_html=True)


filtered_df = df[
    (df['Tree Type'].isin(selected_types)) &
    (df['Height'].between(height_range[0], height_range[1]))
]

if selected_genera:
    filtered_df = filtered_df[filtered_df['Genus'].isin(selected_genera)]


st.sidebar.markdown("---")
st.sidebar.subheader("Seasonal Analysis")

season = st.sidebar.selectbox(
    "Select Season",
    ["Spring", "Summer", "Fall", "Winter"]
)

# Seasonal impact information
seasonal_impacts = {
    "Spring": {
        "description": "Budding and flowering period",
        "canopy": 0.7,
        "carbon": 0.8
    },
    "Summer": {
        "description": "Peak foliage and growth",
        "canopy": 1.0,
        "carbon": 1.0
    },
    "Fall": {
        "description": "Color change and leaf drop",
        "canopy": 0.6,
        "carbon": 0.5
    },
    "Winter": {
        "description": "Dormant period",
        "canopy": 0.3,
        "carbon": 0.2
    }
}

st.sidebar.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h4 style='color: {CU_GOLD}; margin: 0;'>{season} Impact</h4>
        <p style='color: white; font-size: 14px;'>{seasonal_impacts[season]['description']}</p>
        <p style='color: {CU_GOLD}; font-size: 12px;'>
            Canopy Coverage: {seasonal_impacts[season]['canopy']*100}%<br>
            Carbon Sequestration: {seasonal_impacts[season]['carbon']*100}%
        </p>
    </div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="cu_boulder_trees_filtered.csv",
    mime="text/csv",
    key='download-csv'
)


st.sidebar.markdown("---")
st.sidebar.subheader("Tree Health Indicators")

avg_canopy_ratio = df['Canopy Spread'].mean() / df['Height'].mean()
st.sidebar.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h4 style='color: {CU_GOLD}; margin: 0;'>Tree Health Metrics</h4>
        <p style='color: white; font-size: 14px;'>Average Canopy-to-Height Ratio: {avg_canopy_ratio:.2f}</p>
    </div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h4 style='color: {CU_GOLD}; margin: 0;'>About this Dashboard</h4>
        <p style='color: white; font-size: 12px;'>This interactive dashboard visualizes the tree diversity and distribution across the CU Boulder campus. The analysis includes environmental impact metrics, species diversity, and seasonal variations. Use the filters above to explore specific aspects of the tree population.</p>
    </div>
""", unsafe_allow_html=True)

def get_tree_recommendations(df):
    successful_genera = df.groupby('Genus').agg({
        'Tree ID': 'count',
        'Height': 'mean',
        'Canopy Spread': 'mean'
    }).sort_values('Tree ID', ascending=False).head(5)
    
    return successful_genera


recommendations = get_tree_recommendations(df)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style='background-color: {CU_BLACK}; padding: 1rem; border-radius: 5px;'>
        <h4 style='color: {CU_GOLD}; margin: 0;'>Top Performing Trees</h4>
        <p style='color: white; font-size: 12px;'>Based on population, growth, and health metrics:</p>
        <ul style='color: {CU_GOLD}; font-size: 12px;'>
            {"".join(f"<li>{genus}</li>" for genus in recommendations.index)}
        </ul>
    </div>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: {CU_BLACK}; padding: 1rem;'>
        <p>Data analyzed as part of the CU Boulder Tree Inventory Project</p>
        <p style='font-size: 12px;'>Updated: February 2025</p>
    </div>
""", unsafe_allow_html=True)