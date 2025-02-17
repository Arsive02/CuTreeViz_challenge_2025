import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tree_analyzer import TreeAnalyzer
import numpy as np

# Page config
st.set_page_config(
    page_title="CU Boulder Tree Analysis ML Dashboard",
    page_icon="🌳",
    layout="wide"
)

# Initialize ML analyzer
@st.cache_resource
def load_analyzer():
    analyzer = TreeAnalyzer()
    analyzer.prepare_features()
    # Train the height predictor and store it
    analyzer.height_prediction = analyzer.train_height_predictor()
    return analyzer

analyzer = load_analyzer()

# Main title
st.title("🌳 CU Boulder Tree Analysis with Machine Learning")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "ML Predictions", 
    "Clustering Analysis", 
    "Environmental Impact",
    "Growth Analysis"
])

with tab1:
    st.header("Machine Learning Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Height Prediction Model")
        st.metric(
            "Model Accuracy (R²)",
            f"{analyzer.height_prediction['test_score']:.3f}"
        )
        
        # Feature importance plot
        fig_importance = px.bar(
            analyzer.height_prediction['feature_importance'].head(),
            x='importance',
            y='feature',
            orientation='h',
            title="Top Features for Height Prediction",
            color_discrete_sequence=["#228B22"]  # ForestGreen color
        )
        st.plotly_chart(fig_importance)
    
    with col2:
        st.subheader("Predict Tree Height")
        # Create input form for prediction
        canopy = st.slider("Canopy Spread (ft)", 0, 100, 20)
        tree_type = st.selectbox("Tree Type", analyzer.df['Tree Type'].unique())
        genus = st.selectbox("Genus", analyzer.df['Genus'].unique())
        
        if st.button("Predict Height"):
            # Create input features
            input_data = np.zeros((1, len(analyzer.regression_features)))
            input_df = pd.DataFrame(input_data, columns=analyzer.regression_features)
            
            # Fill in known values
            input_df['Canopy Spread'] = canopy
            input_df['Tree Type_Encoded'] = analyzer.label_encoders['Tree Type'].transform([tree_type])[0]
            input_df['Genus_Encoded'] = analyzer.label_encoders['Genus'].transform([genus])[0]
            
            # Make prediction
            predicted_height = analyzer.height_prediction['model'].predict(input_df)[0]
            st.metric("Predicted Height", f"{predicted_height:.1f} ft")

with tab2:
    st.header("Tree Clustering Analysis")
    
    clustering_results = analyzer.cluster_trees()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Characteristics")
        st.dataframe(clustering_results['cluster_stats'])
        
        st.metric(
            "Clustering Quality (Silhouette Score)",
            f"{clustering_results['silhouette_score']:.3f}"
        )
    
    with col2:
        # Cluster visualization
        # Define CU Boulder and tree-appropriate colors
        CU_GOLD = '#CFB87C'
        CU_BLACK = '#000000'
        TREE_COLORS = {
            0: '#2E7D32',  # Dark green
            1: '#CFB87C',  # CU Gold
            2: '#4CAF50',  # Medium green
            3: '#81C784',  # Light green
            4: '#A5D6A7'   # Very light green
        }

        # Create the scatter plot with custom colors
        fig_clusters = px.scatter(
            analyzer.df,
            x='Height',
            y='Canopy Spread',
            color='Cluster',
            title="Tree Clusters by Height and Canopy Spread",
            color_discrete_map=TREE_COLORS
        )

        # Display the plot
        st.plotly_chart(fig_clusters, use_container_width=True)

with tab3:
    st.header("Environmental Impact Analysis")
    
    environmental_impact = analyzer.analyze_environmental_impact()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impact by Species")
        impact_df = pd.DataFrame({
            'Species': environmental_impact['impact_by_species'].index[:10],
            'Impact Score': environmental_impact['impact_by_species'].values[:10]
        })
        
        fig_impact = px.bar(
            impact_df,
            x='Species',
            y='Impact Score',
            title="Top 10 Species by Environmental Impact",
            color_discrete_sequence=["#228B22"]  # ForestGreen color
        )
        st.plotly_chart(fig_impact)
    
    with col2:
        st.subheader("Impact by Location")
        fig_map = px.scatter_mapbox(
            analyzer.df,
            lat='Latitude',
            lon='Longitude',
            color='Environmental_Score',
            size='Canopy Spread',
            title="Environmental Impact Distribution",
            mapbox_style="carto-positron",
            zoom=14,
            color_continuous_scale=px.colors.sequential.Greens
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map)

with tab4:
    st.header("Growth Analysis and Recommendations")
    
    growth_results = analyzer.predict_growth_potential()
    recommendations = analyzer.generate_recommendations()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Growth Potential Analysis")
        st.metric(
            "Growth Prediction Accuracy (R²)",
            f"{growth_results['test_score']:.3f}"
        )
        
        # Plot growth potential distribution
        fig_growth = px.histogram(
            analyzer.df,
            x='Growth_Potential',
            title="Distribution of Growth Potential",
            color_discrete_sequence=["#8B4513"]  # Dark brown color
        )
        st.plotly_chart(fig_growth)
    
    with col2:
        st.subheader("Planting Recommendations")
        st.write("Top Recommended Species:")
        st.dataframe(recommendations['top_species'])
        
        st.write("Diversity Recommendations:")
        for rec in recommendations['diversity_recommendations']['recommendations']:
            st.write(f"• {rec}")

# Sidebar for ML model controls
st.sidebar.header("ML Analysis Controls")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Height Prediction", "Growth Potential", "Environmental Impact", "Clustering"]
)

# Model parameters
st.sidebar.subheader("Model Parameters")

if selected_model == "Height Prediction":
    n_estimators = st.sidebar.slider("Number of Trees in Forest", 50, 200, 100)
    max_depth = st.sidebar.slider("Maximum Tree Depth", 3, 20, 10)
    
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            analyzer.height_prediction = analyzer.train_height_predictor()
            st.success("Model retrained successfully!")

elif selected_model == "Clustering":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    
    if st.sidebar.button("Recalculate Clusters"):
        with st.spinner("Recalculating clusters..."):
            clustering_results = analyzer.cluster_trees(n_clusters=n_clusters)
            st.success("Clusters recalculated!")

# Add feature importance analysis
st.sidebar.markdown("---")
st.sidebar.subheader("Feature Importance Analysis")

feature_importance = pd.DataFrame({
    'Feature': analyzer.regression_features,
    'Importance': analyzer.height_prediction['model'].feature_importances_
}).sort_values('Importance', ascending=False)

fig_feature_imp = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    title="Feature Importance",
    color_discrete_sequence=["#228B22"]  # ForestGreen color
)

st.sidebar.plotly_chart(fig_feature_imp, use_container_width=True)

# Add download section
st.markdown("---")
st.header("Download Analysis Results")

# Create a DataFrame with analysis results
analysis_results = pd.DataFrame({
    'Metric': [
        'Height Prediction R²',
        'Growth Prediction R²',
        'Clustering Silhouette Score',
        'Total Environmental Impact'
    ],
    'Value': [
        analyzer.height_prediction['test_score'],
        growth_results['test_score'],
        clustering_results['silhouette_score'],
        environmental_impact['total_impact']
    ]
})

# Add download button
st.download_button(
    label="Download Analysis Results",
    data=analysis_results.to_csv(index=False),
    file_name="tree_analysis_results.csv",
    mime="text/csv"
)

# Footer with metadata
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Analysis performed using Random Forest and K-Means clustering</p>
        <p>Model last updated: February 2025</p>
    </div>
""", unsafe_allow_html=True)