import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_tree_data(file_path):
    """
    Preprocess the CU Boulder tree dataset.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Preprocessed dataset
    dict: Additional computed features and metadata
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    # Fill missing species with 'Unknown'
    df['Species'] = df['Species'].fillna('Unknown')
    # Fill missing Common Name with Genus name
    df['Common Name'] = df['Common Name'].fillna(df['Genus'])
    # Fill missing Canopy Spread and Height with median values
    df['Canopy Spread'] = df['Canopy Spread'].fillna(df['Canopy Spread'].median())
    df['Height'] = df['Height'].fillna(df['Height'].median())
    
    # 2. Data cleaning and standardization
    # Remove any leading/trailing whitespace
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].str.strip()
    
    # 3. Feature engineering
    # Calculate tree size category
    df['Size Category'] = pd.cut(df['Height'],
                                bins=[0, 20, 40, 60, float('inf')],
                                labels=['Small', 'Medium', 'Large', 'Extra Large'])
    
    # Calculate canopy area
    df['Canopy Area'] = np.pi * (df['Canopy Spread']/2)**2
    
    # Calculate height to spread ratio
    df['Height_Spread_Ratio'] = df['Height'] / df['Canopy Spread']
    
    # 4. Location-based features
    # Calculate distance from campus center (approximate center point)
    campus_center_lat = df['Latitude'].mean()
    campus_center_lon = df['Longitude'].mean()
    
    df['Distance_From_Center'] = np.sqrt(
        (df['Latitude'] - campus_center_lat)**2 +
        (df['Longitude'] - campus_center_lon)**2
    )
    
    # 5. Diversity metrics
    # Calculate species diversity by genus
    genus_diversity = df.groupby('Genus')['Species'].nunique().to_dict()
    df['Genus_Species_Count'] = df['Genus'].map(genus_diversity)
    
    # 6. Environmental impact metrics
    # Approximate CO2 sequestration (based on size category)
    size_co2_map = {
        'Small': 20,
        'Medium': 48,
        'Large': 70,
        'Extra Large': 100
    }
    df['Annual_CO2_Sequestration'] = df['Size Category'].map(size_co2_map)
    
    # 7. Age approximation (rough estimate based on height and species)
    df['Age_Category'] = pd.cut(df['Height'],
                               bins=[0, 15, 30, 45, float('inf')],
                               labels=['Young', 'Mature', 'Old', 'Very Old'])
    
    # 8. Create normalized versions of numeric columns
    numeric_columns = ['Height', 'Canopy Spread', 'Canopy Area']
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[numeric_columns]),
        columns=[f"{col}_Normalized" for col in numeric_columns],
        index=df.index
    )
    df = pd.concat([df, df_normalized], axis=1)
    
    # 9. Calculate additional metadata
    metadata = {
        'total_trees': len(df),
        'unique_species': df['Species'].nunique(),
        'total_canopy_area': df['Canopy Area'].sum(),
        'average_height': df['Height'].mean(),
        'height_std': df['Height'].std(),
        'genus_distribution': df['Genus'].value_counts().to_dict(),
        'type_distribution': df['Tree Type'].value_counts().to_dict(),
        'size_distribution': df['Size Category'].value_counts().to_dict(),
        'campus_center': {
            'latitude': campus_center_lat,
            'longitude': campus_center_lon
        }
    }
    
    # 10. Create seasonal impact estimates
    df['Spring_Canopy'] = df.apply(lambda x: 
        x['Canopy Area'] * (0.7 if x['Tree Type'] == 'DECIDUOUS' else 0.9), axis=1)
    df['Summer_Canopy'] = df['Canopy Area']
    df['Fall_Canopy'] = df.apply(lambda x: 
        x['Canopy Area'] * (0.6 if x['Tree Type'] == 'DECIDUOUS' else 0.9), axis=1)
    df['Winter_Canopy'] = df.apply(lambda x: 
        x['Canopy Area'] * (0.1 if x['Tree Type'] == 'DECIDUOUS' else 0.9), axis=1)
    
    # 11. Data validation
    # Remove any invalid coordinates
    df = df[
        (df['Latitude'].between(39.5, 40.5)) & 
        (df['Longitude'].between(-106, -105))
    ]
    
    # Remove any negative heights or spreads
    df = df[
        (df['Height'] >= 0) &
        (df['Canopy Spread'] >= 0)
    ]
    
    # 12. Clustering for spatial distribution
    from sklearn.cluster import KMeans
    
    # Perform spatial clustering
    coords = df[['Latitude', 'Longitude']].copy()
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Spatial_Cluster'] = kmeans.fit_predict(coords)
    
    return df, metadata

def generate_summary_report(df, metadata):
    """
    Generate a summary report of the preprocessed data.
    """
    report = {
        'Basic Statistics': {
            'Total Trees': metadata['total_trees'],
            'Unique Species': metadata['unique_species'],
            'Average Height': f"{metadata['average_height']:.2f} ft",
            'Total Canopy Coverage': f"{metadata['total_canopy_area']:.2f} sq ft"
        },
        'Tree Types': metadata['type_distribution'],
        'Size Distribution': metadata['size_distribution'],
        'Top 5 Genera': dict(sorted(
            metadata['genus_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
    }
    
    return report

# Example usage:
if __name__ == "__main__":
    # Process the data
    processed_df, metadata = preprocess_tree_data('data/Data_Viz_Challenge_2025-UCB_Trees.csv')
    
    # Generate summary report
    report = generate_summary_report(processed_df, metadata)
    
    # Save processed data
    processed_df.to_csv('processed_tree_data.csv', index=False)
    
    # Print summary report
    print("\nData Processing Summary:")
    for section, data in report.items():
        print(f"\n{section}:")
        for key, value in data.items():
            print(f"{key}: {value}")