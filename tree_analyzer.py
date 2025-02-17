import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import joblib

class TreeAnalyzer:
    def __init__(self, data_path='processed_tree_data.csv'):
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self):
        """Prepare features for ML models"""
        # Encode categorical variables
        categorical_cols = ['Tree Type', 'Genus', 'Species', 'Common Name']
        for col in categorical_cols:
            if col in self.df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(self.df[col].fillna('Unknown'))
        
        # Create feature set for regression
        self.regression_features = [
            'Canopy Spread', 'Height', 'Tree Type_Encoded', 
            'Genus_Encoded', 'Height_Spread_Ratio',
            'Distance_From_Center'
        ]
        
        # Scale numeric features
        self.X_scaled = self.scaler.fit_transform(self.df[self.regression_features])
        
        return self.X_scaled

    def train_height_predictor(self):
        """Train a Random Forest model to predict tree height"""
        X = self.X_scaled
        y = self.df['Height']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.regression_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model
        joblib.dump(rf_model, 'tree_height_predictor.joblib')
        
        return {
            'model': rf_model,
            'mse': mse,
            'feature_importance': feature_importance,
            'test_score': rf_model.score(X_test, y_test)
        }

    def cluster_trees(self, n_clusters=5):
        """Perform clustering analysis on trees"""
        # Select features for clustering
        clustering_features = [
            'Height_Normalized', 'Canopy Spread_Normalized',
            'Height_Spread_Ratio', 'Distance_From_Center'
        ]
        
        X_cluster = self.df[clustering_features].fillna(0)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_cluster)
        
        # Calculate cluster characteristics
        cluster_stats = self.df.groupby('Cluster').agg({
            'Height': 'mean',
            'Canopy Spread': 'mean',
            'Tree Type': lambda x: x.value_counts().index[0],
            'Genus': lambda x: x.value_counts().index[0]
        }).round(2)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_cluster, self.df['Cluster'])
        
        return {
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'model': kmeans
        }

    def predict_growth_potential(self):
        """Predict potential growth based on current characteristics"""
        # Create growth potential features
        growth_features = [
            'Height', 'Canopy Spread', 'Height_Spread_Ratio',
            'Tree Type_Encoded', 'Genus_Encoded'
        ]
        
        X = self.df[growth_features].fillna(0)
        
        # Calculate current size ratio compared to max for species
        species_max_height = self.df.groupby('Species')['Height'].max()
        self.df['Growth_Potential'] = self.df.apply(
            lambda x: 1 - (x['Height'] / species_max_height[x['Species']]), 
            axis=1
        )
        
        # Train growth potential predictor
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.df['Growth_Potential'], 
            test_size=0.2, random_state=42
        )
        
        rf_growth = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_growth.fit(X_train, y_train)
        
        return {
            'model': rf_growth,
            'test_score': rf_growth.score(X_test, y_test),
            'feature_importance': pd.DataFrame({
                'feature': growth_features,
                'importance': rf_growth.feature_importances_
            }).sort_values('importance', ascending=False)
        }

    def analyze_environmental_impact(self):
        """Analyze and predict environmental impact"""
        # Calculate environmental impact score
        self.df['Environmental_Score'] = (
            self.df['Canopy Area'] * 0.4 +
            self.df['Height'] * 0.3 +
            self.df['Annual_CO2_Sequestration'] * 0.3
        )
        
        # Normalize score
        self.df['Environmental_Score'] = (
            self.df['Environmental_Score'] - self.df['Environmental_Score'].min()
        ) / (self.df['Environmental_Score'].max() - self.df['Environmental_Score'].min())
        
        # Analyze impact by species and location
        impact_by_species = self.df.groupby('Species')['Environmental_Score'].mean().sort_values(ascending=False)
        impact_by_location = self.df.groupby('Spatial_Cluster')['Environmental_Score'].mean()
        
        return {
            'impact_by_species': impact_by_species,
            'impact_by_location': impact_by_location,
            'total_impact': self.df['Environmental_Score'].sum()
        }

    def generate_recommendations(self):
        """Generate planting recommendations based on ML insights"""
        # Analyze successful species
        success_metrics = self.df.groupby('Species').agg({
            'Height': ['mean', 'std'],
            'Canopy Spread': ['mean', 'std'],
            'Environmental_Score': 'mean'
        }).round(2)
        
        # Find optimal planting locations
        location_success = self.df.groupby('Spatial_Cluster').agg({
            'Height': 'mean',
            'Environmental_Score': 'mean'
        }).round(2)
        
        # Generate recommendations
        recommendations = {
            'top_species': success_metrics.nlargest(5, ('Environmental_Score', 'mean')),
            'optimal_locations': location_success.nlargest(3, 'Environmental_Score'),
            'diversity_recommendations': self.analyze_diversity_needs()
        }
        
        return recommendations

    def analyze_diversity_needs(self):
        """Analyze species diversity and recommend improvements"""
        genus_distribution = self.df['Genus'].value_counts(normalize=True)
        species_distribution = self.df['Species'].value_counts(normalize=True)
        
        # Calculate diversity metrics
        diversity_metrics = {
            'genus_diversity': 1 - (genus_distribution ** 2).sum(),  # Simpson's diversity index
            'species_diversity': 1 - (species_distribution ** 2).sum(),
            'recommendations': []
        }
        
        # Generate recommendations based on diversity metrics
        if genus_distribution.iloc[0] > 0.2:
            diversity_metrics['recommendations'].append(
                f"Reduce dependency on {genus_distribution.index[0]} genus"
            )
        
        return diversity_metrics

# Example usage
if __name__ == "__main__":
    analyzer = TreeAnalyzer()
    analyzer.prepare_features()
    
    # Train models and generate insights
    height_prediction = analyzer.train_height_predictor()
    clustering_results = analyzer.cluster_trees()
    growth_prediction = analyzer.predict_growth_potential()
    environmental_impact = analyzer.analyze_environmental_impact()
    recommendations = analyzer.generate_recommendations()
    
    # Print summary of results
    print("\nModel Performance:")
    print(f"Height Prediction R2 Score: {height_prediction['test_score']:.3f}")
    print(f"Growth Prediction R2 Score: {growth_prediction['test_score']:.3f}")
    print(f"Clustering Silhouette Score: {clustering_results['silhouette_score']:.3f}")
    
    print("\nTop Feature Importance for Height Prediction:")
    print(height_prediction['feature_importance'].head())
    
    print("\nCluster Characteristics:")
    print(clustering_results['cluster_stats'])
    
    print("\nPlanting Recommendations:")
    for i, rec in enumerate(recommendations['diversity_recommendations']['recommendations'], 1):
        print(f"{i}. {rec}")